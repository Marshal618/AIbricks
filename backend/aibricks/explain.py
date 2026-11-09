"""Shows what a trained model paid attention to.

Photos use Grad-CAM (Selvaraju et al., 2017): the last layer that still keeps the photo's layout is
weighted by how much each of its channels pushes the answer up. Models with no such layer, such as
vision transformers, use the gradient of the answer with respect to each pixel instead.

Text uses leave-one-word-out: the model is asked again with each word removed, and a word matters by
as much as the top answer's chance falls without it (Li et al., 2016).

Both show where the model looked, which is evidence about its reasoning, not proof of it.
"""
from __future__ import annotations

import json
import subprocess

MAX_WORDS = 80
GRID_LIMIT = 32  # Grad-CAM cells across, at most
GRADIENT_GRID = 14  # Pixel gradients are noisy, so they're averaged over coarser cells.

# Runs inside the model's folder, kept loaded between frames. One JSON request per line in, one out.
LIVE_RUNNER = """
import base64, io, json, sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from predict import load_trained_model
from data import _image_transform, _load_image
from model import CLASS_NAMES

model = load_trained_model()
for parameter in model.parameters():
    parameter.requires_grad_(False)
transform = _image_transform()
chosen = []
print('ready', flush=True)


def pick_layer(image):
    seen = []
    def note(module, inputs, output):
        if isinstance(output, torch.Tensor):
            seen.append((module, output.dim(), tuple(output.shape)))
    handles = [module.register_forward_hook(note) for module in model.modules() if module is not model]
    try:
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    last = None
    for index, (module, dims, shape) in enumerate(seen):
        if dims == 4 and shape[2] > 1 and shape[3] > 1:
            last = index
    # A transformer after the last image-shaped layer means that layer is only the patch cutter.
    if last is None or any(dims == 3 for _, dims, _ in seen[last + 1:]):
        return None
    return seen[last][0]


def ranked(logits):
    chances = logits.softmax(dim=1)[0].tolist()
    return sorted(({'category': name, 'chance': chance} for name, chance in zip(CLASS_NAMES, chances)), key=lambda item: -item['chance'])


def heatmap(image):
    if not chosen:
        chosen.append(pick_layer(image))
    layer = chosen[0]
    image = image.clone().requires_grad_(True)
    kept = {}
    handle = layer.register_forward_hook(lambda module, inputs, output: kept.__setitem__('out', output)) if layer is not None else None
    try:
        logits = model(image)
    finally:
        if handle is not None:
            handle.remove()
    top = int(logits[0].argmax())
    method = 'gradcam'
    try:
        if layer is None:
            raise LookupError
        activation = kept['out']
        (gradient,) = torch.autograd.grad(logits[0, top], activation)
        cam = torch.relu((gradient.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1))[0]
        grid = __GRID_LIMIT__
    except (LookupError, RuntimeError):
        method = 'gradients'
        logits = model(image)
        (gradient,) = torch.autograd.grad(logits[0, top], image)
        cam = gradient.abs().sum(dim=1)[0]
        grid = __GRADIENT_GRID__
    cam = cam.detach()[None, None]
    height, width = cam.shape[-2:]
    if max(height, width) > grid:
        cam = F.adaptive_avg_pool2d(cam, (min(height, grid), min(width, grid)))
    cam = cam[0, 0]
    peak = float(cam.max())
    values = (cam / peak if peak > 0 else torch.zeros_like(cam)).flatten().tolist()
    return ranked(logits.detach()), {
        'width': int(cam.shape[1]), 'height': int(cam.shape[0]), 'values': [round(value, 3) for value in values],
        'method': method, 'category': CLASS_NAMES[top],
    }


for line in sys.stdin:
    try:
        request = json.loads(line)
        image = _load_image(io.BytesIO(base64.b64decode(request['image'])), transform).unsqueeze(0)
        if request.get('explain'):
            answers, picture = heatmap(image)
        else:
            with torch.no_grad():
                answers, picture = ranked(model(image)), None
        print(json.dumps({'answers': answers, 'heatmap': picture}), flush=True)
    except Exception as problem:
        print(json.dumps({'error': f'{type(problem).__name__}: {problem}'[:300]}), flush=True)
""".replace("__GRID_LIMIT__", str(GRID_LIMIT)).replace("__GRADIENT_GRID__", str(GRADIENT_GRID))


_TEXT_RUNNER = """
import json, re, sys
sys.path.insert(0, '.')
from model import answers, load

text = sys.stdin.read()
found = [(match.start(), match.end()) for match in re.finditer(r'\\S+', text)]
spans = found[:__MAX_WORDS__]
encoder, head, labels = load()
variants = [text] + [text[:start] + text[end:] for start, end in spans]
results = []
for first in range(0, len(variants), 32):
    results += answers(encoder, head, labels, variants[first:first + 32])
full = results[0]
top = full[0]['category']
def chance(result):
    return next(item['chance'] for item in result if item['category'] == top)
base = chance(full)
words = [{'start': start, 'end': end, 'weight': round(base - chance(result), 4)} for (start, end), result in zip(spans, results[1:])]
print(json.dumps({'answers': full, 'category': top, 'words': words, 'wordCount': len(found)}))
""".replace("__MAX_WORDS__", str(MAX_WORDS))


def explain_text(job_id: str, text: str) -> dict:
    """The model's answers, and how much each word moved the top one."""
    from . import textmodels, training

    job = textmodels._text_job(job_id)
    if job.status == "running":
        raise training.TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise training.TrainingError(409, "This model's file is no longer saved.")
    folder = training._folder(job)
    try:
        with training._predictions:
            done = subprocess.run(
                [training.sys.executable, "-I", "-c", _TEXT_RUNNER], cwd=folder,
                env=training._environment(folder) | {"OMP_NUM_THREADS": "2"},
                input=text, capture_output=True, text=True, errors="replace", timeout=training.PREDICT_LIMIT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        raise training.TrainingError(504, "The model took too long to answer.") from None
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise training.TrainingError(500, lines[-1][:600] if lines else "The model could not read this text.")
    return json.loads(done.stdout.strip().splitlines()[-1])
