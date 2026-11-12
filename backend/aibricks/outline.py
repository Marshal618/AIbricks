"""Outline things in photos: mark the exact shape of each thing, then train a model that outlines them itself.

Outlining by hand is slow, so a click does most of it. MobileSAM (Zhang et al., 2023, Apache 2.0), a
small version of Meta's Segment Anything, finds the shape around the place you click. It runs on this
computer with ONNX Runtime; its two files, about 45 MB, are downloaded once and checked against fixed
hashes. More clicks add parts to the shape or take parts away.

The model that learns your outlines is LRASPP (Howard et al., 2019) with a MobileNetV3 backbone,
pretrained by torchvision to outline 20 everyday kinds of thing. Its last layers are replaced to
outline your own things, and it's fine-tuned on your outlined photos. It's small enough for web pages
and a Raspberry Pi.

Outlines are stored with the boxes of “find things” datasets (see finder.py): each shape carries its
corners and the box around them, so counting, versions and renaming work the same for both.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import codegen, datasets, finder, labelling, training, versions
from .training import TrainingError

MIN_PHOTOS = 10
MIN_OUTLINES = 5
SUGGESTED_OUTLINES = 30
MAX_CLICKS = 20
SAM_REVISION = "0d3b403339b4674a82493d5e97964dd78089ddc8"
SAM_FILES = {
    "encoder": ("mobile_sam_image_encoder.onnx", "580f5fb648ea1062c0aabc26217aed56921985f03f0cbbd852bba81d760cc749"),
    "decoder": ("sam_mask_decoder_single.onnx", "93915fc7c993ab9d59ab8c9ccd3bce37f7509c81ab4150a74abd4d2abbd8570d"),
}
SAM_URL = f"https://huggingface.co/Acly/MobileSAM/resolve/{SAM_REVISION}/"
SAM_SIDE = 1024
REVIEW_FOLDER = "outline-review"  # in the run folder, moved there by training._review_outline

_sam_lock = threading.Lock()
_sam: dict | None = None
_embeddings: OrderedDict[tuple[str, str, int], tuple[np.ndarray, int, int]] = OrderedDict()
_EMBEDDINGS_KEPT = 4


def is_outline(meta: dict) -> bool:
    return meta.get("kind") == "image" and meta.get("goal") == "outline"


# --------------------------------------------------------------------- click to outline

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sam_file(name: str, sha: str) -> Path:
    folder = training._cache() / "mobilesam"
    path = folder / name
    if path.is_file() and _sha256(path) == sha:
        return path
    folder.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(SAM_URL + name, timeout=120, context=context) as reply, open(partial, "wb") as file:
            shutil.copyfileobj(reply, file)
    except OSError as problem:
        partial.unlink(missing_ok=True)
        raise TrainingError(503, f"The outlining helper couldn't be downloaded. It needs the internet once, about 45 MB. ({type(problem).__name__})") from None
    if _sha256(partial) != sha:
        partial.unlink()
        raise TrainingError(502, "The outlining helper that downloaded isn't the expected file, so it wasn't used.")
    os.replace(partial, path)
    return path


def _sessions() -> dict:
    global _sam
    if _sam is None:
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max((os.cpu_count() or 2) - 1, 1)
        options.log_severity_level = 3
        _sam = {role: onnxruntime.InferenceSession(str(_sam_file(*SAM_FILES[role])), options, providers=["CPUExecutionProvider"])
                for role in SAM_FILES}
    return _sam


def _embedding(dataset: str, image_id: str) -> tuple[np.ndarray, int, int]:
    """MobileSAM's reading of a photo, kept for the last few photos so each further click is quick."""
    from PIL import Image, ImageOps
    path = labelling.image_file(image_id, dataset)
    key = (dataset, image_id, path.stat().st_mtime_ns)
    if key in _embeddings:
        _embeddings.move_to_end(key)
        return _embeddings[key]
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    scale = SAM_SIDE / max(image.size)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    pixels = np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.float32)
    embedding = _sessions()["encoder"].run(None, {"input_image": pixels})[0]
    _embeddings[key] = (embedding, width, height)
    while len(_embeddings) > _EMBEDDINGS_KEPT:
        _embeddings.popitem(last=False)
    return _embeddings[key]


def _polygon(mask: np.ndarray, first: tuple[float, float]) -> list[list[float]]:
    """The outline of the part of the mask that holds the first click, or its biggest part."""
    import cv2
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    inside = [c for c in contours if cv2.pointPolygonTest(c, (first[0] * width, first[1] * height), False) >= 0]
    contour = max(inside or contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return []
    detail = 0.0015
    while True:
        corners = cv2.approxPolyDP(contour, max(1.0, detail * cv2.arcLength(contour, True)), True).reshape(-1, 2)
        if len(corners) <= 200:
            break
        detail *= 1.5
    return [[round(float(x) / width, 5), round(float(y) / height, 5)] for x, y in corners]


def click(dataset: str, image_id: str, clicks: list[dict]) -> dict:
    """The shape around the places clicked. Each click is {x, y, inside}, with x and y as shares of the
    photo's width and height; inside is False for a part to leave out."""
    meta = datasets._read(dataset)
    if not finder.has_shapes(meta):
        raise TrainingError(422, "Click to outline works in “outline things” and “find things” datasets.")
    if not 1 <= len(clicks) <= MAX_CLICKS:
        raise TrainingError(422, f"Click between 1 and {MAX_CLICKS} times.")
    try:
        spots = [(min(max(float(c["x"]), 0.0), 1.0), min(max(float(c["y"]), 0.0), 1.0), bool(c.get("inside", True))) for c in clicks]
    except (KeyError, TypeError, ValueError):
        raise TrainingError(422, "Each click needs a place.") from None
    if not any(inside for _, _, inside in spots):
        raise TrainingError(422, "Click on the thing itself first.")
    with _sam_lock:
        embedding, width, height = _embedding(dataset, image_id)
        coords = np.array([[[x * width, y * height] for x, y, _ in spots] + [[0.0, 0.0]]], dtype=np.float32)
        labels = np.array([[1.0 if inside else 0.0 for _, _, inside in spots] + [-1.0]], dtype=np.float32)
        masks, scores, _ = _sessions()["decoder"].run(None, {
            "image_embeddings": embedding, "point_coords": coords, "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32), "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array([height, width], dtype=np.float32),
        })
    first = next((x, y) for x, y, inside in spots if inside)
    points = _polygon(masks[0, 0] > 0, first)
    if not points:
        return {"points": [], "box": None, "score": 0.0}
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    box = {"x": min(xs), "y": min(ys), "w": round(max(xs) - min(xs), 5), "h": round(max(ys) - min(ys), 5)}
    return {"points": points, "box": box, "score": round(float(scores.reshape(-1)[0]), 4)}


# --------------------------------------------------------------------- the project

# The same colours as the app, so a thing looks the same in the labeller, the results and downloads.
COLOURS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#c2255c", "#0c8599", "#f08c00", "#5c7cfa", "#66a80f"]

_MODEL_PY = '''"""Outlines things in photos: it decides, for every spot in the photo, which thing is there, if any.

It's LRASPP (Howard et al., 2019) with a MobileNetV3 backbone, pretrained by torchvision to outline 20
everyday kinds of thing. Its last layers are replaced to outline your own things, then the whole
model is fine-tuned on your outlined photos. Photos are squashed to SIZE by SIZE pixels first.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large
from torchvision.transforms.functional import normalize, to_tensor

SIZE = 320
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
COLOURS = __COLOURS__


def best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build(categories, pretrained=True):
    """The model, with one output for each thing plus one for background."""
    classes = len(categories) + 1
    model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None, weights_backbone=None)
    head = model.classifier
    head.low_classifier = nn.Conv2d(head.low_classifier.in_channels, classes, 1)
    head.high_classifier = nn.Conv2d(head.high_classifier.in_channels, classes, 1)
    return model


def learning(model):
    """Training mode, with every layer keeping the averages it learned when it was pretrained, as a few
    photos at a time would make them jump around."""
    model.train()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()
    return model


def load_image(source):
    """A photo the right way up."""
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def pixels(image):
    """The photo squashed to SIZE by SIZE, as numbers from 0 to 1."""
    return to_tensor(image.resize((SIZE, SIZE), Image.BILINEAR))


def prepare(batch):
    return normalize(batch, MEAN, STD)


def target(shapes, categories):
    """Which thing is at each spot: 0 for nothing, 1 for the first thing and so on. Later outlines lie on top."""
    canvas = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(canvas)
    for shape in shapes:
        draw.polygon([(x * SIZE, y * SIZE) for x, y in shape["points"]], fill=categories.index(shape["category"]) + 1)
    return torch.from_numpy(np.array(canvas, dtype=np.int64))


def paint(labels):
    """A see-through picture with each thing in its colour, to lay over the photo."""
    labels = np.asarray(labels)
    picture = np.zeros((*labels.shape, 4), dtype=np.uint8)
    for index, colour in enumerate(COLOURS * 3):
        spots = labels == index + 1
        if spots.any():
            picture[spots] = [int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16), 150]
    buffer = io.BytesIO()
    Image.fromarray(picture, "RGBA").save(buffer, "PNG")
    return buffer.getvalue()


def areas(labels, categories):
    """How much of the photo each thing covers, as a share."""
    labels = np.asarray(labels)
    return {name: round(float((labels == index + 1).mean()), 5) for index, name in enumerate(categories)}


class Outliner:
    def __init__(self, folder="model", device="cpu"):
        folder = Path(folder)
        self.settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        self.categories = self.settings["categories"]
        self.device = device
        self.model = build(self.categories, pretrained=False)
        self.model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        self.model.to(device).eval()

    def labels(self, image):
        with torch.no_grad():
            return self.model(prepare(pixels(image)[None]).to(self.device))["out"][0].argmax(0).cpu().numpy()

    def outline(self, image):
        """{"areas", "answers", "mask"}: how much each thing covers, biggest first, and the outlines as a PNG."""
        labels = self.labels(image)
        covered = areas(labels, self.categories)
        ranked = sorted(covered.items(), key=lambda item: -item[1])
        return {"areas": covered, "answers": [{"category": name, "chance": share} for name, share in ranked],
                "mask": base64.b64encode(paint(labels)).decode()}
'''

_TRAIN_PY = '''"""Teaches the model to outline your things. The photos are in data/images/ and their outlines in
data/boxes.json, as {"photo.jpg": [{"category": "...", "points": [[0.1, 0.2], [0.3, 0.2], [0.2, 0.4]]}]},
with corners as shares of the photo's width and height. Then run: python train.py"""
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import best_device, build, learning, load_image, paint, pixels, prepare, target

DATA = Path("data")
HOLD_BACK = 0.2
GOOD = 0.7  # how closely an outline must match (intersection over union) to count as right


def fingerprint(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def overlap(found, real, index):
    """Intersection over union of one thing's spots, or None when it's in neither."""
    a, b = found == index, real == index
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else None


def main():
    saved = json.loads((DATA / "boxes.json").read_text(encoding="utf-8"))
    categories, shapes = saved["categories"], saved["photos"]
    names = sorted(shapes, key=fingerprint)
    if len(names) < 5:
        sys.exit("Outline at least 5 photos.")
    held = max(1, int(len(names) * HOLD_BACK))
    check_names, train_names = names[:held], names[held:]
    device = best_device()
    try:
        model = build(categories).to(device)
    except Exception as problem:
        sys.exit(f"The pretrained model couldn't be loaded. The first run downloads it, about 13 MB. ({type(problem).__name__})")

    def load(group):
        return torch.stack([pixels(load_image(DATA / "images" / name)) for name in group]), torch.stack([target(shapes[name], categories) for name in group])

    x, y = load(train_names)
    cx, cy = load(check_names)
    # Rare things count for more, so a small thing isn't drowned out by the background around it.
    share = torch.bincount(y.reshape(-1), minlength=len(categories) + 1).float() / y.numel()
    weights = (1 / share.clamp(min=1e-4).sqrt())
    weights = (weights / weights[0]).clamp(max=10.0).to(device)
    epochs = int(os.environ.get("EPOCHS", "0")) or (30 if len(train_names) < 60 else 20)
    batch = 4
    steps_per_epoch = math.ceil(len(train_names) / batch)
    total_steps = epochs * steps_per_epoch
    print(f"Training on {len(train_names)} samples, checking on {len(check_names)}, using {device}.", flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("LEARNING_RATE", "0")) or 1e-3, weight_decay=1e-4)
    warmup = min(steps_per_epoch, 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
    loss_of = nn.CrossEntropyLoss(weight=weights)
    random.seed(0)
    torch.manual_seed(0)
    best, step = None, 0
    partial = Path("model.partial")

    def predict(images):
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(prepare(images[i:i + 8]).to(device))["out"].cpu() for i in range(0, len(images), 8)])
        return out

    for epoch in range(1, epochs + 1):
        learning(model)
        order = torch.randperm(len(x))
        running = 0.0
        for start in range(0, len(x), batch):
            pick = order[start:start + batch]
            images, goals = x[pick].clone(), y[pick].clone()
            # Mirrored and a little lighter or darker, so it learns shapes rather than one exact photo.
            if random.random() < 0.5:
                images, goals = images.flip(-1), goals.flip(-1)
            images = (images * torch.empty(len(images), 1, 1, 1).uniform_(0.8, 1.2)).clamp(0, 1)
            loss = loss_of(model(prepare(images).to(device))["out"], goals.to(device))
            if not torch.isfinite(loss):
                sys.exit("The loss became invalid.")
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            running += float(loss) * len(images)
            step += 1
            print(f"Step {step}/{total_steps}", flush=True)
        out = predict(cx)
        check_loss = float(loss_of(out, cy))
        found = out.argmax(1)
        scores = [overlap(found, cy, index + 1) for index in range(len(categories))]
        scores = [score for score in scores if score is not None]
        accuracy = sum(scores) / len(scores) if scores else 1.0
        print(f"Epoch {epoch}/{epochs}  training loss {running / len(x):.4f}  check loss {check_loss:.4f}  check accuracy {100 * accuracy:.1f}%", flush=True)
        if best is None or check_loss < best:
            # The version with the lowest check loss is kept, as with every model AI Bricks trains.
            best = check_loss
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir()
            torch.save(model.state_dict(), partial / "model.pt")
            (partial / "settings.json").write_text(json.dumps({"categories": categories}), encoding="utf-8")
    shutil.rmtree("model", ignore_errors=True)
    os.replace(partial, "model")

    model.load_state_dict(torch.load("model/model.pt", map_location="cpu", weights_only=True))
    model.to(device)
    found = predict(cx).argmax(1)
    review = DATA / "review"
    shutil.rmtree(review, ignore_errors=True)
    review.mkdir()
    rows = []
    per = {name: {"shared": 0, "union": 0, "photos": 0, "good": 0} for name in categories}
    for number, name in enumerate(check_names):
        ious, missed, extra = {}, [], []
        for index, category in enumerate(categories):
            a, b = found[number] == index + 1, cy[number] == index + 1
            shared, union = int((a & b).sum()), int((a | b).sum())
            per[category]["shared"] += shared
            per[category]["union"] += union
            if not union:
                continue
            ious[category] = round(shared / union, 4)
            if b.any():
                per[category]["photos"] += 1
                per[category]["good"] += ious[category] >= GOOD
                if ious[category] < 0.5:
                    missed.append(category)
            else:
                extra.append(category)
        score = sum(ious.values()) / len(ious) if ious else 1.0
        (review / f"{number}.png").write_bytes(paint(found[number].numpy()))
        rows.append({"file": name, "score": round(score, 4), "ious": ious, "missed": missed, "extra": extra,
                     "real": shapes[name], "mask": number})
    scores = {name: value["shared"] / value["union"] for name, value in per.items() if value["union"]}
    metrics = {
        "task": "outline", "photos": len(rows), "good": GOOD,
        "meanIoU": sum(scores.values()) / len(scores) if scores else None,
        "pixelAccuracy": float((found == cy).float().mean()),
        "allRight": sum(row["score"] >= GOOD for row in rows),
        "perCategory": {name: {"iou": scores.get(name), "photos": value["photos"], "good": value["good"]} for name, value in per.items()},
    }
    with open(DATA / "review.json", "w", encoding="utf-8") as file:
        json.dump(rows, file)
    with open(DATA / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''"""Outlines things in photos. Train first, then run: python predict.py photo.jpg

Add --save to write each photo's outlines next to it, as photo.outlines.png."""
import base64
import json
import sys
from pathlib import Path

from model import Outliner, load_image


def main():
    paths = [item for item in sys.argv[1:] if not item.startswith("--")]
    if not paths:
        sys.exit("Name at least one photo, for example: python predict.py photo.jpg")
    outliner = Outliner()
    results = []
    for path in paths:
        result = outliner.outline(load_image(path))
        if "--save" in sys.argv:
            Path(path).with_suffix(".outlines.png").write_bytes(base64.b64decode(result["mask"]))
        del result["mask"]
        results.append({"file": path, **result})
    if "--json" in sys.argv:
        print(json.dumps(results))
        return
    for result in results:
        covered = ", ".join(f"{name} {share:.0%}" for name, share in result["areas"].items() if share >= 0.005) or "nothing"
        print(f"{result['file']}: {covered}")


if __name__ == "__main__":
    main()
'''

_README = """# {name}

Outlines things in photos, trained in AI Bricks: for every spot in a photo it decides which thing is
there, if any. It's LRASPP (Howard et al., 2019) with a MobileNetV3 backbone, pretrained by torchvision
and fine-tuned on your outlined photos. Your photos aren't included in this download.

## Use it

```
pip install -r requirements.txt
python predict.py photo.jpg          # how much of the photo each thing covers
python predict.py photo.jpg --save   # also writes photo.outlines.png
```

## Train it again

Put photos in `data/images/` and their outlines in `data/boxes.json`, as described at the top of
`train.py`, then run `python train.py`. The first run downloads the pretrained model, about 13 MB.
"""

LIVE_RUNNER = """
import base64, contextlib, io, json, sys
sys.path.insert(0, '.')
with contextlib.redirect_stdout(sys.stderr):
    from model import Outliner, load_image, best_device
    outliner = Outliner(device=best_device())
print('ready', flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        print(json.dumps(outliner.outline(load_image(io.BytesIO(base64.b64decode(request['data']))))), flush=True)
    except Exception as problem:
        print(json.dumps({'error': f'{type(problem).__name__}: {problem}'[:300]}), flush=True)
"""


def project_files(name: str) -> dict[str, str]:
    return {
        "model.py": _MODEL_PY.replace("__COLOURS__", json.dumps(COLOURS)), "train.py": _TRAIN_PY, "predict.py": _PREDICT_PY,
        "README.md": _README.format(name=name), "requirements.txt": "torch>=2.2\ntorchvision>=0.17\npillow\nnumpy\n",
    }


def readiness(meta: dict) -> list[str]:
    amounts, done, _ = finder.counts(meta)
    problems = []
    if not meta["categories"]:
        problems.append("Name at least one thing to outline.")
    if done < MIN_PHOTOS:
        problems.append(f"Outline at least {MIN_PHOTOS} photos. {done} {'is' if done == 1 else 'are'} done.")
    short = [name for name, amount in amounts.items() if amount < MIN_OUTLINES]
    if short:
        problems.append(f"Outline each thing at least {MIN_OUTLINES} times. Short: {', '.join(short)}.")
    return problems


def start(dataset: str) -> training.Job:
    with training._lock:
        meta = datasets._read(dataset)
        if not is_outline(meta):
            raise TrainingError(422, "This trainer is for “outline things in photos” datasets.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        problems = readiness(meta)
        if problems:
            raise TrainingError(422, " ".join(problems))
        amounts, _, _ = finder.counts(meta)
    files = project_files(meta["name"])
    return training.start_run(meta["name"], codegen.slug(meta["name"]), codegen.build_hash(files), files,
                              amounts, dataset, versions.freeze, "outline", 0.0)


def _outline_job(job_id: str) -> training.Job:
    job = training.stored(job_id)
    if job.task != "outline" or not job.hasModel:
        raise TrainingError(404, "That model doesn't outline things in photos.")
    return job


def details(job_id: str) -> dict:
    job = _outline_job(job_id)
    settings = json.loads((training._folder(job) / "model" / "settings.json").read_text(encoding="utf-8"))
    return {"categories": settings["categories"], "metrics": job.metrics, "colours": COLOURS}


def review_image(job_id: str, number: int) -> Path:
    """The outlines the model drew on one held-back photo."""
    job = _outline_job(job_id)
    path = training._folder(job) / REVIEW_FOLDER / f"{int(number)}.png"
    if number < 0 or not path.is_file():
        raise TrainingError(404, "That picture isn't kept.")
    return path
