"""Find where it fails: groups of held-back photos a model gets wrong far more often, named in words.

It follows the idea of Domino (Eyuboglu et al., 2022). CLIP, an open model that understands photos and
short phrases together, describes every held-back photo. The photos are grouped by what they look like,
with the model's mistakes counted a little so that groups of mistakes stand out. A group that's wrong
far more often than the rest is named by the everyday phrase that fits it best compared with all the
other photos. Everything runs on this computer.

Only held-back photos are used, because they're the ones the model never learned from.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

from . import training
from .training import TrainingError

CLIP_MODEL = "openai/clip-vit-base-patch32"
MIN_MISTAKES = 6
RESULT_FILE = "slices.json"
LIMIT_SECONDS = 15 * 60

# Everyday descriptions to name groups with, and what to do about each kind.
PHRASES = {
    "light": ["a dark photo", "a photo taken at night", "a very bright photo", "an overexposed photo", "a photo with harsh shadows", "a backlit photo"],
    "blur": ["a blurry photo", "an out of focus photo", "a photo with motion blur", "a grainy low quality photo"],
    "zoom": ["a close-up photo", "a photo taken from far away", "a photo where the object is small", "a photo where the object is cut off"],
    "angle": ["a photo taken from above", "a photo taken from below", "a photo of something tilted or upside down", "a photo of the object from the side", "a photo of the object from behind"],
    "background": ["a photo with a cluttered background", "a photo with a plain background", "an outdoor photo", "an indoor photo", "a photo with people in it", "a photo with several objects"],
    "colour": ["a black and white photo", "a very colourful photo", "a photo with a strong colour tint"],
    "other": ["a drawing or illustration", "a photo of a screen", "a photo with text in it", "a partly hidden object"],
}
ADVICE = {
    "light": ("Add photos in this light, and turn on “Vary training photos” with light and colour.", "vary"),
    "blur": ("Add some blurry or moving photos, or hold the camera steadier when using it.", "photos"),
    "zoom": ("Add photos at this distance, and turn on “Vary training photos” with crop and zoom.", "vary"),
    "angle": ("Add photos from this angle. If things can appear turned, turn on “Turn” in “Vary training photos”.", "vary"),
    "background": ("Add photos with backgrounds like these, so it learns the object rather than what's behind it.", "photos"),
    "colour": ("Add photos like these, and turn on light and colour in “Vary training photos”.", "vary"),
    "other": ("Add photos like these.", "photos"),
}

_lock = threading.Lock()
_jobs: dict[str, dict] = {}

_SCRIPT = """
import json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.utils import logging
logging.set_verbosity_error()
request = json.loads(sys.stdin.read())
try:
    model = CLIPModel.from_pretrained(request['model'], local_files_only=True)
    processor = CLIPProcessor.from_pretrained(request['model'], local_files_only=True)
except OSError:
    model = CLIPModel.from_pretrained(request['model'])
    processor = CLIPProcessor.from_pretrained(request['model'])
model.eval()
print('step', flush=True)
rows = []
with torch.no_grad():
    paths = request['paths']
    for start in range(0, len(paths), 32):
        images = [Image.open(path).convert('RGB') for path in paths[start:start + 32]]
        pixels = processor(images=images, return_tensors='pt')['pixel_values']
        rows.append(model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output))
    words = processor(text=request['phrases'], return_tensors='pt', padding=True)
    text = model.text_projection(model.text_model(input_ids=words['input_ids'], attention_mask=words['attention_mask']).pooler_output)
pictures = torch.nn.functional.normalize(torch.cat(rows), dim=1).numpy()
text = torch.nn.functional.normalize(text, dim=1).numpy()
wrong = np.array(request['wrong'], dtype=float)
from sklearn.cluster import KMeans
count = int(min(8, max(3, round(len(paths) / 6))))
# Mistakes count a little, so that groups of mistakes stand out without ignoring what photos look like.
groups = KMeans(count, n_init=10, random_state=0).fit_predict(np.hstack([pictures, wrong[:, None] * 0.35]))
found = []
for group in range(count):
    inside = groups == group
    if inside.all() or not inside.any():
        continue
    difference = pictures[inside].mean(0) - pictures[~inside].mean(0)
    ranked = np.argsort(-(text @ difference))
    found.append({'members': np.flatnonzero(inside).tolist(), 'phrases': [int(i) for i in ranked[:3]]})
print(json.dumps(found), flush=True)
"""


def _job_for(job_id: str) -> training.Job:
    job = training.stored(job_id)
    if job.task != "image":
        raise TrainingError(422, "Finding where it fails works on photo models.")
    if job.status != "finished" or not job.hasModel:
        raise TrainingError(409, "This model hasn't finished training.")
    return job


def _held_back(job: training.Job) -> list[dict]:
    """The held-back photos that are still in the dataset, with what the model answered."""
    held = training.held_out(job)
    if not held:
        raise TrainingError(409, "This model was trained before held-back answers were saved. Train it again.")
    rows = []
    for name, (actual, guess) in held["rows"].items():
        path = training._category_folder(actual, job.dataset) / name
        if path.is_file():
            rows.append({"file": name, "actual": actual, "guess": guess, "path": str(path)})
    return rows


def status(job_id: str) -> dict:
    job = _job_for(job_id)
    running = _jobs.get(job_id)
    if running:
        return running
    try:
        return json.loads((training._folder(job) / RESULT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "none"}


def start(job_id: str) -> dict:
    job = _job_for(job_id)
    rows = _held_back(job)
    mistakes = sum(row["actual"] != row["guess"] for row in rows)
    with _lock:
        if any(state["status"] == "running" for state in _jobs.values()):
            raise TrainingError(409, "It's already looking at another model's mistakes.")
        if training.is_running():
            raise TrainingError(409, "Wait for training to finish. Both need this computer's full attention.")
        if mistakes < MIN_MISTAKES:
            result = {"status": "finished", "enough": False, "held": len(rows), "mistakes": mistakes, "slices": [],
                      "overall": mistakes / len(rows) if rows else 0, "finishedAt": time.time()}
            (training._folder(job) / RESULT_FILE).write_text(json.dumps(result), encoding="utf-8")
            return result
        state = {"status": "running", "startedAt": time.time(), "stage": "loading"}
        _jobs[job_id] = state
    threading.Thread(target=_run, args=(job, rows, state), daemon=True).start()
    return state


def _run(job: training.Job, rows: list[dict], state: dict) -> None:
    phrases = [(kind, text) for kind, texts in PHRASES.items() for text in texts]
    request = {"model": CLIP_MODEL, "paths": [row["path"] for row in rows], "wrong": [row["actual"] != row["guess"] for row in rows],
               "phrases": [text for _, text in phrases]}
    folder = training._folder(job)
    try:
        process = subprocess.Popen([sys.executable, "-I", "-c", _SCRIPT], cwd=folder, env=training._environment(folder),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        process.stdin.write(json.dumps(request))
        process.stdin.close()
        timer = threading.Timer(LIMIT_SECONDS, process.kill)
        timer.start()
        lines = []
        for line in process.stdout:
            if line.strip() == "step":
                state["stage"] = "looking"
            else:
                lines.append(line)
        timer.cancel()
        errors = process.stderr.read()
        if process.wait() != 0:
            last = [line for line in errors.splitlines() if line.strip()]
            raise RuntimeError(last[-1][:300] if last else "CLIP couldn't run.")
        groups = json.loads(lines[-1])
        result = _describe(rows, groups, phrases)
    except Exception as problem:  # The person should see why, rather than a search that never ends.
        result = {"status": "failed", "error": f"{problem}"[:300]}
    result["finishedAt"] = time.time()
    (folder / RESULT_FILE).write_text(json.dumps(result), encoding="utf-8")
    with _lock:
        _jobs.pop(job.id, None)


def _describe(rows: list[dict], groups: list[dict], phrases: list[tuple[str, str]]) -> dict:
    wrong = [row["actual"] != row["guess"] for row in rows]
    overall = sum(wrong) / len(rows)
    slices = []
    for group in groups:
        members = group["members"]
        mistakes = [index for index in members if wrong[index]]
        rate = len(mistakes) / len(members)
        if len(mistakes) < 3 or len(members) < 4 or rate < max(overall + 0.15, 0.3):
            continue
        kind, name = phrases[group["phrases"][0]]
        confusions: dict[tuple[str, str], int] = {}
        for index in mistakes:
            pair = (rows[index]["actual"], rows[index]["guess"])
            confusions[pair] = confusions.get(pair, 0) + 1
        (actual, guess), times = max(confusions.items(), key=lambda item: item[1])
        advice, action = ADVICE[kind]
        slices.append({
            "name": name, "also": [phrases[i][1] for i in group["phrases"][1:3]], "kind": kind,
            "size": len(members), "wrong": len(mistakes), "rate": rate,
            "confusion": {"actual": actual, "guess": guess, "times": times},
            "examples": [{"file": rows[i]["file"], "category": rows[i]["actual"], "guess": rows[i]["guess"]} for i in mistakes[:6]],
            "advice": advice, "action": action,
        })
    slices.sort(key=lambda item: (-item["rate"], -item["wrong"]))
    return {"status": "finished", "enough": True, "held": len(rows), "mistakes": sum(wrong), "overall": overall, "slices": slices[:4]}

