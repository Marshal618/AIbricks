"""Find things by name: type what to look for and a pretrained model draws the first boxes.

Drawing every box by hand is the slow part of teaching a detector. OWLv2 (Minderer et al., 2023,
Apache 2.0, from Google) finds things it has never been trained on, from a few words such as "red
mug". Its boxes are only suggestions: you fix them in the labeller and save, then train the usual
detector (see finder.py), which is far smaller and quicker and can go into web apps and Raspberry Pi
kits. OWLv2 itself is too big for those, about 620 MB, so it's only used while labelling.

It runs inside this app with PyTorch and Transformers, on this computer. The weights download once
from Hugging Face, pinned to one fixed version.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from . import datasets, labelling, training
from .training import TrainingError

REPO = "google/owlv2-base-patch16-ensemble"
REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
FILES = ["*.json", "*.txt", "model.safetensors"]
DOWNLOAD = "620 MB"
MAX_WORDS = 80
MAX_SEARCHES = 30
LOWEST = 0.05  # boxes less sure than this are never sent; the labeller hides more with its own slider
DEFAULT_SURE = 0.2
SAME_THING = 0.5  # boxes of one thing that overlap this much are one find
SAME_PLACE = 0.8  # boxes of different things that overlap this much are one find, given to the surer
MAX_BOXES = 100
IDLE_SECONDS = 600

_lock = threading.Lock()
_download_lock = threading.Lock()
_loaded: dict | None = None
_used = 0.0
_downloading = False
_problem: str | None = None


def _hub() -> Path:
    return training._cache() / "huggingface" / "hub"


def _snapshot() -> Path:
    return _hub() / f"models--{REPO.replace('/', '--')}" / "snapshots" / REVISION


def downloaded() -> bool:
    return (_snapshot() / "model.safetensors").is_file() and (_snapshot() / "preprocessor_config.json").is_file()


def status() -> dict:
    return {"downloaded": downloaded(), "downloading": _downloading, "problem": _problem, "download": DOWNLOAD}


def _fetch() -> None:
    global _downloading, _problem
    try:
        from huggingface_hub import snapshot_download
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        snapshot_download(REPO, revision=REVISION, allow_patterns=FILES, cache_dir=str(_hub()))
        if not downloaded():
            _problem = "The download finished without the model's files. Try again."
    except Exception as problem:  # any network or disk problem ends up here
        _problem = f"The finder couldn't be downloaded. It needs the internet once, about {DOWNLOAD}. ({type(problem).__name__})"
    finally:
        _downloading = False


def start_download() -> dict:
    """Downloads the model in the background. Asking again while it downloads changes nothing."""
    global _downloading, _problem
    with _download_lock:
        if not downloaded() and not _downloading:
            _downloading, _problem = True, None
            threading.Thread(target=_fetch, daemon=True).start()
    return status()


def _model() -> dict:
    """The model, loaded on first use and let go after a while unused, as it holds about 600 MB."""
    global _loaded, _used
    if not downloaded():
        raise TrainingError(409, f"Download the finder first, about {DOWNLOAD}.")
    if _loaded is None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        path = str(_snapshot())
        _loaded = {"processor": Owlv2Processor.from_pretrained(path), "model": Owlv2ForObjectDetection.from_pretrained(path).eval()}
        threading.Thread(target=_let_go_when_idle, daemon=True).start()
    _used = time.time()
    return _loaded


def _let_go_when_idle() -> None:
    global _loaded
    while True:
        time.sleep(30)
        with _lock:
            if _loaded is None:
                return
            if time.time() - _used > IDLE_SECONDS:
                _loaded = None
                return


# ------------------------------------------------------------------ words for each thing

def _words_file(dataset: str) -> Path:
    return training.root(dataset) / "find" / "words.json"


def _require_find(dataset: str) -> dict:
    meta = datasets._read(dataset)
    if not (meta.get("kind") == "image" and meta.get("goal") == "find"):
        raise TrainingError(422, "Finding by name is for “find things in photos” datasets.")
    return meta


def _saved_words(dataset: str) -> dict[str, str]:
    try:
        saved = json.loads(_words_file(dataset).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {key: value for key, value in saved.items() if isinstance(key, str) and isinstance(value, str)}


def _write_words(dataset: str, words: dict[str, str]) -> None:
    path = _words_file(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(words), encoding="utf-8")
    os.replace(temporary, path)


def words(dataset: str) -> dict:
    """What to look for for each thing. A thing's own name is used until other words are given."""
    meta = _require_find(dataset)
    saved = _saved_words(dataset)
    return {"words": {name: saved.get(name) or name for name in meta["categories"]}, **status()}


def set_words(dataset: str, given: dict[str, str]) -> dict:
    with training._lock:
        meta = _require_find(dataset)
        saved = _saved_words(dataset)
        for name, text in given.items():
            if name not in meta["categories"]:
                raise TrainingError(422, f'"{str(name)[:40]}" isn\'t one of the things this dataset finds. Refresh the page.')
            text = " ".join(str(text).split())
            if len(text) > MAX_WORDS:
                raise TrainingError(422, f"Keep the words for each thing under {MAX_WORDS} characters.")
            if text and text != name:
                saved[name] = text
            else:
                saved.pop(name, None)
        _write_words(dataset, {name: text for name, text in saved.items() if name in meta["categories"]})
    return words(dataset)


def rename_category(dataset: str, old: str, new: str) -> None:
    saved = _saved_words(dataset)
    if old in saved:
        saved[new] = saved.pop(old)
        _write_words(dataset, saved)


def remove_category(dataset: str, name: str) -> None:
    saved = _saved_words(dataset)
    if saved.pop(name, None) is not None:
        _write_words(dataset, saved)


def searches(named: dict[str, str]) -> list[tuple[str, str]]:
    """Each thing's words split at commas, so "mug, cup" looks for both. Each becomes (thing, phrase)."""
    found = []
    for name, text in named.items():
        for part in text.split(","):
            part = part.strip()
            if part and (name, part) not in found:
                found.append((name, part))
    if len(found) > MAX_SEARCHES:
        raise TrainingError(422, f"Look for at most {MAX_SEARCHES} things at once, counting each comma-separated name.")
    return found


# ------------------------------------------------------------------ finding

def _overlap(a: dict, b: dict) -> float:
    width = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    height = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    shared = width * height
    union = a["w"] * a["h"] + b["w"] * b["h"] - shared
    return shared / union if union > 0 else 0.0


def tidy(boxes: list[dict]) -> list[dict]:
    """Keeps the surest of boxes that overlap: a lot for the same thing, almost fully for different things."""
    kept: list[dict] = []
    for box in sorted(boxes, key=lambda item: -item["chance"]):
        if all(_overlap(box, other) < (SAME_THING if other["category"] == box["category"] else SAME_PLACE) for other in kept):
            kept.append(box)
        if len(kept) == MAX_BOXES:
            break
    return kept


def to_boxes(rows, labels, scores, searched: list[tuple[str, str]], width: int, height: int) -> list[dict]:
    """The model's corner positions, in pixels of the square it pads every photo to, as shares of the photo."""
    found = []
    for (x1, y1, x2, y2), label, score in zip(rows, labels, scores):
        x1, x2 = max(0.0, x1 / width), min(1.0, x2 / width)
        y1, y2 = max(0.0, y1 / height), min(1.0, y2 / height)
        if score < LOWEST or x2 - x1 < 0.005 or y2 - y1 < 0.005:
            continue
        found.append({"category": searched[label][0], "chance": round(float(score), 4),
                      "x": round(x1, 5), "y": round(y1, 5), "w": round(x2 - x1, 5), "h": round(y2 - y1, 5)})
    return tidy(found)


def find(dataset: str, image_id: str) -> dict:
    """Boxes around everything in one photo that matches each thing's words, surest first."""
    import torch
    from PIL import Image, ImageOps

    named = words(dataset)["words"]
    if not named:
        raise TrainingError(422, "Name at least one thing to find.")
    searched = searches(named)
    path = labelling.image_file(image_id, dataset)
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except OSError:
        raise TrainingError(422, "This photo couldn't be read.") from None
    with _lock:
        loaded = _model()
        processor, model = loaded["processor"], loaded["model"]
        inputs = processor(text=[[f"a photo of a {phrase}" for _, phrase in searched]], images=image, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)
        side = max(image.size)  # OWLv2 pads each photo to a square at its bottom and right
        result = processor.post_process_grounded_object_detection(output, threshold=LOWEST, target_sizes=torch.tensor([[side, side]]))[0]
    boxes = to_boxes(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist(), searched, image.width, image.height)
    return {"boxes": boxes, "words": named}
