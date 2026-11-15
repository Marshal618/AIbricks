"""Is this like what it learned from? A warning for photos and text unlike anything a model trained on.

A model always gives an answer, even for a photo of something it has never seen. This checks how far a
new example is from the examples it learned from, so the app can say when an answer is a guess about
something unfamiliar.

It follows deep nearest neighbours (Sun et al., 2022): a pretrained model (ResNet-18 for photos,
MiniLM for text) describes every example, and an example's distance is how far it is from its 5th
closest training example. The line between familiar and unfamiliar is drawn with conformal p-values
(Bates et al., 2023): the held-back examples, measured against the training ones, and each training
example, measured against the others, show how far examples like the ones it learned from usually are.
A new example is unfamiliar when fewer than 1 in 20 of those were as far, so about 1 ordinary example
in 20 is flagged by mistake, whatever the data.

The descriptions are worked out once per model, from the dataset version it trained on, and kept with
the model. They hold no photos or text, only numbers and example IDs.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from pathlib import Path

from . import datasets, training, versions
from .training import TrainingError

K = 5
LEVEL = 0.05  # unfamiliar when farther than 19 in 20 of the known examples
MIN_USUAL = 19  # with fewer, no example can be farther than 19 in 20 of them, so it never warns
MAX_EXAMPLES = 4000
NEAREST_SHOWN = 3
IDLE_SECONDS = 600
BATCH = 32
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FILE = "familiar.pt"

_lock = threading.Lock()  # the describing models are shared
_builds: dict[str, dict] = {}
_models: dict[str, object] = {}
_used = 0.0
_watching = False


# ------------------------------------------------------------------ describing examples

def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _certificates() -> None:
    """Python from python.org on a Mac has no certificates of its own, so downloads would fail."""
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def _describer(kind: str):
    """The pretrained model for photos or text, loaded on first use and let go after a while unused."""
    global _used, _watching
    if kind not in _models:
        import torch
        if kind == "image":
            from torchvision.models import ResNet18_Weights, resnet18
            weights = ResNet18_Weights.DEFAULT
            model = resnet18()
            try:
                _certificates()
                model.load_state_dict(torch.hub.load_state_dict_from_url(
                    weights.url, model_dir=str(training._cache() / "torch" / "hub" / "checkpoints"), progress=False, check_hash=True))
            except Exception:  # torchvision raises several kinds of error when offline
                raise TrainingError(503, "The photo describer couldn't be downloaded. The first check needs an internet connection.") from None
            model.fc = torch.nn.Identity()
            _models[kind] = (model.to(_device()).eval(), weights.transforms())
        else:
            from transformers import AutoModel, AutoTokenizer
            try:
                _certificates()
                where = str(training._cache() / "huggingface" / "hub")
                tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, cache_dir=where)
                model = AutoModel.from_pretrained(TEXT_MODEL, cache_dir=where, use_safetensors=True)
            except (OSError, ValueError):
                raise TrainingError(503, "The text describer couldn't be downloaded. The first check needs an internet connection.") from None
            _models[kind] = (model.to(_device()).eval(), tokenizer)
        if not _watching:
            _watching = True
            threading.Thread(target=_let_go_when_idle, daemon=True).start()
    _used = time.time()
    return _models[kind]


def _let_go_when_idle() -> None:
    global _watching
    while True:
        time.sleep(30)
        with _lock:
            if time.time() - _used > IDLE_SECONDS:
                _models.clear()
                _watching = False
                return


def _describe_photos(pictures: list) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F
    from PIL import Image
    rows = []
    for start in range(0, len(pictures), BATCH):
        with _lock, torch.no_grad():  # a batch at a time, so a quick check isn't stuck behind a long build
            model, prepare = _describer("image")
            batch = []
            for item in pictures[start:start + BATCH]:
                with Image.open(item if isinstance(item, Path) else io.BytesIO(item)) as picture:
                    batch.append(prepare(picture.convert("RGB")))
            rows.append(F.normalize(model(torch.stack(batch).to(next(model.parameters()).device)), dim=1).cpu())
    return torch.cat(rows)


def _describe_texts(texts: list[str]) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F
    rows = []
    for start in range(0, len(texts), BATCH):
        with _lock, torch.no_grad():
            model, tokenizer = _describer("text")
            where = next(model.parameters()).device
            batch = tokenizer(texts[start:start + BATCH], padding=True, truncation=True, max_length=128, return_tensors="pt").to(where)
            words = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            rows.append(F.normalize((words * mask).sum(1) / mask.sum(1).clamp(min=1e-9), dim=1).cpu())
    return torch.cat(rows)


# ------------------------------------------------------------------ distances and the line

def distances(queries, known, skip_self: bool = False):
    """How far each query is from its Kth closest known example (1 − cosine similarity)."""
    import torch
    gaps = 1 - queries @ known.T
    if skip_self:
        gaps.fill_diagonal_(float("inf"))
    k = min(K, known.shape[0] - (1 if skip_self else 0))
    return torch.topk(gaps, k, dim=1, largest=False).values[:, -1].clamp(min=0)


def p_value(distance: float, usual: list[float]) -> float:
    """The share of usual examples at least this far, counting the new one: (1 + farther) / (n + 1)."""
    return (1 + sum(gap >= distance for gap in usual)) / (len(usual) + 1)


def usual_distances(train, held=None) -> list[float]:
    """How far examples like the training ones usually are: held-back ones against the training
    examples, and each training example against the others."""
    gaps = distances(train, train, skip_self=True).tolist()
    if held is not None and held.shape[0]:
        gaps += distances(held, train).tolist()
    return sorted(gaps)


# ------------------------------------------------------------------ the examples a model learned from

def _job(job_id: str) -> training.Job:
    job = training.stored(job_id)
    if job.task not in ("image", "text") or not job.dataset:
        raise TrainingError(422, "This check is for photo and text sorters.")
    held = training.held_out(job)
    if job.task == "image" and held is not None and held.get("task") != "category":
        raise TrainingError(422, "This check is for photo and text sorters.")  # a model that gives a number or rebuilds photos
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This run didn't save a model.")
    return job


def _sample(rows: list, limit: int) -> list:
    """An even, repeatable sample, so very large datasets stay quick."""
    if len(rows) <= limit:
        return rows
    return sorted(rows, key=lambda row: hashlib.sha256(row[0].encode()).hexdigest())[:limit]


def _examples(job: training.Job) -> tuple[list, list]:
    """(id, category, photo path or text) for training and held-back examples, from the version it trained on."""
    held_keys = set(((training.held_out(job) or {}).get("rows") or {}))
    wanted = set(job.categories)
    if job.task == "text":
        if job.datasetVersion is None:
            raise TrainingError(409, "This model's dataset version is no longer saved.")
        items = versions._items_of(job.dataset, job.datasetVersion)
        from contextlib import closing
        with closing(versions._texts(job.dataset)) as store:
            texts = dict(store.execute("SELECT id, text FROM texts"))
        rows = [(item_id, label, texts[item_id]) for item_id, label in items if label in wanted and item_id in texts]
        key = lambda row: hashlib.sha256(row[2].encode()).hexdigest()[:32]  # noqa: E731
    else:
        if job.datasetVersion is not None:
            objects = versions._folder(job.dataset) / "objects"
            items = versions._items_of(job.dataset, job.datasetVersion)
            rows = [(item_id, label, objects / item_id) for item_id, label in items if label in wanted]
        else:
            rows = []
            for label in sorted(wanted):
                folder = training._category_folder(label, job.dataset)
                rows += [(path.name, label, path) for path in sorted(folder.iterdir()) if path.is_file()] if folder.is_dir() else []
        rows = [row for row in rows if Path(row[2]).is_file()]
        key = lambda row: row[0]  # noqa: E731
    train = [row for row in rows if key(row) not in held_keys]
    held = [row for row in rows if key(row) in held_keys]
    return _sample(train, MAX_EXAMPLES), _sample(held, MAX_EXAMPLES // 4)


def _build(job: training.Job) -> None:
    import torch
    state = _builds[job.id]
    try:
        train, held = _examples(job)
        if len(train) < K + 1:
            raise TrainingError(422, f"It needs at least {K + 1} training examples to compare with.")
        describe = _describe_texts if job.task == "text" else _describe_photos
        known = describe([row[2] for row in train])
        checked = describe([row[2] for row in held]) if held else None
        torch.save({
            "kind": job.task, "known": known, "ids": [row[0] for row in train], "labels": [row[1] for row in train],
            "usual": usual_distances(known, checked), "held": len(held),
        }, training._folder(job) / FILE)
        state.update(status="ready", error=None)
    except TrainingError as error:
        state.update(status="failed", error=error.message)
    except Exception as error:  # a broken photo or a describer problem shouldn't take the app down
        state.update(status="failed", error=f"The check couldn't be prepared ({type(error).__name__}).")


def _saved(job: training.Job) -> dict | None:
    import torch
    path = training._folder(job) / FILE
    if not path.exists():
        return None
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return None


def status(job_id: str) -> dict:
    job = _job(job_id)
    build = _builds.get(job.id)
    if build and build["status"] == "building":
        return {"status": "building", "error": None}
    saved = _saved(job)
    if saved is not None:
        return {"status": "ready", "error": None, "known": len(saved["ids"]), "held": saved["held"], "usual": len(saved["usual"]),
                "canWarn": len(saved["usual"]) >= MIN_USUAL}
    if build and build["status"] == "failed":
        return {"status": "failed", "error": build["error"]}
    return {"status": "missing", "error": None}


def start(job_id: str) -> dict:
    """Works out the descriptions of what the model learned from, in the background."""
    job = _job(job_id)
    if _builds.get(job.id, {}).get("status") != "building" and _saved(job) is None:
        _builds[job.id] = {"status": "building", "error": None}
        threading.Thread(target=_build, args=(job,), daemon=True).start()
    return status(job_id)


# ------------------------------------------------------------------ checking a new example

def _check(job: training.Job, describe) -> dict:
    saved = _saved(job)
    if saved is None:
        raise TrainingError(409, "Prepare the check first.")
    query = describe()
    gap = float(distances(query, saved["known"])[0])
    chance = p_value(gap, saved["usual"])
    similar = (query @ saved["known"].T)[0]
    order = similar.argsort(descending=True)[:NEAREST_SHOWN].tolist()
    return {
        "unfamiliar": chance <= LEVEL,
        "farther": round(1 - sum(usual >= gap for usual in saved["usual"]) / len(saved["usual"]), 4),
        "pValue": round(chance, 4), "level": LEVEL, "usual": len(saved["usual"]),
        "nearest": [
            {"index": index, "id": saved["ids"][index], "category": saved["labels"][index], "similarity": round(float(similar[index]), 3)}
            for index in order
        ],
    }


def check_photo(job_id: str, data: bytes) -> dict:
    training.read_image(data)
    job = _job(job_id)
    if job.task != "image":
        raise TrainingError(422, "This model doesn't read photos.")
    return _check(job, lambda: _describe_photos([data]))


def check_text(job_id: str, text: str) -> dict:
    job = _job(job_id)
    if job.task != "text":
        raise TrainingError(422, "This model doesn't read text.")
    if not text.strip():
        raise TrainingError(422, "Type some text first.")
    reply = _check(job, lambda: _describe_texts([text]))
    from contextlib import closing
    with closing(versions._texts(job.dataset)) as store:
        found = dict(store.execute(
            f"SELECT id, text FROM texts WHERE id IN ({','.join('?' * len(reply['nearest']))})", [row["id"] for row in reply["nearest"]],
        ))
    for row in reply["nearest"]:
        row["text"] = found.get(row["id"])  # gone if it was deleted from the dataset since
    return reply


def nearest_photo(job_id: str, index: int) -> Path:
    """One of the training photos, from the version the model trained on."""
    job = _job(job_id)
    saved = _saved(job)
    if saved is None or job.task != "image" or not 0 <= index < len(saved["ids"]):
        raise TrainingError(404, "That photo isn't part of this check.")
    item_id = saved["ids"][index]
    if job.datasetVersion is not None:
        path = versions._folder(job.dataset) / "objects" / item_id
    else:
        path = training._category_folder(saved["labels"][index], job.dataset) / item_id
    datasets.require(job.dataset)
    if not path.is_file():
        raise TrainingError(404, "That photo is no longer saved.")
    return path
