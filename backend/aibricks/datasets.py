"""Datasets as saved items. Each has its own categories, images or text, and import credits."""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid

from pydantic import BaseModel, Field

from . import labelling, opendata, training
from .graph import NAME_RE
from .training import TrainingError

MAX_CATEGORIES = 100
PREVIEW_SAMPLES = 8


class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    kind: str = Field(pattern="^(image|text)$")
    categories: list[str] = Field(default_factory=list, max_length=MAX_CATEGORIES)


class DatasetDetails(BaseModel):
    description: str = Field(max_length=2000)
    licence: str = Field(max_length=60)


def _meta_file(dataset: str):
    return training.root(dataset) / "dataset.json"


def _write(meta: dict) -> dict:
    meta["updatedAt"] = time.time()
    path = _meta_file(meta["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(temporary, path)
    return meta


def _read(dataset: str) -> dict:
    if dataset == training.DEFAULT_DATASET:
        raise TrainingError(404, "That dataset doesn't exist.")
    try:
        return json.loads(_meta_file(dataset).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise TrainingError(404, "That dataset doesn't exist.") from None


def require(dataset: str) -> None:
    """Checks that a dataset can be used. The default store exists for older callers."""
    training.root(dataset)
    if dataset != training.DEFAULT_DATASET:
        _read(dataset)


def _clean_name(name: str) -> str:
    """Category names become model categories and folder names, so they must suit both."""
    name = name.strip()
    if not NAME_RE.match(name) or "," in name or "/" in name or name.startswith("."):
        raise TrainingError(422, f'"{name[:40]}" can\'t be a category. Use letters, numbers, spaces and . - % ( ) _ only.')
    return name


def _clean_categories(names: list[str]) -> list[str]:
    cleaned = [_clean_name(name) for name in names]
    if len({name.casefold() for name in cleaned}) != len(cleaned):
        raise TrainingError(422, "Two categories have the same name.")
    if len(cleaned) > MAX_CATEGORIES:
        raise TrainingError(422, f"A dataset can have at most {MAX_CATEGORIES} categories.")
    return cleaned


def _summary(meta: dict) -> dict:
    dataset, names = meta["id"], meta["categories"]
    if meta["kind"] == "image":
        amounts, unsorted = training.counts(names, dataset), labelling.queue(dataset)["total"]
    else:
        queue = labelling.text_queue(names, dataset)
        amounts, unsorted = queue["counts"], queue["total"]
    return {"description": "", "licence": ""} | meta | {"counts": amounts, "labelled": sum(amounts.values()), "unsorted": unsorted}


def create(request: DatasetIn) -> dict:
    name = " ".join(request.name.split())
    if not name:
        raise TrainingError(422, "Give the dataset a name.")
    meta = {
        "id": uuid.uuid4().hex[:12], "name": name, "kind": request.kind,
        "categories": _clean_categories(request.categories), "createdAt": time.time(),
    }
    return _summary(_write(meta))


def get(dataset: str) -> dict:
    return _summary(_read(dataset))


def listing() -> list[dict]:
    rows = []
    for path in (training.workspace() / "collections").glob("*/dataset.json"):
        try:
            rows.append(_summary(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError, TrainingError):
            continue  # A damaged dataset shouldn't hide the others.
    return sorted(rows, key=lambda row: row["updatedAt"], reverse=True)


def rename(dataset: str, name: str) -> dict:
    meta = _read(dataset)
    meta["name"] = " ".join(name.split()) or meta["name"]
    return _summary(_write(meta))


def update_details(dataset: str, details: DatasetDetails) -> dict:
    with training._lock:
        meta = _read(dataset)
        meta["description"], meta["licence"] = details.description.strip(), " ".join(details.licence.split())
        return _summary(_write(meta))


def preview(dataset: str) -> dict:
    """A few examples from each category, and where imported data came from."""
    meta = _read(dataset)
    rows = []
    for name in meta["categories"]:
        if meta["kind"] == "image":
            files = sorted(path.name for path in training._images_in(training._category_folder(name, dataset)))
            samples = [{"id": item} for item in files[:PREVIEW_SAMPLES]]
        else:
            samples = labelling.text_samples(name, PREVIEW_SAMPLES, dataset)
        rows.append({"name": name, "samples": samples})
    return {"categories": rows, "sources": opendata.sources(dataset)}


def preview_image(dataset: str, category: str, image_id: str):
    meta = _read(dataset)
    if category not in meta["categories"] or not labelling._IMAGE_ID.match(image_id):
        raise TrainingError(404, "That image doesn't exist.")
    path = training._category_folder(category, dataset) / image_id
    if not path.is_file():
        raise TrainingError(404, "That image doesn't exist.")
    return path


def add_categories(dataset: str, names: list[str]) -> dict:
    with training._lock:
        meta = _read(dataset)
        known = {name.casefold() for name in meta["categories"]}
        fresh = [name for name in _clean_categories(names) if name.casefold() not in known]
        meta["categories"] = _clean_categories(meta["categories"] + fresh)
        return _summary(_write(meta))


def _busy(dataset: str) -> str | None:
    if training.is_running() and training.current().dataset == dataset:
        return "Wait for training on this dataset to finish."
    job = opendata.status(dataset)["job"]
    if job and job["status"] == "running":
        return "Wait for the import into this dataset to finish."
    if labelling.is_suggesting():
        return "Wait for the model to finish making suggestions."
    return None


def rename_category(dataset: str, old: str, new: str) -> dict:
    with training._lock:
        meta = _read(dataset)
        if old not in meta["categories"]:
            raise TrainingError(404, "That category doesn't exist.")
        new = _clean_name(new)
        if new == old:
            return _summary(meta)
        if new.casefold() in {name.casefold() for name in meta["categories"] if name != old}:
            raise TrainingError(422, "Another category already has that name.")
        problem = _busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        source, target = training._category_folder(old, dataset), training._category_folder(new, dataset)
        if source.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            os.replace(source, target)
        labelling.rename_text_label(old, new, dataset)
        meta["categories"] = [new if name == old else name for name in meta["categories"]]
        return _summary(_write(meta))


def remove_category(dataset: str, name: str) -> dict:
    """Removes a category. Its examples go back to being unsorted, so no work is deleted."""
    with training._lock:
        meta = _read(dataset)
        if name not in meta["categories"]:
            raise TrainingError(404, "That category doesn't exist.")
        problem = _busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        folder = training._category_folder(name, dataset)
        if folder.is_dir():
            inbox = labelling.inbox(dataset)
            inbox.mkdir(parents=True, exist_ok=True)
            for path in folder.iterdir():
                os.replace(path, inbox / path.name)
            folder.rmdir()
        labelling.clear_text_label(name, dataset)
        meta["categories"] = [other for other in meta["categories"] if other != name]
        return _summary(_write(meta))


def delete(dataset: str) -> None:
    with training._lock:
        _read(dataset)
        problem = _busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        shutil.rmtree(training.root(dataset), ignore_errors=True)
