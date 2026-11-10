"""Saved versions of a dataset, so any model can be traced to the exact data that trained it.

A version lists every example and its label. Image files are shared by hard link and texts
are stored once, so a version takes little extra disk space. Examples stay in a version after
they're removed from the dataset, until that version is deleted.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from . import datasets, labelling, opendata, training
from .training import TrainingError

MAX_MESSAGE = 120
CREDITS_FILE = "data_sources.md"
ROW_LABEL = "Rows"
Items = list[list]  # [example id, category or None when unsorted]


def _folder(dataset: str) -> Path:
    return training.root(dataset) / "versions"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def _index(dataset: str) -> dict:
    try:
        return json.loads((_folder(dataset) / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"next": 1, "versions": []}


def _items_of(dataset: str, number: int) -> Items:
    try:
        return json.loads((_folder(dataset) / f"{number}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise TrainingError(404, "That version doesn't exist.") from None


def _find(index: dict, number: int) -> dict:
    for version in index["versions"]:
        if version["number"] == number:
            return version
    raise TrainingError(404, "That version doesn't exist.")


def _texts(dataset: str) -> sqlite3.Connection:
    _folder(dataset).mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_folder(dataset) / "texts.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS texts (id TEXT PRIMARY KEY, text TEXT NOT NULL)")
    return connection


def _table_copy(dataset: str, number: int) -> Path:
    return _folder(dataset) / "tables" / f"{number}.csv"


def _table_items(dataset: str) -> Items:
    """Lists a table's rows. The header counts, so renaming a column shows up as a change."""
    path = datasets.table_file(dataset)
    if not path.exists():
        return []
    csv.field_size_limit(100_000_000)
    with open(path, encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = "\x1f".join(next(reader, []))
        return [[hashlib.sha256((header + "\x1e" + "\x1f".join(row)).encode()).hexdigest()[:32], ROW_LABEL] for row in reader]


def _current_items(dataset: str, meta: dict, extra=()) -> Items:
    """Lists what the dataset holds now. Text keeps its order, because exports follow it."""
    if meta["kind"] == "table":
        return _table_items(dataset)
    if meta["kind"] == "text":
        with closing(labelling._database(dataset)) as database:
            return [list(row) for row in database.execute("SELECT id, label FROM items WHERE removed = 0 ORDER BY seq")]
    items = []
    if meta.get("goal") == "find":
        from . import finder  # Imported here, because finder imports this module.
        boxes = finder.read_boxes(dataset)
        for name in finder.photos(dataset):
            things = sorted({box["category"] for box in boxes.get(name, [])})
            items += [[name, thing] for thing in things] or [[name, None]]
        return sorted(items, key=lambda item: (item[0], item[1] or ""))
    if meta["kind"] == "sound":
        from . import sounds  # Imported here, because sounds imports this module.
        for name in dict.fromkeys([*meta["categories"], *extra]):
            items += [[path.name, name] for path in sounds.files_in(training._category_folder(name, dataset))]
        return sorted(items, key=lambda item: (item[0], item[1] or ""))
    for name in dict.fromkeys([*meta["categories"], *extra]):
        items += [[path.name, name] for path in training._images_in(training._category_folder(name, dataset)) if labelling._IMAGE_ID.match(path.name)]
    items += [[path.name, None] for path in training._images_in(labelling.inbox(dataset)) if labelling._IMAGE_ID.match(path.name)]
    return sorted(items, key=lambda item: (item[0], item[1] or ""))


def _labels(items: Items) -> dict[str, set]:
    found: dict[str, set] = {}
    for item_id, label in items:
        found.setdefault(item_id, set())
        if label is not None:
            found[item_id].add(label)
    return found


def _counts(items: Items) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, label in items:
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _difference(old: Items, new: Items, old_categories: list[str], new_categories: list[str]) -> dict:
    before, after = _labels(old), _labels(new)
    old_counts, new_counts = _counts(old), _counts(new)
    return {
        "added": sum(item not in before for item in after),
        "removed": sum(item not in after for item in before),
        "relabelled": sum(item in before and before[item] != labels for item, labels in after.items()),
        "categoriesAdded": [name for name in new_categories if name not in old_categories],
        "categoriesRemoved": [name for name in old_categories if name not in new_categories],
        "counts": [
            {"name": name, "before": old_counts.get(name, 0), "after": new_counts.get(name, 0)}
            for name in sorted({*old_counts, *new_counts}) if old_counts.get(name, 0) != new_counts.get(name, 0)
        ],
    }


def _has_changes(difference: dict) -> bool:
    return any(difference[key] for key in ("added", "removed", "relabelled", "categoriesAdded", "categoriesRemoved"))


def _head(index: dict) -> dict | None:
    """The version the dataset is based on: the last one saved or restored."""
    for version in index["versions"]:
        if version["number"] == index.get("head"):
            return version
    return index["versions"][-1] if index["versions"] else None


def _since_head(dataset: str, meta: dict, index: dict, extra=()) -> tuple[Items, dict | None]:
    """The dataset now, and how it differs from its version. None means no versions yet."""
    head = _head(index)
    if head is None:
        return _current_items(dataset, meta, extra), None
    items = _current_items(dataset, meta, [*head["counts"], *extra])
    return items, _difference(_items_of(dataset, head["number"]), items, head["categories"], meta["categories"])


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        pass
    except OSError:  # Some disks can't hard link.
        shutil.copy2(source, target)


def _store(dataset: str, meta: dict, items: Items, number: int) -> None:
    if meta["kind"] == "table":
        _link(datasets.table_file(dataset), _table_copy(dataset, number))
        return
    if meta["kind"] == "text":
        with closing(labelling._database(dataset)) as database:
            rows = database.execute("SELECT id, text FROM items WHERE removed = 0").fetchall()
        with closing(_texts(dataset)) as store, store:
            store.executemany("INSERT OR IGNORE INTO texts (id, text) VALUES (?, ?)", rows)
        return
    objects = _folder(dataset) / "objects"
    for item_id, label in items:
        target = objects / item_id
        if not target.exists():
            folder = labelling.inbox(dataset) if label is None or meta.get("goal") == "find" else training._category_folder(label, dataset)
            _link(folder / item_id, target)


def _save(dataset: str, meta: dict, index: dict, items: Items, message: str, automatic: bool) -> dict:
    number = index["next"]
    _store(dataset, meta, items, number)
    head = _head(index)
    changes = _difference(_items_of(dataset, head["number"]), items, head["categories"], meta["categories"]) if head else None
    counts = _counts(items)
    version = {
        "number": number, "message": message, "automatic": automatic, "createdAt": time.time(),
        "categories": meta["categories"], "counts": counts, "labelled": sum(counts.values()),
        "unsorted": sum(label is None for _, label in items), "changes": changes,
        "sources": opendata.sources(dataset),
    }
    if meta["kind"] == "table":
        version["columns"] = meta.get("columns", [])
    _write_json(_folder(dataset) / f"{number}.json", items)
    if meta.get("goal") == "find":
        from . import finder
        _write_json(_folder(dataset) / f"{number}.boxes.json", finder.read_boxes(dataset))
    index = {"next": number + 1, "head": number, "versions": [*index["versions"], version]}
    _write_json(_folder(dataset) / "index.json", index)
    return version


def _saved_boxes(dataset: str, number: int) -> dict:
    try:
        return json.loads((_folder(dataset) / f"{number}.boxes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _boxes_changed(dataset: str, meta: dict, index: dict) -> bool:
    """Boxes can move without the things in a photo changing, which the item list alone wouldn't show."""
    if meta.get("goal") != "find" or not _head(index):
        return False
    from . import finder
    return finder.read_boxes(dataset) != _saved_boxes(dataset, _head(index)["number"])


def overview(dataset: str) -> dict:
    """Lists the versions, newest first, and what has changed since the newest one."""
    with training._lock:
        meta = datasets._read(dataset)
        index = _index(dataset)
        items, changes = _since_head(dataset, meta, index)
    return {
        "versions": index["versions"][::-1],
        "head": head["number"] if (head := _head(index)) else None,
        "unsaved": changes,
        "canSave": bool(items) if changes is None else _has_changes(changes) or _boxes_changed(dataset, meta, index),
    }


def save(dataset: str, message: str) -> dict:
    with training._lock:
        meta = datasets._read(dataset)
        job = opendata.status(dataset)["job"]
        if job and job["status"] == "running":
            raise TrainingError(409, "Wait for the import into this dataset to finish.")
        index = _index(dataset)
        items, changes = _since_head(dataset, meta, index)
        if changes is None and not items:
            raise TrainingError(422, "Add some examples before saving a version.")
        if changes is not None and not _has_changes(changes) and not _boxes_changed(dataset, meta, index):
            raise TrainingError(409, f"Nothing has changed since v{_head(index)['number']}.")
        return _save(dataset, meta, index, items, " ".join(message.split())[:MAX_MESSAGE], False)


def save_before_change(dataset: str, message: str) -> int | None:
    """Saves any unsaved work as a version, and returns its number if one was needed."""
    with training._lock:
        meta = datasets._read(dataset)
        index = _index(dataset)
        items, changes = _since_head(dataset, meta, index)
        if items and (changes is None or _has_changes(changes) or _boxes_changed(dataset, meta, index)):
            return _save(dataset, meta, index, items, message, True)["number"]
        return None


def compare(dataset: str, number: int) -> dict:
    """What changed between a saved version and the dataset as it is now."""
    with training._lock:
        meta = datasets._read(dataset)
        index = _index(dataset)
        version = _find(index, number)
        extra = [*version["counts"], *_head(index)["counts"]]
        return _difference(_items_of(dataset, number), _current_items(dataset, meta, extra), version["categories"], meta["categories"])


def _freeze_text(dataset: str, names: list[str], items: Items, run: Path) -> None:
    """Writes the labelled text for one run, in the same format as the dataset's export."""
    with closing(_texts(dataset)) as store:
        texts = dict(store.execute("SELECT id, text FROM texts"))
    wanted = set(names)
    with open(run / "data" / "texts.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows((texts[item_id], label) for item_id, label in items if label in wanted)


def freeze(dataset: str, names: list[str], run: Path) -> int:
    """Makes sure a version matches the dataset, and copies its examples for one training run."""
    with training._lock:
        meta = datasets._read(dataset)
        index = _index(dataset)
        items, changes = _since_head(dataset, meta, index, names)
        if changes is None or _has_changes(changes) or _boxes_changed(dataset, meta, index):
            version = _save(dataset, meta, index, items, "Saved automatically for training", True)
        else:
            version = _head(index)
        if meta["kind"] == "table":
            (run / "data").mkdir(parents=True, exist_ok=True)
            _link(_table_copy(dataset, version["number"]), run / "data" / "table.csv")
        elif meta["kind"] == "text":
            _freeze_text(dataset, names, items, run)
        elif meta.get("goal") == "find":
            from . import finder
            finder.freeze(dataset, run)
        else:
            target = run / "data" / ("sounds" if meta["kind"] == "sound" else "images")
            objects = _folder(dataset) / "objects"
            for name in names:
                (target / name).mkdir(parents=True, exist_ok=True)
            wanted = set(names)
            for item_id, label in items:
                if label in wanted:
                    _link(objects / item_id, target / label / item_id)
        credit = opendata.credits_markdown(version["sources"])
        if credit:
            (run / CREDITS_FILE).write_text(credit, encoding="utf-8")
        return version["number"]


def _restore_images(dataset: str, current: Items, items: Items) -> None:
    objects = _folder(dataset) / "objects"
    if datasets._read(dataset)["kind"] == "image" and training.image_total() - len(current) + len(items) > training.MAX_IMAGES:
        raise TrainingError(413, f"The image store would go over {training.MAX_IMAGES:,} images. Remove some first.")
    staging = training.root(dataset) / "restoring"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        # Everything is prepared first, so a missing file can't leave a half-restored dataset.
        for item_id, label in items:
            folder = staging / ("inbox" if label is None else training._category_folder(label, dataset).name)
            source = objects / item_id
            if not source.is_file():
                raise TrainingError(500, "This version is missing some of its images, so it can't be restored.")
            _link(source, folder / item_id)
        names = {label for _, label in [*current, *items] if label is not None}
        for name in names:
            folder = training._category_folder(name, dataset)
            shutil.rmtree(folder, ignore_errors=True)
            if (staging / folder.name).is_dir():
                folder.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / folder.name, folder)
        inbox = labelling.inbox(dataset)
        shutil.rmtree(inbox, ignore_errors=True)
        if (staging / "inbox").is_dir():
            inbox.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / "inbox", inbox)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _restore_text(dataset: str, items: Items) -> None:
    with closing(_texts(dataset)) as store:
        texts = dict(store.execute("SELECT id, text FROM texts"))
    if any(item_id not in texts for item_id, _ in items):
        raise TrainingError(500, "This version is missing some of its text, so it can't be restored.")
    with closing(labelling._database(dataset)) as database, database:
        database.execute("DELETE FROM items")
        database.executemany("INSERT INTO items (id, text, label) VALUES (?, ?, ?)", [(item_id, texts[item_id], label) for item_id, label in items])


def _restore_table(dataset: str, number: int) -> None:
    source = _table_copy(dataset, number)
    if not source.is_file():
        raise TrainingError(500, "This version is missing its table, so it can't be restored.")
    target = datasets.table_file(dataset)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{uuid.uuid4().hex}.partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def restore(dataset: str, number: int) -> dict:
    """Makes the dataset match a version. Unsaved work is saved as a new version first."""
    with training._lock:
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        meta = datasets._read(dataset)
        index = _index(dataset)
        version = _find(index, number)
        items = _items_of(dataset, number)
        current, changes = _since_head(dataset, meta, index, version["counts"])
        saved_first = None
        if changes is None or _has_changes(changes):
            saved_first = _save(dataset, meta, index, current, f"Saved automatically before restoring v{number}", True)["number"]
            index = _index(dataset)
        if meta.get("goal") == "find":
            from . import finder
            photos = sorted({item_id for item_id, _ in items})
            _restore_images(dataset, current, [[item_id, None] for item_id in photos])
            finder.write_boxes(dataset, _saved_boxes(dataset, number))
        elif meta["kind"] in ("image", "sound"):
            _restore_images(dataset, current, items)
        elif meta["kind"] == "table":
            _restore_table(dataset, number)
            meta["rows"], meta["columns"] = len(items), version.get("columns", [])
        else:
            _restore_text(dataset, items)
        if version["sources"]:
            _write_json(opendata._sources_file(dataset), version["sources"])
        else:
            opendata._sources_file(dataset).unlink(missing_ok=True)
        labelling._suggestions_file(dataset).unlink(missing_ok=True)
        _write_json(_folder(dataset) / "index.json", index | {"head": number})
        meta["categories"] = version["categories"]
        datasets._write(meta)
        return {"dataset": datasets.get(dataset), "savedFirst": saved_first}


def delete(dataset: str, number: int) -> None:
    """Deletes a version and any files that only it was keeping."""
    with training._lock:
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        meta = datasets._read(dataset)
        index = _index(dataset)
        _find(index, number)
        kept = [version for version in index["versions"] if version["number"] != number]
        _write_json(_folder(dataset) / "index.json", index | {"versions": kept})
        (_folder(dataset) / f"{number}.json").unlink(missing_ok=True)
        _table_copy(dataset, number).unlink(missing_ok=True)
        used = {item_id for version in kept for item_id, _ in _items_of(dataset, version["number"])}
        if meta["kind"] == "text":
            with closing(_texts(dataset)) as store:
                store.execute("PRAGMA secure_delete = ON")
                stored = [row[0] for row in store.execute("SELECT id FROM texts")]
                with store:
                    store.executemany("DELETE FROM texts WHERE id = ?", [(item,) for item in stored if item not in used])
                store.execute("VACUUM")
        else:
            objects = _folder(dataset) / "objects"
            for path in (objects.iterdir() if objects.is_dir() else []):
                if path.name not in used:
                    path.unlink(missing_ok=True)


def credits(job: training.Job) -> str | None:
    """Credits for the data a model used, saved with the run so they outlast the dataset."""
    try:
        return (training._folder(job) / CREDITS_FILE).read_text(encoding="utf-8")
    except OSError:
        return None if job.datasetVersion is not None else opendata.sources_markdown(job.dataset)
