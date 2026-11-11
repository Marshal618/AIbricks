"""Sorts unlabelled images and text by hand, with optional suggestions from a trained model.

Unsorted images wait in an inbox. Labelling one moves it into the same category folders
that training reads, so nothing needs converting afterwards. Everything stays on this computer.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import threading
import uuid
from contextlib import closing
from pathlib import Path

from . import training
from .training import TrainingError

MAX_QUEUE = 5000
MAX_TEXT_ITEMS = 100_000
MAX_TEXT_LENGTH = 5000
MAX_TEXT_FILE_BYTES = 50 * 1024 * 1024
SUGGEST_LIMIT_SECONDS = 60 * 60
REMOVED_KEPT = 100

_IMAGE_ID = re.compile(r"^[0-9a-f]{32}\.(jpg|png|bmp|webp)$")
_suggesting = False
_suggest_error: str | None = None


def inbox(dataset: str = training.DEFAULT_DATASET) -> Path:
    return training.root(dataset) / "inbox" / "images"


def _suggestions_file(dataset: str = training.DEFAULT_DATASET) -> Path:
    return training.root(dataset) / "inbox" / "suggestions.json"


def _inbox_path(image_id: str, dataset: str = training.DEFAULT_DATASET) -> Path:
    if not _IMAGE_ID.match(image_id):
        raise TrainingError(404, "That image doesn't exist.")
    return inbox(dataset) / image_id


_KIND_NAMES = {"image": "photo", "text": "text", "table": "spreadsheet", "sound": "sound"}


def _require_kind(dataset: str, kind: str) -> None:
    """Stops images going into a text dataset, text into a spreadsheet, and so on."""
    if dataset == training.DEFAULT_DATASET:
        return
    from . import datasets  # imported here because datasets imports this module
    actual = datasets._read(dataset)["kind"]
    if actual != kind:
        things = {"image": "photos", "text": "text", "table": "spreadsheets", "sound": "recordings"}
        raise TrainingError(422, f"This is a {_KIND_NAMES.get(actual, actual)} dataset, so it can't hold {things[kind]}.")


# ---------------------------------------------------------------------- images

def add_image(data: bytes, dataset: str = training.DEFAULT_DATASET) -> dict:
    _require_kind(dataset, "image")
    extension = training.read_image(data)
    image_id = hashlib.sha256(data).hexdigest()[:32] + extension
    with training._lock:
        known = [inbox(dataset)] + list((training.root(dataset) / "datasets").glob("*"))
        if any((folder / image_id).exists() for folder in known):
            return {"duplicate": True}
        if training.image_total() >= training.MAX_IMAGES:
            raise TrainingError(413, f"The image store is full at {training.MAX_IMAGES:,} images. Remove some first.")
        inbox(dataset).mkdir(parents=True, exist_ok=True)
        temporary = inbox(dataset) / f"{uuid.uuid4().hex}.partial"
        temporary.write_bytes(data)
        os.replace(temporary, inbox(dataset) / image_id)
    return {"duplicate": False}


def _text_suggestions_file(dataset: str = training.DEFAULT_DATASET) -> Path:
    return training.root(dataset) / "text" / "suggestions.json"


def _meta(dataset: str) -> dict:
    if dataset == training.DEFAULT_DATASET:
        return {"kind": "image"}
    from . import datasets  # imported here because datasets imports this module
    return datasets._read(dataset)


def _suggester(dataset: str, meta: dict | None = None) -> training.Job | None:
    """The newest finished model on this dataset that can sort what's waiting: a photo, pose or text model."""
    meta = meta or _meta(dataset)
    task = "text" if meta["kind"] == "text" else "pose" if meta.get("goal") == "pose" else "image" if meta["kind"] == "image" and meta.get("goal", "sort") == "sort" else None
    if task is None:
        return None
    for job in training._saved_jobs():
        if job.dataset == dataset and job.task == task and job.status == "finished" and job.hasModel:
            return job
    return None


def _suggestions(dataset: str, path: Path, job: training.Job | None) -> dict:
    """The saved suggestions, if they came from the model that would make them now.

    A value of None means the model looked but couldn't answer, such as a photo with no one in it.
    """
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return saved["answers"] if job is not None and saved.get("jobId") == job.id else {}


def _first_least_sure(ids: list[str], answers: dict) -> list[str]:
    """Suggested items first, starting with the model's least sure, because they teach it the most."""
    return sorted(ids, key=lambda item: (not answers.get(item), (answers.get(item) or {}).get("chance", 0)))


def _assist(job: training.Job | None, answers: dict, ids: list[str]) -> dict:
    return {
        "unsuggested": sum(item not in answers for item in ids),
        "suggesting": _suggesting,
        "suggestError": _suggest_error,
        "modelCategories": sorted(job.categories) if job is not None else None,
    }


def queue(dataset: str = training.DEFAULT_DATASET) -> dict:
    """Lists what's left. Suggested images come first, starting with the model's least sure."""
    job = _suggester(dataset)
    answers = _suggestions(dataset, _suggestions_file(dataset), job)
    folder = inbox(dataset)
    ids = _first_least_sure(sorted(path.name for path in folder.iterdir() if _IMAGE_ID.match(path.name)) if folder.is_dir() else [], answers)
    return {
        "items": [{"id": image_id, "suggestion": answers.get(image_id)} for image_id in ids[:MAX_QUEUE]],
        "total": len(ids),
    } | _assist(job, answers, ids)


def image_file(image_id: str, dataset: str = training.DEFAULT_DATASET) -> Path:
    path = _inbox_path(image_id, dataset)
    if not path.is_file():
        raise TrainingError(404, "That image doesn't exist.")
    return path


def require_sorting(dataset: str) -> None:
    """Photos in a “find things” dataset get boxes, so they can't be filed under a category."""
    if dataset == training.DEFAULT_DATASET:
        return
    from . import datasets  # imported here because datasets imports this module
    if datasets._read(dataset).get("goal") == "find":
        raise TrainingError(422, "In a “find things” dataset, photos get boxes rather than a category.")


def _require_category(dataset: str, category: str) -> None:
    """A label outside the dataset's categories would be stored where nothing can see it."""
    if dataset == training.DEFAULT_DATASET:
        return
    from . import datasets  # imported here because datasets imports this module
    meta = datasets._read(dataset)
    if meta.get("goal") == "find":
        raise TrainingError(422, "In a “find things” dataset, photos get boxes rather than a category.")
    if category not in meta["categories"]:
        raise TrainingError(422, f'"{category[:40]}" isn\'t one of this dataset\'s categories. Refresh the page to see the current ones.')


def assign_image(image_id: str, category: str, dataset: str = training.DEFAULT_DATASET) -> None:
    with training._lock:
        _require_category(dataset, category)
        source = image_file(image_id, dataset)
        target = training._category_folder(category, dataset)
        target.mkdir(parents=True, exist_ok=True)
        os.replace(source, target / image_id)


def unassign_image(image_id: str, category: str | None, dataset: str = training.DEFAULT_DATASET) -> None:
    """Puts an image back in the inbox. A category of None restores a removed image."""
    with training._lock:
        _inbox_path(image_id, dataset)
        if category is not None and training.is_running():
            raise TrainingError(409, "Wait for training to finish before undoing this label.")
        source = (training._category_folder(category, dataset) if category is not None else training.removed_folder(dataset)) / image_id
        if not source.is_file():
            raise TrainingError(404, "That image can't be restored any more.")
        inbox(dataset).mkdir(parents=True, exist_ok=True)
        os.replace(source, inbox(dataset) / image_id)


def remove_image(image_id: str, dataset: str = training.DEFAULT_DATASET) -> None:
    with training._lock:
        source = image_file(image_id, dataset)
        folder = training.removed_folder(dataset)
        folder.mkdir(parents=True, exist_ok=True)
        os.replace(source, folder / image_id)
        # Undo only reaches back so far, so older removals are deleted for good.
        for old in sorted(folder.iterdir(), key=lambda path: path.stat().st_mtime)[:-REMOVED_KEPT]:
            old.unlink(missing_ok=True)


def clear_images(dataset: str = training.DEFAULT_DATASET) -> None:
    with training._lock:
        folder = inbox(dataset)
        for path in (folder.iterdir() if folder.is_dir() else []):
            path.unlink(missing_ok=True)
        _suggestions_file(dataset).unlink(missing_ok=True)


def is_suggesting() -> bool:
    return _suggesting


# Runs in a pose model's folder: its answer for each photo, or None when no one could be seen.
_POSE_SUGGEST = """
import contextlib, json, sys
from pathlib import Path
sys.path.insert(0, '.')
with contextlib.redirect_stdout(sys.stderr):
    from model import PoseSorter, load_image
    sorter = PoseSorter()
answers = {}
for path in sorted(Path(sys.argv[1]).iterdir()):
    try:
        found = sorter.check(load_image(path))
    except Exception:
        continue
    answers[path.name] = found['answers'][0] if found['found'] else None
print(json.dumps(answers))
"""

# Runs in a text model's folder. It reads [id, text] pairs and answers them in batches.
_TEXT_SUGGEST = """
import contextlib, json, sys
sys.path.insert(0, '.')
with contextlib.redirect_stdout(sys.stderr):
    from model import answers, load
    encoder, head, labels = load()
rows = json.loads(sys.stdin.read())
found = {}
for start in range(0, len(rows), 64):
    batch = rows[start:start + 64]
    for (item, _), result in zip(batch, answers(encoder, head, labels, [text for _, text in batch])):
        found[item] = result[0]
print(json.dumps(found))
"""


def _run_suggestions(job: training.Job, command: list[str], given: str | None, path: Path) -> None:
    global _suggesting, _suggest_error
    folder = training._folder(job)
    try:
        done = subprocess.run(
            command, cwd=folder, env=training._environment(folder), input=given, stdin=None if given is not None else subprocess.DEVNULL,
            capture_output=True, text=True, errors="replace", timeout=SUGGEST_LIMIT_SECONDS,
        )
        if done.returncode != 0:
            lines = [line for line in done.stderr.splitlines() if line.strip()]
            raise ValueError(lines[-1][:600] if lines else "The model could not read them.")
        reply = json.loads(done.stdout.strip().splitlines()[-1])
        if isinstance(reply, list):  # a photo model's predict.py lists every file with its answers
            reply = {Path(row["file"]).name: row["answers"][0] for row in reply}
        answers = {item: None if best is None else {"category": best["category"], "chance": best["chance"]} for item, best in reply.items()}
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps({"jobId": job.id, "answers": answers}), encoding="utf-8")
        os.replace(temporary, path)
    except (subprocess.TimeoutExpired, OSError, ValueError, KeyError, IndexError, TypeError) as problem:
        _suggest_error = f"The model couldn't make suggestions: {problem}"
    finally:
        _suggesting = False


def _inline(script: str, *arguments: str) -> list[str]:
    import sys
    return [sys.executable, "-I", "-u", "-c", script, *arguments]


def start_suggestions(dataset: str = training.DEFAULT_DATASET) -> None:
    """Has the dataset's newest model guess a category for everything still unsorted."""
    global _suggesting, _suggest_error
    with training._lock:
        if _suggesting:
            raise TrainingError(409, "Suggestions are already being made.")
        if training.is_running():
            raise TrainingError(409, "Wait for training to finish. Both need this computer's full attention.")
        meta = _meta(dataset)
        job = _suggester(dataset, meta)
        if job is None:
            raise TrainingError(409, "Train a model on this dataset first. It makes the suggestions.")
        if meta["kind"] == "text":
            with closing(_database(dataset)) as database:
                rows = database.execute("SELECT id, text FROM items WHERE label IS NULL AND removed = 0").fetchall()
            if not rows:
                raise TrainingError(409, "There's no unsorted text to suggest labels for.")
            command, given, path = _inline(_TEXT_SUGGEST), json.dumps(rows), _text_suggestions_file(dataset)
        else:
            images = inbox(dataset)
            if not images.is_dir() or not any(images.iterdir()):
                raise TrainingError(409, "There are no unsorted images to suggest labels for.")
            folder = str(images.resolve())
            command = _inline(_POSE_SUGGEST, folder) if job.task == "pose" else training._python_command("predict.py", "--folder", folder, "--json")
            given, path = None, _suggestions_file(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        _suggesting, _suggest_error = True, None
    threading.Thread(target=_run_suggestions, args=(job, command, given, path), daemon=True).start()


# ------------------------------------------------------------------------ text

def _database(dataset: str = training.DEFAULT_DATASET) -> sqlite3.Connection:
    folder = training.root(dataset) / "text"
    folder.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(folder / "labels.sqlite")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS items (seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, text TEXT NOT NULL, "
        "label TEXT, removed INTEGER NOT NULL DEFAULT 0)"
    )
    return connection


def _read_texts(data: bytes, kind: str) -> tuple[list[str], str | None]:
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise TrainingError(422, "This file isn't UTF-8 text. Save it as UTF-8 and try again.") from None
    if kind != "csv":
        return [line.strip() for line in content.splitlines()], None
    try:
        csv.field_size_limit(MAX_TEXT_FILE_BYTES)
        rows = csv.reader(io.StringIO(content, newline=""))
        header = next(rows, None)
        if not header:
            raise TrainingError(422, "This CSV file is empty.")
        lowered = [name.strip().lower() for name in header]
        column = lowered.index("text") if "text" in lowered else 0
        return [row[column].strip() for row in rows if len(row) > column], header[column].strip()
    except csv.Error:
        raise TrainingError(422, "This CSV file couldn't be read.") from None


def add_text(data: bytes, kind: str, dataset: str = training.DEFAULT_DATASET) -> dict:
    _require_kind(dataset, "text")
    texts, column = _read_texts(data, kind)
    added = duplicates = too_long = 0
    with closing(_database(dataset)) as database, database:
        total = database.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        for text in texts:
            if not text:
                continue
            if len(text) > MAX_TEXT_LENGTH:
                too_long += 1
                continue
            if total + added >= MAX_TEXT_ITEMS:
                raise TrainingError(413, f"The text store is full at {MAX_TEXT_ITEMS:,} items. Remove some first.")
            item_id = hashlib.sha256(text.encode()).hexdigest()[:32]
            inserted = database.execute("INSERT OR IGNORE INTO items (id, text) VALUES (?, ?)", (item_id, text)).rowcount
            added += inserted
            duplicates += 1 - inserted
    return {"added": added, "duplicates": duplicates, "tooLong": too_long, "column": column}


def import_texts(rows: list[tuple[str, str]], dataset: str = training.DEFAULT_DATASET) -> list[str]:
    """Stores texts that already have a label. Returns "added" or "duplicate" for each one."""
    _require_kind(dataset, "text")
    outcomes = []
    with closing(_database(dataset)) as database, database:
        total = database.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        for text, label in rows:
            if total >= MAX_TEXT_ITEMS:
                raise TrainingError(413, f"The text store is full at {MAX_TEXT_ITEMS:,} items.")
            item_id = hashlib.sha256(text.encode()).hexdigest()[:32]
            inserted = database.execute("INSERT OR IGNORE INTO items (id, text, label) VALUES (?, ?, ?)", (item_id, text, label)).rowcount
            total += inserted
            if not inserted:
                database.execute("UPDATE items SET label = ? WHERE id = ? AND label IS NULL", (label, item_id))
            outcomes.append("added" if inserted else "duplicate")
    return outcomes


def text_queue(categories: list[str], dataset: str = training.DEFAULT_DATASET, assist: bool = True) -> dict:
    """What's left to label. With suggestions, the texts the model is least sure about come first."""
    job = _suggester(dataset) if assist else None
    answers = _suggestions(dataset, _text_suggestions_file(dataset), job) if job is not None else {}
    with closing(_database(dataset)) as database:
        counts = dict(database.execute("SELECT label, COUNT(*) FROM items WHERE label IS NOT NULL AND removed = 0 GROUP BY label").fetchall())
        if answers:
            ids = _first_least_sure([row[0] for row in database.execute("SELECT id FROM items WHERE label IS NULL AND removed = 0 ORDER BY seq")], answers)
            texts = texts_by_id(ids[:MAX_QUEUE], dataset)
            rows = [(item_id, texts[item_id]["text"]) for item_id in ids[:MAX_QUEUE] if item_id in texts]
            total = len(ids)
        else:
            rows = database.execute(
                "SELECT id, text FROM items WHERE label IS NULL AND removed = 0 ORDER BY seq LIMIT ?", (MAX_QUEUE,)
            ).fetchall()
            total = database.execute("SELECT COUNT(*) FROM items WHERE label IS NULL AND removed = 0").fetchone()[0]
            ids = None
    reply = {
        "items": [{"id": item_id, "text": text, "suggestion": answers.get(item_id)} for item_id, text in rows],
        "total": total,
        "counts": {name: counts.get(name, 0) for name in categories},
        "otherLabels": sum(amount for name, amount in counts.items() if name not in categories),
    }
    if assist:
        if ids is None:
            with closing(_database(dataset)) as database:
                ids = [row[0] for row in database.execute("SELECT id FROM items WHERE label IS NULL AND removed = 0")] if job is not None else []
        reply |= _assist(job, answers, ids)
    return reply


def texts_by_id(ids: list[str], dataset: str) -> dict[str, dict]:
    """The texts still in a dataset, with their current labels."""
    with closing(_database(dataset)) as database:
        found = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for item_id, text, label in database.execute(f"SELECT id, text, label FROM items WHERE removed = 0 AND id IN ({marks})", chunk):
                found[item_id] = {"text": text, "label": label}
    return found


def text_samples(category: str, limit: int, dataset: str) -> list[dict]:
    with closing(_database(dataset)) as database:
        rows = database.execute("SELECT id, text FROM items WHERE label = ? AND removed = 0 ORDER BY seq LIMIT ?", (category, limit)).fetchall()
    return [{"id": item_id, "text": text} for item_id, text in rows]


def _update_text(dataset: str, item_id: str, statement: str, *values: object) -> None:
    with closing(_database(dataset)) as database, database:
        if database.execute(f"UPDATE items SET {statement} WHERE id = ?", (*values, item_id)).rowcount == 0:
            raise TrainingError(404, "That text doesn't exist.")


def assign_text(item_id: str, category: str, dataset: str = training.DEFAULT_DATASET) -> None:
    _require_category(dataset, category)
    _update_text(dataset, item_id, "label = ?", category)


def unassign_text(item_id: str, dataset: str = training.DEFAULT_DATASET) -> None:
    _update_text(dataset, item_id, "label = NULL, removed = 0")


def remove_text(item_id: str, dataset: str = training.DEFAULT_DATASET) -> None:
    _update_text(dataset, item_id, "removed = 1")


def rename_text_label(old: str, new: str, dataset: str = training.DEFAULT_DATASET) -> None:
    with closing(_database(dataset)) as database, database:
        database.execute("UPDATE items SET label = ? WHERE label = ?", (new, old))


def clear_text_label(name: str, dataset: str = training.DEFAULT_DATASET) -> None:
    with closing(_database(dataset)) as database, database:
        database.execute("UPDATE items SET label = NULL WHERE label = ?", (name,))


def clear_text(dataset: str = training.DEFAULT_DATASET) -> None:
    with closing(_database(dataset)) as database, database:
        database.execute("DELETE FROM items")
    _text_suggestions_file(dataset).unlink(missing_ok=True)


def export_text(categories: list[str], dataset: str = training.DEFAULT_DATASET) -> str:
    """Writes the text,label CSV that a generated text project reads as data/texts.csv."""
    with closing(_database(dataset)) as database:
        rows = database.execute("SELECT text, label FROM items WHERE label IS NOT NULL AND removed = 0 ORDER BY seq").fetchall()
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["text", "label"])
    writer.writerows((text, label) for text, label in rows if label in categories)
    return buffer.getvalue()
