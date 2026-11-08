"""Data health: quality levels, the label check, and acting on what it finds.

The label check scores every example with a model that never saw it (health_check.py). Its
flags are suggestions for a person to review. In research on public datasets about half of
such flags were real mistakes, so nothing here changes a label without the user choosing to.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

from . import datasets, labelling, opendata, training, versions
from .training import TrainingError

CHECK_LIMIT_SECONDS = 60 * 60
SOLID_PER_CATEGORY = 20
SOLID_BALANCE = 10
CHECKED_OPEN_SHARE = 0.05
LEVELS = ["Not ready", "Started", "Solid", "Checked", "Reviewed"]
_SCRIPT = Path(__file__).with_name("health_check.py")
_lock = threading.Lock()
_job: dict | None = None


def _folder(dataset: str) -> Path:
    return training.root(dataset) / "health"


def _read(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _write(path: Path, value) -> None:
    versions._write_json(path, value)


def is_checking(dataset: str | None = None) -> bool:
    return bool(_job and _job["status"] == "running" and (dataset is None or _job["dataset"] == dataset))


# ----------------------------------------------------------------- what's there now

def _current(dataset: str, meta: dict) -> dict[str, set]:
    """Each labelled example's ID and its current labels. A photo can be in two categories."""
    found: dict[str, set] = {}
    if meta["kind"] == "image":
        for name in meta["categories"]:
            for path in training._images_in(training._category_folder(name, dataset)):
                found.setdefault(path.name, set()).add(name)
    else:
        with closing(labelling._database(dataset)) as database:
            for item_id, label in database.execute("SELECT id, label FROM items WHERE label IS NOT NULL AND removed = 0"):
                if label in meta["categories"]:
                    found.setdefault(item_id, set()).add(label)
    return found


def _exact_conflicts(dataset: str, meta: dict, current: dict[str, set]) -> list[dict]:
    """The same photo filed under two categories, or the same text with two different labels."""
    if meta["kind"] == "image":
        return [{"ids": [item], "labels": sorted(labels)} for item, labels in current.items() if len(labels) > 1]
    groups: dict[str, list[tuple[str, str]]] = {}
    known = labelling.texts_by_id(list(current), dataset)
    for item, entry in known.items():
        groups.setdefault(" ".join(entry["text"].casefold().split()), []).append((item, entry["label"]))
    return [
        {"ids": [item for item, _ in rows], "labels": sorted({label for _, label in rows})}
        for rows in groups.values() if len({label for _, label in rows}) > 1
    ]


def _table_stats(dataset: str) -> dict | None:
    try:
        with open(datasets.table_file(dataset), encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            rows = list(reader)
    except (OSError, StopIteration):
        return None
    cells = len(rows) * len(header)
    blank = sum(1 for row in rows for cell in row if not cell.strip())
    constant = [name for index, name in enumerate(header) if len({row[index].strip() for row in rows}) <= 1]
    return {
        "rows": len(rows), "columns": len(header), "repeatedRows": len(rows) - len({tuple(row) for row in rows}),
        "blankShare": blank / cells if cells else 0.0, "constantColumns": constant,
    }


# --------------------------------------------------------------------- overview

def overview(dataset: str) -> dict:
    meta = datasets._read(dataset)
    job = _job if _job and _job["dataset"] == dataset else None
    if meta["kind"] == "table":
        return {"kind": "table", "table": _table_stats(dataset), "job": None}
    with training._lock:
        current = _current(dataset, meta)
        conflicts = _exact_conflicts(dataset, meta, current)
    counts = {name: 0 for name in meta["categories"]}
    for labels in current.values():
        for label in labels:
            counts[label] += 1
    labelled = len(current)
    result = _read(_folder(dataset) / "result.json", None)
    reviews = _read(_folder(dataset) / "reviews.json", {})

    issues, pairs, unchecked, open_issues, open_pairs = [], [], 0, 0, 0
    if result:
        checked = result["items"]
        flagged = {issue["id"] for issue in result["issues"]}
        unchecked = sum(1 for item, labels in current.items() if item not in checked or (checked[item] not in labels and item not in flagged))
        texts = labelling.texts_by_id([issue["id"] for issue in result["issues"]], dataset) if meta["kind"] == "text" else {}
        for issue in result["issues"]:
            labels = current.get(issue["id"])
            if labels is None:
                state = "removed"
            elif issue["given"] not in labels:
                state = "fixed"
            elif reviews.get(issue["id"]) == issue["given"]:
                state = "kept"
            else:
                state = "open"
            open_issues += state == "open"
            row = issue | {"state": state, "labels": sorted(labels or [])}
            if meta["kind"] == "text" and issue["id"] in texts:
                row["text"] = texts[issue["id"]]["text"]
            issues.append(row)
        for pair in result["duplicates"]:
            first, second = current.get(pair["first"]), current.get(pair["second"])
            if first is None or second is None:
                continue
            key = f"{pair['first']}|{pair['second']}"
            differs = first != second
            state = "kept" if reviews.get(key) else "open" if differs else "same"
            open_pairs += state == "open"
            row = pair | {"state": state, "firstLabels": sorted(first), "secondLabels": sorted(second)}
            if meta["kind"] == "text":
                found = labelling.texts_by_id([pair["first"], pair["second"]], dataset)
                row["firstText"], row["secondText"] = found.get(pair["first"], {}).get("text"), found.get(pair["second"], {}).get("text")
            pairs.append(row)

    level, steps = _level(meta, counts, labelled, conflicts, result, unchecked, open_issues, open_pairs)
    return {
        "kind": meta["kind"], "level": level, "levelName": LEVELS[level], "levels": LEVELS, "nextSteps": steps,
        "labelled": labelled, "counts": counts, "conflicts": conflicts,
        "check": None if not result else {
            "ranAt": result["ranAt"], "agreement": result["agreement"], "issues": issues, "duplicates": pairs,
            "unchecked": unchecked, "openIssues": open_issues, "openDuplicates": open_pairs,
            "passedShare": (labelled - open_issues) / labelled if labelled else None,
        },
        "job": job,
    }


def _level(meta, counts, labelled, conflicts, result, unchecked, open_issues, open_pairs) -> tuple[int, list[str]]:
    minimum = training.MIN_PER_CATEGORY if meta["kind"] == "image" else 4
    filled = [amount for amount in counts.values() if amount > 0]
    started = len(counts) >= 2 and min(counts.values(), default=0) >= minimum and labelled >= training.MIN_TOTAL
    if not started:
        return 0, [f"Add at least 2 categories, each with {minimum} or more examples, and {training.MIN_TOTAL} in total."]
    steps = []
    short = [name for name, amount in counts.items() if amount < SOLID_PER_CATEGORY]
    if short:
        steps.append(f"Add examples until each category has {SOLID_PER_CATEGORY}. Short: {', '.join(short)}.")
    if filled and max(filled) > SOLID_BALANCE * min(filled):
        steps.append(f"Even out the categories. The largest has more than {SOLID_BALANCE} times as many as the smallest.")
    if conflicts:
        steps.append(f"Resolve {len(conflicts)} example{'s' if len(conflicts) != 1 else ''} filed under more than one category.")
    if steps:
        return 1, steps
    if result is None:
        return 2, ["Run the label check."]
    if unchecked:
        return 2, [f"{unchecked} example{'s were' if unchecked != 1 else ' was'} added or changed since the last check. Run it again."]
    if open_issues > CHECKED_OPEN_SHARE * labelled:
        return 2, [f"Review flagged labels until no more than {round(CHECKED_OPEN_SHARE * 100)}% are left. {open_issues} to go."]
    if open_issues or open_pairs:
        left = [f"{open_issues} flagged label{'s' if open_issues != 1 else ''}"] if open_issues else []
        left += [f"{open_pairs} pair{'s' if open_pairs != 1 else ''} of near-duplicates with different labels"] if open_pairs else []
        return 3, [f"Review {' and '.join(left)}."]
    return 4, []


# ------------------------------------------------------------------- the check

def start(dataset: str) -> dict:
    global _job
    meta = datasets._read(dataset)
    if meta["kind"] == "table":
        raise TrainingError(422, "The label check is for image and text datasets.")
    with _lock, training._lock:
        if is_checking():
            raise TrainingError(409, "A label check is already running.")
        if training.is_running():
            raise TrainingError(409, "Wait for training to finish. Both need this computer's full attention.")
        job = opendata.status(dataset)["job"]
        if job and job["status"] == "running":
            raise TrainingError(409, "Wait for the import into this dataset to finish.")
        current = _current(dataset, meta)
        single = {item: next(iter(labels)) for item, labels in current.items() if len(labels) == 1}
        names = [name for name in meta["categories"] if sum(label == name for label in single.values()) >= 2]
        if len(names) < 2:
            raise TrainingError(422, "Label at least 2 examples in each of 2 categories first.")
        work = _folder(dataset) / "work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        items = []
        if meta["kind"] == "image":
            # Linked copies, so labelling during the check can't move files out from under it.
            for item, label in single.items():
                if label in names:
                    copy = work / "images" / item
                    versions._link(training._category_folder(label, dataset) / item, copy)
                    items.append({"id": item, "label": label, "path": str(copy)})
        else:
            texts = labelling.texts_by_id(list(single), dataset)
            items = [{"id": item, "label": label, "text": texts[item]["text"]} for item, label in single.items() if label in names and item in texts]
        (work / "manifest.json").write_text(json.dumps({"kind": meta["kind"], "categories": names, "items": items}), encoding="utf-8")
        _job = {"dataset": dataset, "status": "running", "stepsDone": 0, "stepsTotal": 3, "error": None, "startedAt": time.time(), "count": len(items)}
        snapshot = {item["id"]: item["label"] for item in items}
    threading.Thread(target=_run, args=(dataset, work, snapshot), daemon=True).start()
    return _job


def _run(dataset: str, work: Path, snapshot: dict[str, str]) -> None:
    command = [sys.executable, "-I", "-u", str(_SCRIPT), str(work)]
    try:
        process = subprocess.Popen(command, cwd=work, env=training._environment(work), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        timer = threading.Timer(CHECK_LIMIT_SECONDS, process.terminate)
        timer.start()
        for line in process.stdout:
            match = training._STEP_LINE.match(line.strip())
            if match:
                _job["stepsDone"], _job["stepsTotal"] = int(match[1]), int(match[2])
        errors = process.stderr.read()
        code = process.wait()
        timer.cancel()
        if code != 0:
            lines = [line for line in errors.splitlines() if line.strip()]
            raise RuntimeError(lines[-1][:600] if lines else "The label check stopped unexpectedly.")
        result = json.loads((work / "result.json").read_text(encoding="utf-8"))
        result |= {"ranAt": time.time(), "items": snapshot}
        _write(_folder(dataset) / "result.json", result)
        _job["status"] = "finished"
    except (OSError, ValueError, RuntimeError) as problem:
        _job["status"], _job["error"] = "failed", str(problem)
    finally:
        # The check's copies of the text and images aren't kept.
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ acting on it

def review(dataset: str, action: str, item: str | None, category: str | None, pair: list[str] | None) -> dict:
    meta = datasets._read(dataset)
    if meta["kind"] == "table":
        raise TrainingError(422, "The label check is for image and text datasets.")
    reviews_file = _folder(dataset) / "reviews.json"
    with training._lock:
        if training.is_running() and training.current().dataset == dataset:
            raise TrainingError(409, "Wait for training on this dataset to finish.")
        reviews = _read(reviews_file, {})
        current = _current(dataset, meta)
        if action == "keep-pair":
            if not pair or len(pair) != 2:
                raise TrainingError(422, "Choose a pair.")
            reviews[f"{pair[0]}|{pair[1]}"] = True
        else:
            if not item or item not in current:
                raise TrainingError(404, "That example is no longer in this dataset.")
            labels = current[item]
            if action == "keep":
                if category not in labels:
                    raise TrainingError(422, "That isn't this example's label.")
                reviews[item] = category
            elif action == "move":
                if category not in meta["categories"]:
                    raise TrainingError(422, "Choose one of this dataset's categories.")
                _move(dataset, meta, item, labels, category)
            elif action == "remove":
                _remove(dataset, meta, item, labels)
            elif action == "drop":
                # Takes one copy of a photo out of one category, leaving it in the others.
                if meta["kind"] != "image" or category not in labels or len(labels) < 2:
                    raise TrainingError(422, "This only applies to a photo filed under more than one category.")
                (training._category_folder(category, dataset) / item).unlink(missing_ok=True)
            else:
                raise TrainingError(422, "Unknown action.")
        _write(reviews_file, reviews)
    return overview(dataset)


def _move(dataset: str, meta: dict, item: str, labels: set, category: str) -> None:
    if meta["kind"] == "text":
        labelling.assign_text(item, category, dataset)
        return
    target = training._category_folder(category, dataset)
    target.mkdir(parents=True, exist_ok=True)
    for label in labels - {category}:
        source = training._category_folder(label, dataset) / item
        if (target / item).exists():
            source.unlink(missing_ok=True)
        else:
            os.replace(source, target / item)


def _remove(dataset: str, meta: dict, item: str, labels: set) -> None:
    if meta["kind"] == "text":
        labelling.remove_text(item, dataset)
        return
    removed = training.removed_folder(dataset)
    removed.mkdir(parents=True, exist_ok=True)
    for label in labels:
        source = training._category_folder(label, dataset) / item
        if source.exists():
            os.replace(source, removed / item)
