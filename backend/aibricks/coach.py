"""The data coach: how much more labelled data would help, and which examples to label next.

The work happens in coach_check.py, in its own process, like the label check. This module starts it,
keeps its result, turns the result into advice, and orders the labelling queue so the examples the
coach picked come first. Nothing here changes a label.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

from . import datasets, labelling, opendata, training, versions
from .training import TrainingError

LIMIT_SECONDS = 60 * 60
MIN_PER_CATEGORY = 5
MAX_LABELLED = 4000
MAX_UNLABELLED = 3000
ROUGH_BELOW = 60  # with fewer labelled examples, the curve is a rough guide
HELPS = 0.04  # doubling the data is expected to add at least this much accuracy
LEVELLING = 0.015  # or less than this
FOCUS_GAIN = 0.01  # at least this much, and
FOCUS_SHARE = 1.5  # half as much again as the average category's gain, to stand out
STALE_SHARE = 0.15
_SCRIPT = Path(__file__).with_name("coach_check.py")
_lock = threading.Lock()
_job: dict | None = None


def _folder(dataset: str) -> Path:
    return training.root(dataset) / "coach"


def _read(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def is_running(dataset: str | None = None) -> bool:
    return bool(_job and _job["status"] == "running" and (dataset is None or _job["dataset"] == dataset))


def _suits(meta: dict) -> bool:
    return meta["kind"] in ("image", "text") and meta.get("goal", "sort") == "sort"


def _require(dataset: str) -> dict:
    meta = datasets._read(dataset)
    if not _suits(meta):
        raise TrainingError(422, "The data coach is for photo and text datasets that sort into categories.")
    return meta


def _unlabelled(dataset: str, meta: dict) -> list[str]:
    if meta["kind"] == "image":
        folder = labelling.inbox(dataset)
        return sorted(path.name for path in folder.iterdir() if labelling._IMAGE_ID.match(path.name)) if folder.is_dir() else []
    with closing(labelling._database(dataset)) as database:
        return [row[0] for row in database.execute("SELECT id FROM items WHERE label IS NULL AND removed = 0 ORDER BY seq")]


def start(dataset: str) -> dict:
    global _job
    from . import health  # imported here, because health imports datasets, which this module also uses
    meta = _require(dataset)
    with _lock, training._lock:
        if is_running():
            raise TrainingError(409, "The data coach is already running.")
        if health.is_checking():
            raise TrainingError(409, "Wait for the label check to finish.")
        if training.is_running():
            raise TrainingError(409, "Wait for training to finish. Both need this computer's full attention.")
        job = opendata.status(dataset)["job"]
        if job and job["status"] == "running":
            raise TrainingError(409, "Wait for the import into this dataset to finish.")
        current = health._current(dataset, meta)
        single = {item: next(iter(labels)) for item, labels in current.items() if len(labels) == 1}
        names = [name for name in meta["categories"] if sum(label == name for label in single.values()) >= MIN_PER_CATEGORY]
        if len(names) < 2:
            raise TrainingError(422, f"Label at least {MIN_PER_CATEGORY} examples in each of 2 categories first.")
        chosen = sorted((item, label) for item, label in single.items() if label in names)
        if len(chosen) > MAX_LABELLED:
            chosen = sorted(random.Random(0).sample(chosen, MAX_LABELLED))
        waiting = _unlabelled(dataset, meta)[:MAX_UNLABELLED]
        work = _folder(dataset) / "work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        if meta["kind"] == "image":
            # Linked copies, so labelling meanwhile can't move files out from under it.
            items, unlabelled = [], []
            for item, label in chosen:
                versions._link(training._category_folder(label, dataset) / item, work / "images" / item)
                items.append({"id": item, "label": label, "path": str(work / "images" / item)})
            for item in waiting:
                versions._link(labelling.inbox(dataset) / item, work / "inbox" / item)
                unlabelled.append({"id": item, "path": str(work / "inbox" / item)})
        else:
            texts = labelling.texts_by_id([item for item, _ in chosen] + waiting, dataset)
            items = [{"id": item, "label": label, "text": texts[item]["text"]} for item, label in chosen if item in texts]
            unlabelled = [{"id": item, "text": texts[item]["text"]} for item in waiting if item in texts]
        (work / "manifest.json").write_text(json.dumps({"kind": meta["kind"], "categories": names, "items": items, "unlabelled": unlabelled}), encoding="utf-8")
        _job = {"dataset": dataset, "status": "running", "stepsDone": 0, "stepsTotal": 8, "error": None, "startedAt": time.time()}
    threading.Thread(target=_run, args=(dataset, work), daemon=True).start()
    return _job


def _run(dataset: str, work: Path) -> None:
    command = [sys.executable, "-I", "-u", str(_SCRIPT), str(work)]
    try:
        process = subprocess.Popen(command, cwd=work, env=training._environment(work), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        timer = threading.Timer(LIMIT_SECONDS, process.terminate)
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
            raise RuntimeError(lines[-1][:600] if lines else "The data coach stopped unexpectedly.")
        result = json.loads((work / "result.json").read_text(encoding="utf-8")) | {"ranAt": time.time()}
        versions._write_json(_folder(dataset) / "result.json", result)
        _job["status"] = "finished"
    except (OSError, ValueError, RuntimeError) as problem:
        _job["status"], _job["error"] = "failed", str(problem)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def advice(result: dict) -> dict:
    """What the curve says, in a word: more data helps, helps a little, or is levelling off.

    Each category's own curve is noisier than the overall one, so the categories' gains are scaled to
    add up to the overall gain from doubling everything. They then only decide how it splits, and a
    category whose share stands out is worth adding to first.
    """
    curve = result.get("curve")
    if not curve:
        return {"verdict": "unknown", "gain": None, "focus": [], "rough": result["labelled"] < ROUGH_BELOW}
    gain = curve["more"][1]["accuracy"] - curve["now"]["accuracy"]  # with twice the data
    verdict = "helps" if gain >= HELPS else "levelling" if gain < LEVELLING else "some"
    rows = [dict(row) for row in result["categories"]]
    known = [row for row in rows if row["gainIfDoubled"] is not None]
    weight = sum(row.get("count", 1) for row in known)
    average = sum(row["gainIfDoubled"] * row.get("count", 1) for row in known) / weight if weight else 0
    for row in known:
        row["gainIfDoubled"] = round(row["gainIfDoubled"] * max(gain, 0) / average, 4) if average > 0 else 0.0
    overall = max(gain, 0)
    focus = sorted((row for row in known if row["gainIfDoubled"] >= max(FOCUS_SHARE * overall, FOCUS_GAIN)),
                   key=lambda row: -row["gainIfDoubled"])
    return {"verdict": verdict, "gain": round(gain, 4), "focus": [row["category"] for row in focus],
            "rough": result["labelled"] < ROUGH_BELOW, "categories": rows}


def overview(dataset: str) -> dict:
    from . import health
    meta = _require(dataset)
    job = _job if _job and _job["dataset"] == dataset else None
    result = _read(_folder(dataset) / "result.json", None)
    labelled_now = sum(1 for labels in health._current(dataset, meta).values() if len(labels) == 1)
    reply = {"job": job, "labelledNow": labelled_now, "minPerCategory": MIN_PER_CATEGORY, "result": None}
    if result:
        waiting = set(_unlabelled(dataset, meta))
        still = [row for row in result["next"] if row["id"] in waiting]
        picked = [dict(row) for row in still if row["picked"]]
        if meta["kind"] == "text":
            texts = labelling.texts_by_id([row["id"] for row in picked], dataset)
            for row in picked:
                row["text"] = texts.get(row["id"], {}).get("text", "")
        reply["result"] = {
            "ranAt": result["ranAt"], "labelled": result["labelled"], "curve": result["curve"], "categories": result["categories"],
            "next": picked, "unlabelledRanked": len(still),
            "stale": abs(labelled_now - result["labelled"]) > max(10, STALE_SHARE * result["labelled"]),
        } | advice(result)
    return reply


def order(dataset: str) -> dict[str, int] | None:
    """Where each unlabelled example comes in the coach's ranking, most useful first, if it has run."""
    result = _read(_folder(dataset) / "result.json", None)
    if not result or not result.get("next"):
        return None
    return {row["id"]: position for position, row in enumerate(result["next"])}
