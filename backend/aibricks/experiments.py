"""Find a better model: tries several photo models on a dataset and explains what each change did.

Round 1 trains a starting point and four variants, each changing one thing, for the same short
time. They are checked on the same held-back photos, so each change can be compared with the
starting point using McNemar's test. Round 2 trains the two best, and a mix of the changes that
helped, for longer. This is successive halving (as in ASHA, Li et al., 2020): every option gets a
little time, and the promising ones get more.

Runs that lose are deleted once they've been measured, so they don't push your own models out of
the ones kept. The winner stays as a normal model, and its blocks can be opened in the builder.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from . import codegen, compare, datasets, labelling, training, versions
from .graph import GraphIn, analyse
from .training import TrainingError

BUDGETS = (5, 10, 20, 30)  # minutes
SHORT_EPOCHS = 5
LONG_EPOCHS = 15
STEP_LEARNING_RATE = 3e-4
SCRATCH_SIZE = 64
PRETRAINED_SIZE = 128
KEPT_EXPERIMENTS = 5

_lock = threading.Lock()
_running: dict | None = None
_worker: threading.Thread | None = None
_stop = threading.Event()

# Each change is something a person could do themselves in the builder.
CHANGES = {
    "augment": ("Vary training photos", "Random crops, flips and lighting on the training photos."),
    "wider": ("Wider model", "Three convolutions with more filters, so it can learn finer patterns."),
    "pretrained": ("Start from a pretrained model", "ResNet-18, which already learned from a million photos, with only the last layer trained."),
    "slower": ("Learn more slowly", f"A learning rate of {STEP_LEARNING_RATE:g} instead of 0.001, for steadier steps."),
}


def _folder(dataset: str) -> Path:
    return training.root(dataset) / "experiments"


def _save(state: dict) -> None:
    folder = _folder(state["dataset"])
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{uuid.uuid4().hex}.partial"
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(folder / f"{state['id']}.json")


def _node(index: int, brick: str, **params) -> dict:
    return {"id": f"n{index}", "type": brick, "params": params}


def graph_for(name: str, categories: list[str], changes: tuple[str, ...], epochs: int) -> dict:
    """The blocks for one option: the starting point with some changes applied."""
    classes = ", ".join(categories)
    pretrained = "pretrained" in changes
    nodes = [_node(0, "image_folder", image_size=PRETRAINED_SIZE if pretrained else SCRATCH_SIZE, channels=3, augment="augment" in changes)]
    if pretrained:
        nodes += [_node(1, "backbone", freeze=True), _node(2, "global_pool")]
    elif "wider" in changes:
        nodes += [_node(1, "conv", filters=48, kernel_size=3), _node(2, "pool", mode="max"),
                  _node(3, "conv", filters=96, kernel_size=3), _node(4, "pool", mode="max"),
                  _node(5, "conv", filters=128, kernel_size=3), _node(6, "global_pool")]
    else:
        nodes += [_node(1, "cnn_encoder")]
    nodes.append(_node(len(nodes), "classify", classes=classes))
    edges = [{"source": f"n{i}", "target": f"n{i + 1}", "targetHandle": "in"} for i in range(len(nodes) - 1)]
    override = {"epochs": epochs, "batch_size": 32, "learning_rate": STEP_LEARNING_RATE if "slower" in changes else 1e-3}
    return {"projectName": name, "training": "quick", "trainingOverride": override, "nodes": nodes, "edges": edges}


def _label(changes: tuple[str, ...]) -> str:
    return " + ".join(CHANGES[change][0] for change in changes) if changes else "Starting point"


def _project_name(dataset_name: str, changes: tuple[str, ...]) -> str:
    base = re.sub(r"[^A-Za-z0-9 _-]+", "", dataset_name).strip()[:30] or "Photos"
    tail = "-".join(sorted(changes)) if changes else "start"
    return f"{base} - {tail}"


# ------------------------------------------------------------------ public

def start(dataset: str, minutes: int) -> dict:
    global _running
    if minutes not in BUDGETS:
        raise TrainingError(422, f"Choose {', '.join(map(str, BUDGETS))} minutes.")
    meta = datasets.get(dataset)
    if meta["kind"] != "image" or meta.get("goal") == "unusual":
        raise TrainingError(422, "Finding a better model works on photo datasets that sort into categories, for now.")
    names = [name for name in meta["categories"] if meta["counts"].get(name, 0) >= training.MIN_PER_CATEGORY]
    if len(names) < 2:
        raise TrainingError(422, f"Add at least {training.MIN_PER_CATEGORY} photos to each of 2 categories first.")
    if sum(meta["counts"].get(name, 0) for name in names) < 20:
        raise TrainingError(422, "Add at least 20 photos in total first, so the options can be told apart.")
    with _lock:
        if _running and _running["status"] == "running":
            raise TrainingError(409, "An experiment is already running.")
        if training.is_running():
            raise TrainingError(409, "A model is already training. Wait for it to finish.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        _stop.clear()
        state = {
            "id": uuid.uuid4().hex[:12], "dataset": dataset, "status": "running", "minutes": minutes,
            "startedAt": time.time(), "finishedAt": None, "categories": names, "round": 1,
            "current": None, "options": [], "effects": [], "winner": None, "verdict": None, "error": None, "skipped": [],
        }
        _running = state
        _save(state)
    global _worker
    _worker = threading.Thread(target=_run, args=(state, meta["name"]), daemon=True)
    _worker.start()
    return state


def stop(dataset: str) -> dict:
    state = _running
    if not state or state["dataset"] != dataset or state["status"] != "running":
        raise TrainingError(404, "No experiment is running on this dataset.")
    _stop.set()
    job = training.current()
    if job and job.status == "running" and state.get("current") == job.id:
        training.stop(job.id)
    return state


def latest(dataset: str) -> dict | None:
    if _running and _running["dataset"] == dataset:
        return _running
    folder = _folder(dataset)
    found = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True) if folder.is_dir() else []
    for path in found:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def is_running() -> bool:
    return bool(_running and _running["status"] == "running")


def blocks(dataset: str | None = None) -> str | None:
    """Why something else must wait while an experiment runs. Its own training is never blocked."""
    state = _running
    if not state or state["status"] != "running" or threading.current_thread() is _worker:
        return None
    if dataset is None or state["dataset"] == dataset:
        return "Wait for “Find a better model” to finish, or stop it."
    return None


# ------------------------------------------------------------------ the work

def _train(state: dict, dataset_name: str, changes: tuple[str, ...], epochs: int, round_number: int) -> dict | None:
    """Trains one option to the end and returns what it scored, or None if stopped."""
    graph = graph_for(_project_name(dataset_name, changes), state["categories"], changes, epochs)
    payload = GraphIn(**graph)
    analysis = analyse(payload)
    if not analysis.ok:
        raise TrainingError(500, f"The {_label(changes)} option isn't a valid build: {analysis.issues[0].message}")
    files = codegen.generate(payload, analysis)
    if labelling.is_suggesting():
        raise TrainingError(409, "Wait for the model to finish making suggestions.")
    started = time.time()
    job = training.start(graph["projectName"], codegen.slug(graph["projectName"]), codegen.build_hash(files), files,
                         state["categories"], epochs, state["dataset"], versions.freeze)
    state["current"] = job.id
    state["currentLabel"] = _label(changes)
    _save(state)
    while True:
        time.sleep(0.5)
        live = training.stored(job.id)
        if live.status != "running":
            break
        state["progress"] = {"epoch": len(live.epochs), "epochs": epochs}
    state["current"], state["progress"] = None, None
    if live.status != "finished":
        if live.status == "failed" and not _stop.is_set():
            raise TrainingError(500, f"The {_label(changes)} option failed to train: {live.error}")
        return None
    review = live.review or {}
    return {
        "key": "+".join(changes) or "start", "changes": list(changes), "label": _label(changes), "round": round_number,
        "epochs": epochs, "jobId": job.id, "seconds": round(time.time() - started, 1),
        "accuracy": (review["correct"] / review["total"]) if review.get("total") else (live.best or {}).get("accuracy"),
        "correct": review.get("correct"), "total": review.get("total"),
        "checkLoss": (live.best or {}).get("checkLoss"), "modelBytes": training._model_bytes(live),
        "graph": graph,
    }


def _rank(option: dict) -> tuple:
    return (option["accuracy"] or 0, -(option["checkLoss"] if option["checkLoss"] is not None else 1e9))


def _effect(state: dict, base: dict, other: dict) -> dict:
    result = compare.compare(state["dataset"], other["jobId"], base["jobId"])
    gain = round(((other["accuracy"] or 0) - (base["accuracy"] or 0)) * 100)
    if result["verdict"] == "first":
        meaning = "helped"
    elif result["verdict"] == "second":
        meaning = "hurt"
    elif result["verdict"] == "too-few":
        meaning = "too few photos to tell"
    else:
        meaning = "no clear difference"
    return {"key": other["key"], "label": other["label"], "description": CHANGES[other["changes"][0]][1],
            "points": gain, "meaning": meaning, "chanceOfLuck": result.get("chanceOfLuck"), "shared": result.get("shared")}


def _discard(option: dict) -> None:
    try:
        training.delete_model(option["jobId"])
    except TrainingError:
        pass
    option["deleted"] = True


def _fits(state: dict, seconds: float) -> bool:
    return time.time() - state["startedAt"] + seconds <= state["minutes"] * 60


def _run(state: dict, dataset_name: str) -> None:
    global _running
    try:
        # Round 1: the same short training for every option, one change at a time.
        base = _train(state, dataset_name, (), SHORT_EPOCHS, 1)
        if base is None:
            raise InterruptedError
        state["options"].append(base)
        _save(state)
        for change in CHANGES:
            estimate = base["seconds"] * (3 if change == "pretrained" else 1.5 if change == "wider" else 1)
            if _stop.is_set():
                raise InterruptedError
            if not _fits(state, estimate):
                state["skipped"].append(CHANGES[change][0])
                continue
            option = _train(state, dataset_name, (change,), SHORT_EPOCHS, 1)
            if option is None:
                raise InterruptedError
            state["options"].append(option)
            state["effects"].append(_effect(state, base, option))
            _save(state)

        # Round 2: the two best, and a mix of what helped, trained for longer.
        state["round"] = 2
        helped = [effect["key"] for effect in state["effects"] if effect["meaning"] == "helped" or effect["points"] >= 5]
        if "pretrained" in helped and "wider" in helped:
            helped.remove("wider")  # They replace the same blocks, so they can't be mixed.
        finalists = [tuple(option["changes"]) for option in sorted(state["options"], key=_rank, reverse=True)[:2]]
        if len(helped) >= 2 and tuple(sorted(helped)) not in {tuple(sorted(f)) for f in finalists}:
            finalists.append(tuple(sorted(helped)))
        for option in state["options"]:
            _discard(option)
        longer = []
        for changes in finalists:
            seconds = next((o["seconds"] for o in state["options"] if tuple(o["changes"]) == changes), base["seconds"] * 3)
            estimate = seconds * LONG_EPOCHS / SHORT_EPOCHS
            if _stop.is_set():
                break
            if not _fits(state, estimate):
                state["skipped"].append(f"{_label(changes)}, trained longer")
                continue
            option = _train(state, dataset_name, changes, LONG_EPOCHS, 2)
            if option is None:
                break
            longer.append(option)
            state["options"].append(option)
            _save(state)
        if not longer:
            # Not enough time for longer training: the best short run is retrained so a model is kept.
            best = max(state["options"], key=_rank)
            if not _stop.is_set():
                option = _train(state, dataset_name, tuple(best["changes"]), SHORT_EPOCHS, 2)
                if option:
                    longer.append(option)
                    state["options"].append(option)
        if longer:
            longer.sort(key=_rank, reverse=True)
            winner = longer[0]
            state["winner"] = winner["key"]
            if len(longer) > 1:
                result = compare.compare(state["dataset"], winner["jobId"], longer[1]["jobId"])
                state["verdict"] = {"against": longer[1]["label"], "result": result["verdict"], "chanceOfLuck": result.get("chanceOfLuck")}
            for option in longer[1:]:
                _discard(option)
        state["status"] = "stopped" if _stop.is_set() else "finished"
    except InterruptedError:
        state["status"] = "stopped"
    except TrainingError as error:
        state["status"], state["error"] = "failed", error.message
    except Exception as error:  # A failed experiment must say why, not hang as running.
        state["status"], state["error"] = "failed", f"{type(error).__name__}: {error}"[:300]
    finally:
        state["finishedAt"] = time.time()
        state["current"] = None
        if state["status"] != "finished":
            for option in state["options"]:
                if not option.get("deleted") and option["key"] != state.get("winner"):
                    _discard(option)
        _save(state)
        with _lock:
            _running = None
        _tidy(state["dataset"])


def _tidy(dataset: str) -> None:
    folder = _folder(dataset)
    old = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[KEPT_EXPERIMENTS:]
    for path in old:
        path.unlink(missing_ok=True)
