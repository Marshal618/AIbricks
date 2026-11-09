"""Trains a build on the user's own images, on this computer.

The generated train.py and predict.py do the work, so a model trained here behaves
exactly like one trained from the downloaded project. Images are stored under the
workspace folder, and nothing is sent anywhere.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from . import explain

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_IMAGES = 20_000
MIN_PER_CATEGORY = 2
MIN_TOTAL = 10
SUGGESTED_PER_CATEGORY = 20
TIME_LIMIT_SECONDS = 6 * 60 * 60
REVIEW_LIMIT_SECONDS = 30 * 60
MISTAKES_KEPT = 500
PREDICT_LIMIT_SECONDS = 120
MODELS_KEPT = 10

_FORMATS = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "BMP": ".bmp", "WEBP": ".webp"}
_EXTENSIONS = set(_FORMATS.values())
_START_LINE = re.compile(r"^Training on (\d+) samples, checking on (\d+), using (\w+)\.$")
_EPOCH_LINE = re.compile(
    r"^Epoch (\d+)/(\d+)\s+training loss (\S+)\s+check loss (\S+)\s+check accuracy (\S+)%$"
)

DEFAULT_DATASET = "default"
_DATASET_ID = re.compile(r"^[0-9a-f]{12}$")
_STEP_LINE = re.compile(r"^Step (\d+)/(\d+)$")
_RUN_ID = re.compile(r"^[0-9a-f]{12}$")
_lock = threading.RLock()
_job: "Job | None" = None
_process: subprocess.Popen | None = None
_predictions = threading.BoundedSemaphore(2)
_shutting_down = False


class TrainingError(Exception):
    """A problem to show the user, with the HTTP status that describes it."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def workspace() -> Path:
    configured = os.environ.get("AI_BRICKS_WORKSPACE")
    return Path(configured) if configured else Path(__file__).resolve().parents[1] / ".workspace"


# ------------------------------------------------------------------- the dataset

def root(dataset: str = DEFAULT_DATASET) -> Path:
    """The folder holding one dataset's images and labels."""
    if dataset == DEFAULT_DATASET:
        return workspace()
    if not _DATASET_ID.match(dataset):
        raise TrainingError(404, "That dataset doesn't exist.")
    return workspace() / "collections" / dataset


def _category_folder(name: str, dataset: str = DEFAULT_DATASET) -> Path:
    return root(dataset) / "datasets" / hashlib.sha256(name.encode()).hexdigest()[:24]


def removed_folder(dataset: str = DEFAULT_DATASET) -> Path:
    """Images removed while labelling or reviewing, kept so that Undo can bring them back."""
    return root(dataset) / "inbox" / "removed"


def image_total() -> int:
    total = 0
    for base in [workspace(), *(workspace() / "collections").glob("*")]:
        for folder in [*(base / "datasets").glob("*"), base / "inbox" / "images"]:
            total += len(_images_in(folder))
    return total


def is_running() -> bool:
    job = current()
    return job is not None and job.status == "running"


def _images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [path for path in folder.iterdir() if path.suffix in _EXTENSIONS]


def read_image(data: bytes) -> str:
    """Fully decodes an image and returns its file extension, or raises TrainingError."""
    if len(data) > MAX_IMAGE_BYTES:
        raise TrainingError(413, "This image is larger than 25 MB.")
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftyphevc"):
        raise TrainingError(422, "HEIC photos aren't supported. Convert them to JPEG first.")
    try:
        with Image.open(io.BytesIO(data), formats=("JPEG", "PNG", "BMP", "WEBP")) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise TrainingError(422, "This image has more than 50 megapixels.")
            image.load()
            return _FORMATS[image.format]
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, KeyError, Image.DecompressionBombError):
        raise TrainingError(422, "This file isn't a readable JPEG, PNG, BMP or WebP image.") from None


def add_image(category: str, data: bytes, dataset: str = DEFAULT_DATASET) -> dict:
    from . import labelling  # imported here, because labelling imports this module
    labelling._require_kind(dataset, "image")
    extension = read_image(data)
    with _lock:
        if image_total() >= MAX_IMAGES:
            raise TrainingError(413, f"The image store is full at {MAX_IMAGES:,} images. Remove some first.")
        folder = _category_folder(category, dataset)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / (hashlib.sha256(data).hexdigest()[:32] + extension)
        duplicate = target.exists()
        if not duplicate:
            temporary = folder / f"{uuid.uuid4().hex}.partial"
            temporary.write_bytes(data)
            os.replace(temporary, target)
        # It has a label now, so it no longer waits to be sorted.
        (root(dataset) / "inbox" / "images" / target.name).unlink(missing_ok=True)
    return {"duplicate": duplicate, "count": len(_images_in(folder))}


def counts(names: list[str], dataset: str = DEFAULT_DATASET) -> dict[str, int]:
    return {name: len(_images_in(_category_folder(name, dataset))) for name in names}


def clear_category(category: str, dataset: str = DEFAULT_DATASET) -> None:
    with _lock:
        if _job and _job.status == "running":
            raise TrainingError(409, "Stop training before removing images.")
        shutil.rmtree(_category_folder(category, dataset), ignore_errors=True)


def readiness_problems(names: list[str], dataset: str = DEFAULT_DATASET) -> list[str]:
    problems = []
    for name in names:
        if "/" in name or name in {".", ".."} or name.startswith("."):
            problems.append(f'Rename the category "{name}". In-app training needs names that work as folder names, so no slashes or leading dots.')
    if len({name.casefold() for name in names}) != len(names):
        problems.append("Two categories differ only by capital letters. Give them different names.")
    amounts = counts(names, dataset)
    short = [name for name, amount in amounts.items() if amount < MIN_PER_CATEGORY]
    if short:
        problems.append(f"Add at least {MIN_PER_CATEGORY} images to: {', '.join(short)}.")
    if sum(amounts.values()) < MIN_TOTAL:
        problems.append(f"Add at least {MIN_TOTAL} images in total.")
    return problems


# --------------------------------------------------------------------- training

@dataclass
class Job:
    id: str
    projectName: str
    slug: str
    buildHash: str
    categories: dict[str, int]
    baseline: float
    files: list[str]
    status: str = "running"
    totalEpochs: int = 0
    trainCount: int = 0
    checkCount: int = 0
    device: str = ""
    epochs: list[dict] = field(default_factory=list)
    best: dict | None = None
    hasModel: bool = False
    error: str | None = None
    startedAt: float = 0.0
    finishedAt: float | None = None
    stopRequested: bool = False
    stage: str = "training"
    review: dict | None = None
    reviewError: str | None = None
    dataset: str = DEFAULT_DATASET
    datasetVersion: int | None = None
    modelRemoved: bool = False
    task: str = "image"
    stepsDone: int = 0
    stepsTotal: int = 0
    metrics: dict | None = None


def _folder(job: Job) -> Path:
    return workspace() / "runs" / job.id


def _model_path(job: Job) -> Path:
    return _folder(job) / ("model.pt" if job.task == "image" else "model")


def _model_bytes(job: Job) -> int:
    path = _model_path(job)
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size if path.exists() else 0


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _save(job: Job) -> None:
    temporary = _folder(job) / "job.json.partial"
    temporary.write_text(json.dumps(asdict(job)), encoding="utf-8")
    os.replace(temporary, _folder(job) / "job.json")


def _read_job(folder: Path) -> Job | None:
    """Reads a saved run. A run still marked as running stopped when the server did."""
    try:
        job = Job(**json.loads((folder / "job.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None
    if job.status == "running":
        job.status = "stopped"
        job.finishedAt = (folder / "job.json").stat().st_mtime
        job.hasModel = job.best is not None and _model_path(job).exists()
        shutil.rmtree(folder / "data", ignore_errors=True)
    return job


def _saved_jobs() -> list[Job]:
    """Every saved run, newest first."""
    live = _job
    jobs = []
    for folder in (workspace() / "runs").glob("*"):
        job = live if live is not None and live.id == folder.name else _read_job(folder)
        if job is not None:
            jobs.append(job)
    return sorted(jobs, key=lambda job: job.startedAt, reverse=True)


def _load_latest() -> Job | None:
    jobs = _saved_jobs()
    return jobs[0] if jobs else None


def current() -> Job | None:
    global _job
    with _lock:
        if _job is None:
            _job = _load_latest()
        return _job


def latest(task: str) -> Job | None:
    """The newest run of one kind, so a text model doesn't hide an image model, or the reverse."""
    with _lock:
        current()
        return next((job for job in _saved_jobs() if job.task == task), None)


def public(job: Job | None) -> dict | None:
    if job is None:
        return None
    data = asdict(job)
    data["elapsedSeconds"] = (job.finishedAt or time.time()) - job.startedAt
    del data["files"], data["slug"], data["stopRequested"]
    return data


def _model_row(job: Job) -> dict:
    row = public(job)
    del row["epochs"]
    if row["review"]:
        row["review"] = {key: row["review"][key] for key in ("total", "correct", "categories")}
    row["modelBytes"] = _model_bytes(job) if job.hasModel else 0
    return row


def saved_models(dataset: str) -> list[dict]:
    """The runs trained from a dataset, newest first, without their per-epoch detail."""
    with _lock:
        current()
        return [_model_row(job) for job in _saved_jobs() if job.dataset == dataset]


def all_models(limit: int = 50) -> list[dict]:
    """The newest runs from every dataset, for the home screen."""
    with _lock:
        current()
        return [_model_row(job) for job in _saved_jobs()[:limit]]


def delete_model(job_id: str) -> None:
    global _job
    with _lock:
        job = stored(job_id)
        if job.status == "running":
            raise TrainingError(409, "Stop this model's training first.")
        close_live(job.id)
        shutil.rmtree(_folder(job), ignore_errors=True)
        if _job is not None and _job.id == job.id:
            _job = None


def _tidy_runs(keep: Path) -> None:
    """Drops runs with no model, and the model files beyond the newest few. Their records stay."""
    with_model = []
    for folder in (workspace() / "runs").iterdir():
        if folder == keep:
            continue
        job = _read_job(folder)
        if job is None or not (job.hasModel or job.modelRemoved):
            shutil.rmtree(folder, ignore_errors=True)
        elif job.hasModel:
            with_model.append(job)
    with_model.sort(key=lambda job: job.startedAt, reverse=True)
    for job in with_model[MODELS_KEPT - 1:]:
        path = _model_path(job)
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
        job.hasModel, job.modelRemoved = False, True
        _save(job)


def _cache() -> Path:
    configured = os.environ.get("AI_BRICKS_CACHE")
    return Path(configured) if configured else workspace() / "cache"


def _environment(folder: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(folder),
        "OMP_NUM_THREADS": str(max((os.cpu_count() or 2) - 1, 1)),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TORCH_HOME": str(_cache() / "torch"),
        "HF_HOME": str(_cache() / "huggingface"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    try:
        import certifi  # Python from python.org on a Mac has no certificates of its own.
        env["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass
    return env


def _python_command(script: str, *arguments: str) -> list[str]:
    runner = f"import runpy,sys; sys.path.insert(0, '.'); runpy.run_path({script!r}, run_name='__main__')"
    return [sys.executable, "-I", "-u", "-c", runner, *arguments]


def _record(job: Job, line: str) -> None:
    step = _STEP_LINE.match(line)
    if step:
        job.stepsDone, job.stepsTotal = int(step[1]), int(step[2])
        return
    started = _START_LINE.match(line)
    if started:
        job.trainCount, job.checkCount, job.device = int(started[1]), int(started[2]), started[3]
        return
    epoch = _EPOCH_LINE.match(line)
    if not epoch:
        return
    entry = {
        "epoch": int(epoch[1]),
        "trainLoss": _number(epoch[3]),
        "checkLoss": _number(epoch[4]),
        "accuracy": None if _number(epoch[5]) is None else _number(epoch[5]) / 100,
    }
    job.totalEpochs = int(epoch[2])
    job.epochs.append(entry)
    # train.py keeps the version with the lowest check loss, so the result reports that one.
    if entry["checkLoss"] is not None and (job.best is None or entry["checkLoss"] < job.best["checkLoss"]):
        job.best = entry


def _last_error(log: Path) -> str:
    try:
        lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        lines = []
    return lines[-1][:600] if lines else "Training stopped unexpectedly."


def _review(job: Job) -> None:
    """Records how the saved model did on each held-back image."""
    folder = _folder(job)
    try:
        done = subprocess.run(
            _python_command("review.py", "--json"), cwd=folder, env=_environment(folder),
            stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace", timeout=REVIEW_LIMIT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        job.reviewError = "Checking the held-back images took too long."
        return
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        job.reviewError = lines[-1][:600] if lines else "The held-back images could not be checked."
        return
    rows = json.loads(done.stdout.strip().splitlines()[-1])
    _save_held_out(job, "category", [(Path(row["path"]).name, row["actual"], row["guess"]) for row in rows])
    _summarise_review(job, rows)


HELD_OUT_FILE = "held_out.json"


def _save_held_out(job: Job, task: str, rows: list[tuple[str, object, object]]) -> None:
    """Keeps each held-back example's answer and the model's guess, under an ID only.

    Held-back examples are picked by fingerprint, so another model on this dataset was checked on
    mostly the same ones. These files let two models be compared on exactly the examples they share.
    """
    data = {"task": task, "rows": {key: [actual, guess] for key, actual, guess in rows}}
    (_folder(job) / HELD_OUT_FILE).write_text(json.dumps(data), encoding="utf-8")


def held_out(job: Job) -> dict | None:
    try:
        return json.loads((_folder(job) / HELD_OUT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _review_text(job: Job) -> None:
    """Reads what train.py recorded. Only an ID for each text is kept with the run."""
    rows = json.loads((_folder(job) / "data" / "review.json").read_text(encoding="utf-8"))
    for row in rows:
        row["path"] = hashlib.sha256(row.pop("text").encode()).hexdigest()[:32]
    _save_held_out(job, "category", [(row["path"], row["actual"], row["guess"]) for row in rows])
    _summarise_review(job, rows)


def _review_files(job: Job) -> None:
    """Reads what a sound train.py recorded. Clips are named by their content, so the name is the ID."""
    rows = json.loads((_folder(job) / "data" / "review.json").read_text(encoding="utf-8"))
    for row in rows:
        row["path"] = row.pop("file")
    _save_held_out(job, "category", [(row["path"], row["actual"], row["guess"]) for row in rows])
    _summarise_review(job, rows)


def _review_table(job: Job) -> None:
    """Keeps train.py's scores and column importance, which hold no rows of data."""
    metrics = json.loads((_folder(job) / "data" / "metrics.json").read_text(encoding="utf-8"))
    job.metrics = metrics
    held = json.loads((_folder(job) / "data" / "held_out.json").read_text(encoding="utf-8"))
    _save_held_out(job, metrics["task"], [tuple(row) for row in held])
    if metrics["task"] == "category":
        job.review = {
            "total": metrics["total"], "correct": metrics["correct"], "categories": metrics["perClass"],
            "mistakeCount": metrics["total"] - metrics["correct"], "mistakes": [],
        }


def _summarise_review(job: Job, rows: list[dict]) -> None:
    categories: dict[str, dict[str, int]] = {name: {"correct": 0, "total": 0} for name in job.categories}
    for row in rows:
        categories[row["actual"]]["total"] += 1
        categories[row["actual"]]["correct"] += row["guess"] == row["actual"]
    wrong = sorted((row for row in rows if row["guess"] != row["actual"]), key=lambda row: row["chance"], reverse=True)
    job.review = {
        "total": len(rows),
        "correct": len(rows) - len(wrong),
        "categories": categories,
        "mistakeCount": len(wrong),
        "mistakes": [
            {"actual": row["actual"], "guess": row["guess"], "chance": row["chance"], "file": Path(row["path"]).name}
            for row in wrong[:MISTAKES_KEPT]
        ],
    }


def _watch(job: Job, process: subprocess.Popen, log: Path) -> None:
    global _process
    timer = threading.Timer(TIME_LIMIT_SECONDS, process.terminate)
    timer.start()
    try:
        for line in process.stdout:
            with _lock:
                _record(job, line.strip())
                _save(job)
        code = process.wait()
    finally:
        timer.cancel()
    with _lock:
        _process = None
        job.hasModel = job.best is not None and _model_path(job).exists()
        job.stage = "reviewing"
        _save(job)
    if job.hasModel and not _shutting_down:
        try:
            {"text": _review_text, "table": _review_table, "sound": _review_files}.get(job.task, _review)(job)
        except (OSError, ValueError, KeyError) as problem:
            job.reviewError = f"The held-back images could not be checked: {problem}"
    with _lock:
        job.finishedAt = time.time()
        if job.stopRequested:
            job.status = "stopped"
        elif code != 0:
            job.status = "failed"
            job.error = f"Training stopped after {TIME_LIMIT_SECONDS // 3600} hours." if code < 0 else _last_error(log)
        elif not job.hasModel:
            job.status = "failed"
            job.error = "The loss became invalid, so no model was saved. Try a lower learning rate or a simpler model."
        else:
            job.status = "finished"
        # The frozen copy of the data is only for this run. Stored results keep no pictures.
        shutil.rmtree(_folder(job) / "data", ignore_errors=True)
        _save(job)


def _require_idle(dataset: str) -> None:
    """Training needs the computer to itself, and a copy of data that isn't changing underneath it."""
    from . import datasets, health  # imported here, because both import this module
    if health.is_checking():
        raise TrainingError(409, "Wait for the label check to finish. Both need this computer's full attention.")
    if dataset != DEFAULT_DATASET:
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)


def start(project_name: str, slug: str, build_hash: str, files: dict[str, str], names: list[str], epochs: int, dataset: str = DEFAULT_DATASET, freeze=None) -> Job:
    """Starts training in the background. Earlier models are kept, up to MODELS_KEPT.

    freeze(dataset, names, folder) copies the data for this run and returns its version number.
    """
    with _lock:
        if current() is not None and _job.status == "running":
            raise TrainingError(409, "A model is already training. Stop it first.")
        _require_idle(dataset)
        problems = readiness_problems(names, dataset)
        if problems:
            raise TrainingError(422, " ".join(problems))
        amounts = counts(names, dataset)
        job = Job(
            id=uuid.uuid4().hex[:12], projectName=project_name, slug=slug, buildHash=build_hash,
            categories=amounts, baseline=max(amounts.values()) / sum(amounts.values()),
            files=sorted(files), totalEpochs=epochs, startedAt=time.time(), dataset=dataset,
        )
        folder = _folder(job)
        folder.mkdir(parents=True)
        for name, contents in files.items():
            (folder / name).write_text(contents, encoding="utf-8")
        images = folder / "data" / "images"
        images.mkdir(parents=True)
        if freeze is None:
            for name in names:
                (images / name).symlink_to(_category_folder(name, dataset).resolve(), target_is_directory=True)
        else:
            try:
                job.datasetVersion = freeze(dataset, names, folder)
            except Exception:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        return _launch(job, folder)


def start_run(project_name: str, slug: str, build_hash: str, files: dict[str, str], amounts: dict[str, int], dataset: str, freeze, task: str, baseline: float | None = None) -> Job:
    """Starts a text or spreadsheet run on a frozen copy of a dataset."""
    with _lock:
        if current() is not None and _job.status == "running":
            raise TrainingError(409, "A model is already training. Stop it first.")
        _require_idle(dataset)
        job = Job(
            id=uuid.uuid4().hex[:12], projectName=project_name, slug=slug, buildHash=build_hash,
            categories=amounts, baseline=max(amounts.values()) / sum(amounts.values()) if baseline is None else baseline,
            files=sorted(files), totalEpochs=1, startedAt=time.time(), dataset=dataset, task=task,
        )
        folder = _folder(job)
        (folder / "data").mkdir(parents=True)
        for name, contents in files.items():
            (folder / name).write_text(contents, encoding="utf-8")
        try:
            job.datasetVersion = freeze(dataset, list(amounts), folder)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return _launch(job, folder)


def _launch(job: Job, folder: Path) -> Job:
    global _job, _process
    _tidy_runs(folder)
    log = folder / "train.log"
    with open(log, "wb") as errors:
        process = subprocess.Popen(
            _python_command("train.py"), cwd=folder, env=_environment(folder),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors, text=True, errors="replace",
        )
    _job, _process = job, process
    _save(job)
    threading.Thread(target=_watch, args=(job, process, log), daemon=True).start()
    return job


def _require(job_id: str) -> Job:
    for task in ("image", "text", "table", "sound"):
        job = latest(task)
        if job is not None and job.id == job_id:
            return job
    raise TrainingError(404, "That training run no longer exists. Train the model again.")


def stored(job_id: str) -> Job:
    """Finds any saved run, not only the newest."""
    live = current()
    if live is not None and live.id == job_id:
        return live
    job = _read_job(workspace() / "runs" / job_id) if _RUN_ID.match(job_id) else None
    if job is None:
        raise TrainingError(404, "That training run no longer exists. Train the model again.")
    return job


def stop(job_id: str) -> Job:
    with _lock:
        job = _require(job_id)
        if job.status == "running" and _process is not None:
            job.stopRequested = True
            _process.terminate()
        return job


def stop_any() -> None:
    with _lock:
        if _process is not None and _job is not None:
            _job.stopRequested = True
            _process.terminate()


def shutdown() -> None:
    """Stops training when the server stops, without starting the review."""
    global _shutting_down
    _shutting_down = True
    stop_any()
    close_live()


def _mistake(job: Job, index: int) -> dict:
    mistakes = (job.review or {}).get("mistakes", [])
    if not 0 <= index < len(mistakes):
        raise TrainingError(404, "That image isn't in this run's mistakes.")
    return mistakes[index]


def _mistake_path(entry: dict, dataset: str) -> Path:
    if entry.get("removed"):
        folder = removed_folder(dataset)
    else:
        folder = _category_folder(entry.get("movedTo") or entry["actual"], dataset)
    return folder / Path(entry["file"]).name


def mistake_image(job_id: str, index: int) -> Path:
    """Finds a mistake's image. It reads the user's own copy, so removing images really removes them."""
    job = _trained(job_id)
    path = _mistake_path(_mistake(job, index), job.dataset)
    if not path.is_file():
        raise TrainingError(404, "That image has been removed.")
    return path


def fix_mistake(job_id: str, index: int, action: str, category: str | None = None) -> Job:
    """Refiles, removes or restores a mistake's image. The change applies from the next training."""
    with _lock:
        job = _trained(job_id)
        entry = _mistake(job, index)
        source = _mistake_path(entry, job.dataset)
        if not source.is_file():
            raise TrainingError(404, "That image has been removed.")
        name = source.name
        if action == "move":
            if category not in job.categories:
                raise TrainingError(422, "Choose one of this model's categories.")
            target = _category_folder(category, job.dataset)
            moved_to, removed = (None if category == entry["actual"] else category), False
        elif action == "remove":
            target, moved_to, removed = removed_folder(job.dataset), None, True
        else:
            target, moved_to, removed = _category_folder(entry["actual"], job.dataset), None, False
        target.mkdir(parents=True, exist_ok=True)
        destination = target / name
        # The same photo can already be in the target category. Undo must not take that copy away.
        keep_source = entry.get("merged", False)
        already_there = destination.exists() and destination != source
        if destination != source:
            if keep_source and not already_there:
                shutil.copy2(source, destination)
            elif not keep_source and already_there:
                source.unlink()
            elif not keep_source:
                os.replace(source, destination)
            entry["merged"] = already_there
        entry["movedTo"], entry["removed"] = moved_to, removed
        _save(job)
        return job


def _trained(job_id: str) -> Job:
    job = _require(job_id)
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This run didn't save a model.")
    return job


def predict(job_id: str, data: bytes) -> list[dict]:
    extension = read_image(data)
    job = _trained(job_id)
    if job.task != "image":
        raise TrainingError(422, "This model doesn't read images.")
    folder = _folder(job)
    picture = folder / f"try-{uuid.uuid4().hex}{extension}"
    picture.write_bytes(data)
    try:
        with _predictions:
            done = subprocess.run(
                _python_command("predict.py", picture.name, "--json"), cwd=folder, env=_environment(folder) | {"OMP_NUM_THREADS": "2"},
                stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace", timeout=PREDICT_LIMIT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        raise TrainingError(504, "The model took too long to answer.") from None
    finally:
        picture.unlink(missing_ok=True)
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise TrainingError(500, lines[-1][:600] if lines else "The model could not read this image.")
    return json.loads(done.stdout.strip().splitlines()[-1])[0]["answers"]


# ------------------------------------------------------------------ live predictions

LIVE_IDLE_SECONDS = 120
LIVE_ANSWER_SECONDS = 20
LIVE_CHECK_SECONDS = 5
# Runs inside the model's own folder. It loads the model once, then answers one image per line of
# input, so a webcam can be asked several times a second instead of waiting for Python to start.


class _LivePredictor:
    def __init__(self, job: Job, script: str, unreadable: str):
        import queue

        self.job_id = job.id
        self.unreadable = unreadable
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.errors: collections.deque[str] = collections.deque(maxlen=20)
        folder = _folder(job)
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-u", "-c", script], cwd=folder, env=_environment(folder) | {"OMP_NUM_THREADS": "2"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1,
        )
        self.used = time.time()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()
        if self._next(PREDICT_LIMIT_SECONDS) != "ready":
            self.close()
            raise TrainingError(500, self._problem("The model couldn't be loaded for live answers."))

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def _read_errors(self) -> None:
        for line in self.process.stderr:
            if line.strip():
                self.errors.append(line.strip())

    def _problem(self, fallback: str) -> str:
        return self.errors[-1][:600] if self.errors else fallback

    def _next(self, timeout: float) -> str | None:
        import queue

        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def ask(self, data: bytes, explain_it: bool = False) -> dict:
        import base64

        self.used = time.time()
        try:
            self.process.stdin.write(json.dumps({"data": base64.b64encode(data).decode(), "explain": explain_it}) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise TrainingError(500, self._problem("The live model stopped. Try again.")) from None
        line = self._next(LIVE_ANSWER_SECONDS)
        if line is None:
            raise TrainingError(504, "The model took too long to answer.")
        reply = json.loads(line)
        if "error" in reply:
            raise TrainingError(422, self.unreadable)
        return reply

    def alive(self) -> bool:
        return self.process.poll() is None

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(10)


_live: _LivePredictor | None = None
_live_lock = threading.Lock()


def predict_live(job_id: str, data: bytes, explain_it: bool = False) -> dict:
    """Answers quickly for a webcam: one model stays loaded while it's being asked.

    Returns {"answers", "heatmap"}, where the heatmap of where it looked is None unless asked for.
    """
    read_image(data)
    job = stored(job_id)
    if job.task != "image":
        raise TrainingError(422, "This model doesn't read images.")
    return ask_live(job, explain.LIVE_RUNNER, data, explain_it, "The model couldn't read this image.")


def ask_live(job: Job, script: str, data: bytes, explain_it: bool, unreadable: str) -> dict:
    """Asks the one model kept loaded, loading this one in its place first if needed."""
    global _live
    if job.status == "running" or not job.hasModel:
        raise TrainingError(409, "This model has no saved file to answer with.")
    with _live_lock:
        if _live is not None and (_live.job_id != job.id or not _live.alive()):
            _live.close()
            _live = None
        if _live is None:
            _live = _LivePredictor(job, script, unreadable)
            threading.Thread(target=_close_when_idle, args=(_live,), daemon=True).start()
        return _live.ask(data, explain_it)


def _close_when_idle(predictor: _LivePredictor) -> None:
    global _live
    while predictor.alive():
        time.sleep(LIVE_CHECK_SECONDS)
        with _live_lock:
            if time.time() - predictor.used > LIVE_IDLE_SECONDS:
                predictor.close()
                if _live is predictor:
                    _live = None
                return


def close_live(job_id: str | None = None) -> None:
    """Stops the live model, for example before its folder is deleted."""
    global _live
    with _live_lock:
        if _live is not None and (job_id is None or _live.job_id == job_id):
            _live.close()
            _live = None


def bundle(job_id: str, target: Path, credits: str | None = None, card: str | None = None) -> Job:
    """Writes the project and its trained model, without the user's images, to a zip file."""
    import zipfile

    job = stored(job_id)
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This run didn't save a model.")
    folder = _folder(job)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in job.files:
            archive.write(folder / name, f"{job.slug}/{name}")
        model = _model_path(job)
        for path in sorted(model.rglob("*")) if model.is_dir() else [model]:
            if path.is_file():
                archive.write(path, f"{job.slug}/{path.relative_to(folder)}")
        if credits:
            archive.writestr(f"{job.slug}/DATA_SOURCES.md", credits)
        if card:
            archive.writestr(f"{job.slug}/MODEL_CARD.md", card)
    return job
