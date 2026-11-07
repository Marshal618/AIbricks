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


def _folder(job: Job) -> Path:
    return workspace() / "runs" / job.id


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
        job.hasModel = job.best is not None and (folder / "model.pt").exists()
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


def public(job: Job | None) -> dict | None:
    if job is None:
        return None
    data = asdict(job)
    data["elapsedSeconds"] = (job.finishedAt or time.time()) - job.startedAt
    del data["files"], data["slug"], data["stopRequested"]
    return data


def saved_models(dataset: str) -> list[dict]:
    """The runs trained from a dataset, newest first, without their per-epoch detail."""
    with _lock:
        current()
        rows = []
        for job in _saved_jobs():
            if job.dataset != dataset:
                continue
            row = public(job)
            del row["epochs"]
            if row["review"]:
                row["review"] = {key: row["review"][key] for key in ("total", "correct", "categories")}
            model = _folder(job) / "model.pt"
            row["modelBytes"] = model.stat().st_size if job.hasModel and model.exists() else 0
            rows.append(row)
        return rows


def delete_model(job_id: str) -> None:
    global _job
    with _lock:
        job = stored(job_id)
        if job.status == "running":
            raise TrainingError(409, "Stop this model's training first.")
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
        (_folder(job) / "model.pt").unlink(missing_ok=True)
        job.hasModel, job.modelRemoved = False, True
        _save(job)


def _environment(folder: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(folder),
        "OMP_NUM_THREADS": str(max((os.cpu_count() or 2) - 1, 1)),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TORCH_HOME": str(workspace() / "cache" / "torch"),
        "HF_HOME": str(workspace() / "cache" / "huggingface"),
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
        job.hasModel = job.best is not None and (_folder(job) / "model.pt").exists()
        job.stage = "reviewing"
        _save(job)
    if job.hasModel and not _shutting_down:
        try:
            _review(job)
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


def start(project_name: str, slug: str, build_hash: str, files: dict[str, str], names: list[str], epochs: int, dataset: str = DEFAULT_DATASET, freeze=None) -> Job:
    """Starts training in the background. Earlier models are kept, up to MODELS_KEPT.

    freeze(dataset, names, folder) copies the data for this run and returns its version number.
    """
    global _job, _process
    with _lock:
        if current() is not None and _job.status == "running":
            raise TrainingError(409, "A model is already training. Stop it first.")
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
                job.datasetVersion = freeze(dataset, names, images)
            except Exception:
                shutil.rmtree(folder, ignore_errors=True)
                raise
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
    job = current()
    if job is None or job.id != job_id:
        raise TrainingError(404, "That training run no longer exists. Train the model again.")
    return job


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


def bundle(job_id: str, target: Path, credits: str | None = None) -> Job:
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
        archive.write(folder / "model.pt", f"{job.slug}/model.pt")
        if credits:
            archive.writestr(f"{job.slug}/DATA_SOURCES.md", credits)
    return job
