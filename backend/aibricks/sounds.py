"""Sound datasets, and a sound sorter trained from a few recordings of each sound.

Recordings are cut into 1-second clips and stored like photos, one WAV file per clip in each
category's folder, so versions, training copies and fixing mistakes work the same way.

The sorter turns each clip into a spectrogram, a picture of which pitches are loud at each moment,
and reads it with a pretrained Audio Spectrogram Transformer (Gong et al., 2021) that stays frozen.
A small classifier learns your categories from what the transformer reports. The generated
train.py and predict.py do the work, so the downloaded project behaves the same way.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import os
import re
import uuid
import wave
from pathlib import Path

from . import codegen, datasets, labelling, training, versions
from .training import TrainingError

BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE
SHORTEST_SAMPLES = SAMPLE_RATE // 4  # Less than a quarter of a second is too little to learn from.
MAX_SOUND_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_SECONDS = 10 * 60
MAX_CLIPS = 20_000
MIN_PER_CATEGORY = 4
SUGGESTED_PER_CATEGORY = 20
CLIPS_LISTED = 60

_SOUND_ID = re.compile(r"^[0-9a-f]{32}\.wav$")


# ------------------------------------------------------------- reading recordings
# These two functions are also copied into the generated model.py, so the app and the download
# read sound the same way.

def read_wav(source, rate=16000, longest=None):
    """Reads a PCM WAV file as mono samples between -1 and 1, at `rate` samples a second."""
    import wave
    from math import gcd

    import numpy as np

    with wave.open(source, "rb") as file:
        channels, width, original, frames = file.getnchannels(), file.getsampwidth(), file.getframerate(), file.getnframes()
        if channels < 1 or original < 1:
            raise ValueError("the file has no audio")
        if longest is not None and frames > longest * original:
            raise ValueError(f"it is longer than {longest // 60} minutes")
        raw = file.readframes(frames)
    if width == 1:
        samples = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        samples = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 3:
        parts = np.frombuffer(raw[: len(raw) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
        joined = parts[:, 0] | (parts[:, 1] << 8) | (parts[:, 2] << 16)
        samples = np.where(joined >= 1 << 23, joined - (1 << 24), joined).astype(np.float32) / (1 << 23)
    elif width == 4:
        samples = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
    else:
        raise ValueError(f"{width * 8}-bit audio isn't supported")
    samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
    if original != rate:
        from scipy.signal import resample_poly

        divisor = gcd(rate, original)
        samples = resample_poly(samples, rate // divisor, original // divisor)
    return np.clip(samples, -1, 1).astype(np.float32)


def clips_of(samples, length=16000, shortest=4000):
    """Cuts a recording into 1-second clips. A short last piece is padded with silence, or dropped if
    it is under a quarter of a second."""
    import numpy as np

    clips = []
    for start in range(0, len(samples), length):
        piece = samples[start:start + length]
        if len(piece) < shortest:
            break
        clips.append(np.pad(piece, (0, length - len(piece))).astype(np.float32))
    return clips


def decode(data: bytes):
    """A recording's samples, or a TrainingError a person can act on."""
    if len(data) > MAX_SOUND_BYTES:
        raise TrainingError(413, "This recording is larger than 50 MB.")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise TrainingError(422, "This file isn't a WAV recording. Record in the app, or convert the file to WAV first.")
    try:
        return read_wav(io.BytesIO(data), SAMPLE_RATE, MAX_UPLOAD_SECONDS)
    except (wave.Error, EOFError) as problem:
        raise TrainingError(422, f"This WAV file couldn't be read ({problem}). Save it as 16-bit PCM WAV.") from None
    except ValueError as problem:
        raise TrainingError(422, f"This recording couldn't be used: {problem}.") from None


def encode(samples) -> bytes:
    """One clip as a 16-bit, 16 kHz mono WAV file."""
    import numpy as np

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------- storage

def files_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [path for path in folder.iterdir() if _SOUND_ID.match(path.name)]


def counts(names: list[str], dataset: str) -> dict[str, int]:
    return {name: len(files_in(training._category_folder(name, dataset))) for name in names}


def _total(dataset: str) -> int:
    folder = training.root(dataset) / "datasets"
    return sum(len(files_in(item)) for item in folder.iterdir()) if folder.is_dir() else 0


def add_recording(category: str, data: bytes, dataset: str) -> dict:
    """Cuts a recording into 1-second clips and files them all under one category."""
    labelling._require_kind(dataset, "sound")
    samples = decode(data)
    clips = clips_of(samples, CLIP_SAMPLES, SHORTEST_SAMPLES)
    if not clips:
        raise TrainingError(422, "This recording is shorter than a quarter of a second.")
    with training._lock:
        labelling._require_category(dataset, category)
        if datasets._busy(dataset) or training.is_running() and training.current().dataset == dataset:
            raise TrainingError(409, "Wait for training or the label check on this dataset to finish.")
        if _total(dataset) + len(clips) > MAX_CLIPS:
            raise TrainingError(413, f"A sound dataset can hold {MAX_CLIPS:,} clips. Remove some first.")
        folder = training._category_folder(category, dataset)
        folder.mkdir(parents=True, exist_ok=True)
        added = duplicates = 0
        for clip in clips:
            content = encode(clip)
            target = folder / (hashlib.sha256(content).hexdigest()[:32] + ".wav")
            if target.exists():
                duplicates += 1
                continue
            temporary = folder / f"{uuid.uuid4().hex}.partial"
            temporary.write_bytes(content)
            os.replace(temporary, target)
            added += 1
    return {"added": added, "duplicates": duplicates, "count": len(files_in(folder))}


def _clip_path(dataset: str, category: str, clip_id: str) -> Path:
    meta = datasets._read(dataset)
    if meta["kind"] != "sound" or category not in meta["categories"] or not _SOUND_ID.match(clip_id):
        raise TrainingError(404, "That clip doesn't exist.")
    return training._category_folder(category, dataset) / clip_id


def clip_file(dataset: str, category: str, clip_id: str) -> Path:
    path = _clip_path(dataset, category, clip_id)
    if not path.is_file():
        raise TrainingError(404, "That clip has been removed.")
    return path


def clips(dataset: str, category: str) -> dict:
    """The newest clips in a category, to listen to and remove bad ones."""
    meta = datasets._read(dataset)
    if meta["kind"] != "sound" or category not in meta["categories"]:
        raise TrainingError(404, "That category doesn't exist.")
    found = sorted(files_in(training._category_folder(category, dataset)), key=lambda path: path.stat().st_mtime, reverse=True)
    return {"clips": [path.name for path in found[:CLIPS_LISTED]], "total": len(found)}


def remove_clip(dataset: str, category: str, clip_id: str) -> None:
    """Removes a clip from the dataset. Saved versions keep their own copy until they're deleted."""
    with training._lock:
        if training.is_running() and training.current().dataset == dataset:
            raise TrainingError(409, "Wait for training on this dataset to finish.")
        _clip_path(dataset, category, clip_id).unlink(missing_ok=True)


# --------------------------------------------------------------- the project

_MODEL_PY = '''"""The sound sorter: a pretrained Audio Spectrogram Transformer turns each 1-second clip into 768
numbers, and a small classifier learned from your recordings sorts those numbers.

The transformer learned from AudioSet, about two million labelled sound clips, and stays frozen. It
was made for 10-second spectrograms. For 1-second clips its table of positions is cut to the first
1.28 seconds, as the AST paper does for shorter inputs.
"""
import json
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

BASE_MODEL = "__BASE_MODEL__"
SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE
FRAMES = 128  # 10 ms each, so 1.28 seconds: a 1-second clip and some silence
MAX_CLIPS = 30  # A longer recording is judged on its first 30 seconds.

warnings.filterwarnings("ignore", message="At least one mel filter")


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


__READ_WAV__

__CLIPS_OF__

def _shorten(model, frames):
    config = model.config
    across = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
    before = (config.max_length - config.patch_size) // config.time_stride + 1
    after = (frames - config.patch_size) // config.time_stride + 1
    table = model.embeddings.position_embeddings.data
    grid = table[:, 2:].reshape(1, across, before, -1)[:, :, :after].reshape(1, across * after, -1)
    model.embeddings.position_embeddings = torch.nn.Parameter(torch.cat([table[:, :2], grid], dim=1))
    model.config.max_length = frames


class SoundEncoder:
    def __init__(self, device="cpu"):
        from transformers import ASTFeatureExtractor, ASTModel
        from transformers.utils import logging

        logging.set_verbosity_error()
        try:
            model = ASTModel.from_pretrained(BASE_MODEL, local_files_only=True)
        except OSError:
            model = ASTModel.from_pretrained(BASE_MODEL)
        _shorten(model, FRAMES)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model.to(device).eval()
        self.features = ASTFeatureExtractor(max_length=FRAMES)
        self.device = device

    def spectrograms(self, clips):
        """Each clip as the picture the transformer reads: 128 moments, each with 128 pitch bands."""
        return self.features(list(clips), sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"]

    def __call__(self, spectrograms):
        return F.normalize(self.model(input_values=spectrograms.to(self.device)).pooler_output, dim=1).cpu()

    def embed(self, clips, batch_size=32, progress=None):
        rows = []
        with torch.no_grad():
            for start in range(0, len(clips), batch_size):
                rows.append(self(self.spectrograms(clips[start:start + batch_size])))
                if progress:
                    progress(min(start + batch_size, len(clips)))
        return torch.cat(rows)


def load(folder="model", device="cpu"):
    folder = Path(folder)
    head = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    return SoundEncoder(device), head, labels


def ranked(labels, chances):
    return sorted(({"category": name, "chance": float(chance)} for name, chance in zip(labels, chances)), key=lambda answer: -answer["chance"])


def answers(encoder, head, labels, clips):
    """The categories for a recording, most likely first, averaged over its 1-second clips."""
    embedded = encoder.embed(clips[:MAX_CLIPS])
    return ranked(labels, F.softmax(embedded @ head["weight"].T + head["bias"], dim=1).mean(dim=0).tolist())


def _grid(values, height, width):
    pooled = F.adaptive_avg_pool2d(values[None, None], (height, width))[0, 0]
    low, high = float(pooled.min()), float(pooled.max())
    scaled = (pooled - low) / (high - low) if high > low else torch.zeros_like(pooled)
    return {"width": width, "height": height, "values": [round(value, 2) for value in scaled.flatten().tolist()]}


def picture(encoder, head, labels, clip, explain=False):
    """One clip's answers and spectrogram, high pitches at the top. With explain, also where in it
    the answer came from: how strongly the top answer reacts to each point of the spectrogram."""
    spectrogram = encoder.spectrograms([clip])
    spectrogram.requires_grad_(explain)
    with torch.set_grad_enabled(explain):
        logits = (encoder(spectrogram) @ head["weight"].T + head["bias"])[0]
    moments = min(FRAMES, 1 + (len(clip) - 400) // 160)
    shown = spectrogram.detach()[0, :moments].T.flip(0)
    result = {"answers": ranked(labels, F.softmax(logits.detach(), dim=0).tolist()), "spectrogram": _grid(shown, 64, moments), "heatmap": None}
    if explain:
        top = int(logits.argmax())
        (gradient,) = torch.autograd.grad(logits[top], spectrogram)
        heat = _grid(gradient.abs()[0, :moments].T.flip(0), 16, 20)
        result["heatmap"] = heat | {"method": "gradients", "category": labels[top]}
    return result
'''.replace("__READ_WAV__", inspect.getsource(read_wav)).replace("__CLIPS_OF__", inspect.getsource(clips_of)).replace("__BASE_MODEL__", BASE_MODEL)

_TRAIN_PY = '''"""Trains the sound sorter. Put 1-second WAV clips in data/sounds/<category>/, then run: python train.py

Each clip, and two altered copies of it, are turned into numbers by the frozen transformer, then a
small classifier learns to tell the categories apart from those numbers.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import BASE_MODEL, CLIP_SAMPLES, SoundEncoder, best_device, read_wav

DATA = Path("data/sounds")
SEED = 42
HOLD_BACK = 0.2
VARIANTS = 2  # altered copies of each training clip: shifted in time and made louder or quieter
MAX_PER_CATEGORY = 2000


def read_data():
    grouped = {}
    for folder in sorted(DATA.iterdir()):
        files = sorted(folder.glob("*.wav")) if folder.is_dir() else []
        if files:
            grouped[folder.name] = files
    return grouped


def fingerprint(path):
    return hashlib.sha256(path.name.encode("utf-8")).hexdigest()


def split(grouped):
    """Holds back part of each category. Clips are picked by a fingerprint rather than at random, so
    models trained on the same clips are checked on the same ones and compare fairly."""
    train, check = {}, []
    for name, files in grouped.items():
        files = sorted(files, key=fingerprint)
        held = max(1, int(len(files) * HOLD_BACK))
        check += [(path, name) for path in files[:held]]
        train[name] = files[held:held + MAX_PER_CATEGORY]
    return train, check


def load_clip(path):
    samples = read_wav(str(path))[:CLIP_SAMPLES]
    return np.pad(samples, (0, CLIP_SAMPLES - len(samples))).astype(np.float32)


def variant(clip, rng):
    shifted = np.roll(clip, int(rng.integers(-CLIP_SAMPLES // 5, CLIP_SAMPLES // 5)))
    return np.clip(shifted * rng.uniform(0.6, 1.4), -1, 1).astype(np.float32)


def fit_classifier(embeddings, targets, count):
    layer = torch.nn.Linear(embeddings.shape[1], count)
    optimizer = torch.optim.LBFGS(layer.parameters(), max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(layer(embeddings), targets) + 1e-3 * layer.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        loss = F.cross_entropy(layer(embeddings), targets).item()
    return {"weight": layer.weight.detach().clone(), "bias": layer.bias.detach().clone()}, loss


def save(head, names):
    """Saves in one step, so stopping early can't leave a half-written model."""
    partial = Path("model.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    torch.save(head, partial / "head.pt")
    (partial / "labels.json").write_text(json.dumps(names), encoding="utf-8")
    (partial / "base.json").write_text(json.dumps({"base": BASE_MODEL}), encoding="utf-8")
    shutil.rmtree("model", ignore_errors=True)
    os.replace(partial, "model")


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    grouped = read_data()
    names = list(grouped)
    if len(names) < 2:
        sys.exit("Add clips to at least 2 categories.")
    train, check = split(grouped)
    device = best_device()
    try:
        encoder = SoundEncoder(device)
    except (OSError, ValueError) as problem:
        sys.exit(f"The base model {BASE_MODEL} couldn't be loaded. The first run downloads it, about 350 MB, so it needs an internet connection. ({type(problem).__name__})")
    clips, targets = [], []
    for index, name in enumerate(names):
        for path in train[name]:
            clip = load_clip(path)
            clips += [clip] + [variant(clip, rng) for _ in range(VARIANTS)]
            targets += [index] * (1 + VARIANTS)
    held = [load_clip(path) for path, _ in check]
    print(f"Training on {sum(len(files) for files in train.values())} samples, checking on {len(check)}, using {device}.", flush=True)
    total = len(clips) + len(held)
    embedded = encoder.embed(clips, progress=lambda done: print(f"Step {done}/{total}", flush=True))
    head, train_loss = fit_classifier(embedded, torch.tensor(targets), len(names))
    checked = encoder.embed(held, progress=lambda done: print(f"Step {len(clips) + done}/{total}", flush=True))
    chances = F.softmax(checked @ head["weight"].T + head["bias"], dim=1)
    actual = torch.tensor([names.index(name) for _, name in check])
    right = int((chances.argmax(dim=1) == actual).sum())
    check_loss = F.nll_loss(chances.clamp(min=1e-9).log(), actual).item()
    save(head, names)
    with open("data/review.json", "w", encoding="utf-8") as file:
        json.dump([
            {"file": path.name, "actual": name, "guess": names[int(row.argmax())], "chance": float(row.max())}
            for (path, name), row in zip(check, chances)
        ], file)
    print(f"Epoch 1/1  training loss {train_loss:.4f}  check loss {check_loss:.4f}  check accuracy {100 * right / len(check):.1f}%", flush=True)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''"""Asks the trained sound sorter about a WAV recording. Train first, then run: python predict.py sound.wav

A recording longer than a second is cut into 1-second clips, and their answers are averaged.
"""
import json
import sys

from model import answers, clips_of, load, read_wav


def main():
    paths = [item for item in sys.argv[1:] if item != "--json"]
    if not paths:
        sys.exit("Name a WAV file, for example: python predict.py sound.wav")
    encoder, head, labels = load()
    results = []
    for path in paths:
        clips = clips_of(read_wav(path))
        if not clips:
            sys.exit(f"{path} is shorter than a quarter of a second.")
        results.append({"file": path, "answers": answers(encoder, head, labels, clips)})
    if "--json" in sys.argv:
        print(json.dumps(results))
        return
    for result in results:
        print(result["file"])
        for answer in result["answers"]:
            print(f"  {answer['category']}: {answer['chance']:.0%}")


if __name__ == "__main__":
    main()
'''

_README = """# {name}

A sound sorter trained in AI Bricks. It sorts 1-second sounds into: {categories}.

Each clip becomes a spectrogram, a picture of which pitches are loud at each moment. The pretrained
Audio Spectrogram Transformer `{base}` reads that picture and stays frozen, and a small classifier
trained on your recordings sorts what it reports. Your recordings aren't included in this download.

## Use it

```
pip install -r requirements.txt
python predict.py sound.wav
```

The first run downloads the transformer, about 350 MB, from Hugging Face. Recordings must be WAV
files. Longer ones are cut into 1-second clips and the answers averaged.

## Train it again

Put 1-second WAV clips in `data/sounds/<category>/`, one folder per category, then run
`python train.py`.
"""

_REQUIREMENTS = "torch>=2.2\ntransformers>=4.40\nnumpy\nscipy\n"

# Runs in the model's folder and keeps it loaded, answering one recording per line of input.
LIVE_RUNNER = """
import base64, contextlib, io, json, sys
sys.path.insert(0, '.')
with contextlib.redirect_stdout(sys.stderr):
    from model import answers, clips_of, load, picture, read_wav
    encoder, head, labels = load()
print('ready', flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        clips = clips_of(read_wav(io.BytesIO(base64.b64decode(request['data']))))
        if not clips:
            raise ValueError('shorter than a quarter of a second')
        reply = picture(encoder, head, labels, clips[0], bool(request.get('explain')))
        if len(clips) > 1:
            reply['answers'] = answers(encoder, head, labels, clips)
        reply['clips'] = len(clips)
        print(json.dumps(reply), flush=True)
    except Exception as problem:
        print(json.dumps({'error': f'{type(problem).__name__}: {problem}'[:300]}), flush=True)
"""


def project_files(name: str, categories: list[str]) -> dict[str, str]:
    return {
        "model.py": _MODEL_PY,
        "train.py": _TRAIN_PY,
        "predict.py": _PREDICT_PY,
        "README.md": _README.format(name=name, categories=", ".join(categories), base=BASE_MODEL),
        "requirements.txt": _REQUIREMENTS,
    }


# ---------------------------------------------------------------------- training

def _problems(amounts: dict[str, int]) -> list[str]:
    if len(amounts) < 2:
        return ["Add at least 2 categories."]
    short = [name for name, amount in amounts.items() if amount < MIN_PER_CATEGORY]
    return [f"Record at least {MIN_PER_CATEGORY} clips in: {', '.join(short)}."] if short else []


def start(dataset: str) -> training.Job:
    """Starts training on a sound dataset's clips."""
    with training._lock:
        info = datasets.get(dataset)
        if info["kind"] != "sound":
            raise TrainingError(422, "This trainer is for sound datasets.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        problems = _problems(info["counts"])
        if problems:
            raise TrainingError(422, " ".join(problems))
        files = project_files(info["name"], info["categories"])
        return training.start_run(info["name"], codegen.slug(info["name"]), codegen.build_hash(files), files, info["counts"], dataset, versions.freeze, "sound")


def listen(job_id: str, data: bytes, explain_it: bool = False) -> dict:
    """A trained model's answers for a recording, with its first second's spectrogram, kept loaded
    between requests so a microphone can be asked about every second as it comes."""
    decode(data)
    job = training.stored(job_id)
    if job.task != "sound":
        raise TrainingError(422, "This model doesn't sort sounds.")
    return training.ask_live(job, LIVE_RUNNER, data, explain_it, "The model couldn't read this recording.")


MISTAKES_RETURNED = 60


def mistakes(job_id: str) -> dict:
    """The held-back clips a sound model got wrong, most confident first. Each plays from the
    dataset as it is now, so removing a clip really removes it."""
    job = training.stored(job_id)
    if job.task != "sound":
        raise TrainingError(404, "That model doesn't sort sounds.")
    review = job.review or {}
    rows = [entry | {"index": index} for index, entry in enumerate(review.get("mistakes", [])[:MISTAKES_RETURNED])]
    return {"mistakes": rows, "total": review.get("mistakeCount", 0)}
