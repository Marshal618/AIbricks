"""Trains a text sorter from a saved dataset, using few examples.

It follows the SetFit method: fine-tune a small pretrained sentence model so that examples in
the same category sit close together, then train a simple classifier on top. The generated
train.py and predict.py do the work, so the downloaded project behaves the same way.
"""
from __future__ import annotations

import json
import subprocess

from . import codegen, datasets, labelling, training, versions
from .training import TrainingError

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_PER_CATEGORY = 4
SUGGESTED_PER_CATEGORY = 16
MISTAKES_RETURNED = 60

_MODEL_PY = '''"""The text sorter: a sentence model with a small classifier on top."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MAX_WORDS = 128


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class SentenceEncoder:
    def __init__(self, source, device):
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModel.from_pretrained(source, use_safetensors=True).to(device)
        self.device = device

    def __call__(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_WORDS, return_tensors="pt").to(self.device)
        words = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return F.normalize((words * mask).sum(1) / mask.sum(1).clamp(min=1e-9), dim=1)

    def embed(self, texts, batch_size=64):
        self.model.eval()
        with torch.no_grad():
            return torch.cat([self(texts[start:start + batch_size]) for start in range(0, len(texts), batch_size)]).cpu()


def load(folder="model", device="cpu"):
    folder = Path(folder)
    encoder = SentenceEncoder(folder / "encoder", device)
    head = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    return encoder, head, labels


def answers(encoder, head, labels, texts):
    """Returns each text's categories, most likely first."""
    chances = F.softmax(encoder.embed(texts) @ head["weight"].T + head["bias"], dim=1)
    return [
        sorted(({"category": name, "chance": float(chance)} for name, chance in zip(labels, row)), key=lambda answer: -answer["chance"])
        for row in chances
    ]
'''

_TRAIN_PY = '''"""Trains the text sorter. Put your labelled text in data/texts.csv (columns: text,label), then run: python train.py"""
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model import SentenceEncoder, answers, best_device

BASE_MODEL = "__BASE_MODEL__"
DATA_FILE = "data/texts.csv"
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MIN_STEPS, MAX_STEPS = 40, 400
PAIRS_PER_EXAMPLE = 40
HOLD_BACK = 0.2
MAX_PER_CATEGORY = 2000  # The classifier reaches its limit well before this on most tasks.


def read_data():
    csv.field_size_limit(10_000_000)
    grouped = {}
    with open(DATA_FILE, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            grouped.setdefault(row["label"], []).append(row["text"])
    return {name: grouped[name] for name in sorted(grouped)}


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split(grouped):
    """Holds back part of each category. Texts are picked by a fingerprint of their words rather than
    at random, so models trained on the same texts are checked on the same ones and compare fairly."""
    train, check = {}, []
    for name, texts in grouped.items():
        texts = sorted(texts, key=fingerprint)
        held = max(1, int(len(texts) * HOLD_BACK))
        check += [(text, name) for text in texts[:held]]
        train[name] = texts[held:held + MAX_PER_CATEGORY]
    return train, check


def fine_tune(encoder, train, names, device):
    """Moves examples from the same category together, and different ones apart."""
    rng = random.Random(SEED)
    total = sum(len(texts) for texts in train.values())
    steps = min(MAX_STEPS, max(MIN_STEPS, total * PAIRS_PER_EXAMPLE // BATCH_SIZE))
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=LEARNING_RATE)
    encoder.model.train()
    last = 0.0
    for step in range(1, steps + 1):
        first, second, same = [], [], []
        for index in range(BATCH_SIZE):
            name = rng.choice(names)
            other = name if index % 2 == 0 else rng.choice([item for item in names if item != name])
            first.append(rng.choice(train[name]))
            second.append(rng.choice(train[other]))
            same.append(float(name == other))
        loss = F.mse_loss((encoder(first) * encoder(second)).sum(1), torch.tensor(same, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = loss.item()
        print(f"Step {step}/{steps}", flush=True)
    return last


def fit_classifier(encoder, train, names):
    texts = [text for name in names for text in train[name]]
    targets = torch.tensor([index for index, name in enumerate(names) for _ in train[name]])
    embeddings = encoder.embed(texts)
    layer = torch.nn.Linear(embeddings.shape[1], len(names))
    optimizer = torch.optim.LBFGS(layer.parameters(), max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(layer(embeddings), targets) + 1e-3 * layer.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"weight": layer.weight.detach().clone(), "bias": layer.bias.detach().clone()}


def save(encoder, head, names):
    """Saves in one step, so stopping early can't leave a half-written model."""
    partial = Path("model.partial")
    shutil.rmtree(partial, ignore_errors=True)
    encoder.model.save_pretrained(partial / "encoder")
    encoder.tokenizer.save_pretrained(partial / "encoder")
    torch.save(head, partial / "head.pt")
    (partial / "labels.json").write_text(json.dumps(names), encoding="utf-8")
    shutil.rmtree("model", ignore_errors=True)
    os.replace(partial, "model")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    grouped = read_data()
    names = list(grouped)
    if len(names) < 2:
        sys.exit("Add examples to at least 2 categories.")
    train, check = split(grouped)
    device = best_device()
    try:
        encoder = SentenceEncoder(BASE_MODEL, device)
    except (OSError, ValueError) as problem:
        sys.exit(f"The base model {BASE_MODEL} couldn't be loaded. The first run needs an internet connection. ({type(problem).__name__})")
    print(f"Training on {sum(len(texts) for texts in train.values())} samples, checking on {len(check)}, using {device}.", flush=True)
    train_loss = fine_tune(encoder, train, names, device)
    head = fit_classifier(encoder, train, names)
    results = answers(encoder, head, names, [text for text, _ in check])
    right = sum(result[0]["category"] == actual for result, (_, actual) in zip(results, check))
    check_loss = sum(-torch.log(torch.tensor(max(next(a["chance"] for a in result if a["category"] == actual), 1e-9))).item() for result, (_, actual) in zip(results, check)) / len(check)
    save(encoder, head, names)
    with open("data/review.json", "w", encoding="utf-8") as file:
        json.dump([{"text": text, "actual": actual, "guess": result[0]["category"], "chance": result[0]["chance"]} for result, (text, actual) in zip(results, check)], file)
    print(f"Epoch 1/1  training loss {train_loss:.4f}  check loss {check_loss:.4f}  check accuracy {100 * right / len(check):.1f}%", flush=True)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''"""Asks the trained text sorter about some text. Train first, then run: python predict.py "your text"

With no text given, it reads the text from standard input.
"""
import json
import sys

from model import answers, load


def main():
    arguments = [item for item in sys.argv[1:] if item != "--json"]
    text = arguments[0] if arguments else sys.stdin.read()
    encoder, head, labels = load()
    result = answers(encoder, head, labels, [text])[0]
    if "--json" in sys.argv:
        print(json.dumps(result))
    else:
        for answer in result:
            print(f"{answer['category']}: {answer['chance']:.0%}")


if __name__ == "__main__":
    main()
'''

_README = """# {name}

A text sorter trained in AI Bricks. It sorts text into: {categories}.

It follows the SetFit method, fine-tuning the small `{base}` sentence model and training a
classifier on top. Your training text isn't included in this download.

## Use it

```
pip install -r requirements.txt
python predict.py "text to sort"
```

## Train it again

Put your labelled text in `data/texts.csv`, with the columns `text` and `label`, then run
`python train.py`. The first run downloads the base model, which is about 90 MB.

The model folder includes a sentence model derived from `{base}`, released by its authors
under the Apache 2.0 licence. Check the licence of your own data before sharing this model.
"""

_REQUIREMENTS = "torch>=2.2\ntransformers>=4.40\n"


def project_files(name: str, categories: list[str]) -> dict[str, str]:
    return {
        "model.py": _MODEL_PY,
        "train.py": _TRAIN_PY.replace("__BASE_MODEL__", BASE_MODEL),
        "predict.py": _PREDICT_PY,
        "README.md": _README.format(name=name, categories=", ".join(categories), base=BASE_MODEL),
        "requirements.txt": _REQUIREMENTS,
    }


def _problems(amounts: dict[str, int]) -> list[str]:
    if len(amounts) < 2:
        return ["Add at least 2 categories."]
    short = [name for name, amount in amounts.items() if amount < MIN_PER_CATEGORY]
    return [f"Label at least {MIN_PER_CATEGORY} examples in: {', '.join(short)}."] if short else []


def start(dataset: str) -> training.Job:
    """Starts training on a text dataset's labelled examples."""
    with training._lock:
        info = datasets.get(dataset)
        if info["kind"] != "text":
            raise TrainingError(422, "This trainer is for text datasets.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        problems = _problems(info["counts"])
        if problems:
            raise TrainingError(422, " ".join(problems))
        files = project_files(info["name"], info["categories"])
        return training.start_run(info["name"], codegen.slug(info["name"]), codegen.build_hash(files), files, info["counts"], dataset, versions.freeze, "text")


def _text_job(job_id: str) -> training.Job:
    job = training.stored(job_id)
    if job.task != "text":
        raise TrainingError(404, "That model doesn't sort text.")
    return job


def try_text(job_id: str, text: str) -> list[dict]:
    job = _text_job(job_id)
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This model's file is no longer saved.")
    folder = training._folder(job)
    try:
        with training._predictions:
            done = subprocess.run(
                training._python_command("predict.py", "--json"), cwd=folder, env=training._environment(folder) | {"OMP_NUM_THREADS": "2"},
                input=text, capture_output=True, text=True, errors="replace", timeout=training.PREDICT_LIMIT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        raise TrainingError(504, "The model took too long to answer.") from None
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise TrainingError(500, lines[-1][:600] if lines else "The model could not read this text.")
    return json.loads(done.stdout.strip().splitlines()[-1])


def mistakes(job_id: str) -> dict:
    """The held-back texts the model got wrong, read from the dataset as it is now.

    Runs store only an ID for each text, so deleting a text from the dataset really removes it.
    """
    job = _text_job(job_id)
    datasets._read(job.dataset)
    found = (job.review or {}).get("mistakes", [])[:MISTAKES_RETURNED]
    known = labelling.texts_by_id([row["file"] for row in found], job.dataset)
    rows = [
        {"id": row["file"], "text": known[row["file"]]["text"], "label": known[row["file"]]["label"],
         "actual": row["actual"], "guess": row["guess"], "chance": row["chance"]}
        for row in found if row["file"] in known
    ]
    return {"mistakes": rows, "total": (job.review or {}).get("mistakeCount", 0)}
