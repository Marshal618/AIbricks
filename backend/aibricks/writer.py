"""Adjusts a small open language model to write like a dataset's text, then writes and chats with it.

The model downloads once from Hugging Face and runs on this computer. Only its last few layers
change while training, so it keeps what it already knows, and the saved file holds just those.
"""
from __future__ import annotations

import json
import re
from contextlib import closing

from . import codegen, datasets, labelling, training, versions
from .catalogue import WRITERS
from .training import TrainingError

MIN_WORDS = 1500
SUGGESTED_WORDS = 20_000
STRENGTHS = {"light": 2, "more": 6}
LENGTHS = {"quick": 1, "balanced": 2, "thorough": 4}
MAX_PROMPT = 2000
MAX_TURNS = 12
DEFAULT_MODEL = "SmolLM2 135M"
WORDS = re.compile(r"\S+")
ANSWER_SECONDS = 180


def is_writing(meta: dict) -> bool:
    return meta.get("kind") == "text" and meta.get("goal") == "write"


def _require_writing(dataset: str) -> dict:
    meta = datasets._read(dataset)
    if not is_writing(meta):
        raise TrainingError(422, "This trainer is for “write like this text” datasets.")
    return meta


def _downloaded(repo: str) -> bool:
    folder = training._cache() / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}"
    return any(folder.glob("snapshots/*/*.safetensors"))


def models() -> list[dict]:
    return [
        {"name": name, "download": spec["download"], "downloaded": _downloaded(spec["repo"]), "layers": spec["layers"]}
        for name, spec in WRITERS.items()
    ]


def amounts(dataset: str) -> tuple[int, int]:
    """How many pieces of text the dataset holds, and how many words."""
    with closing(labelling._database(dataset)) as database:
        rows = database.execute("SELECT text FROM items WHERE removed = 0").fetchall()
    return len(rows), sum(len(WORDS.findall(text)) for (text,) in rows)


def analyse(dataset: str) -> dict:
    _require_writing(dataset)
    pieces, words = amounts(dataset)
    problems, notes = [], []
    if words < MIN_WORDS:
        problems.append(f"Add at least {MIN_WORDS:,} words of text. It has {words:,}.")
    elif words < SUGGESTED_WORDS:
        notes.append(f"It can train now. With {SUGGESTED_WORDS:,} words or more, it picks up your style more reliably.")
    return {"pieces": pieces, "words": words, "problems": problems, "notes": notes, "models": models(),
            "minWords": MIN_WORDS, "suggestedWords": SUGGESTED_WORDS}


# ----------------------------------------------------------------------- the project it trains

_MODEL_PY = '''"""The writer: a small open language model, and the few layers training adjusts."""
import json
import os

import torch

FOLDER = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(FOLDER, "config.json"), encoding="utf-8") as file:
    CONFIG = json.load(file)
ADJUSTED_FILE = os.path.join(FOLDER, "model", "adjusted.pt")


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CONFIG["repo"])


def load_writer(adjusted=True, device="cpu"):
    """The downloaded model, with your adjusted layers put in when there are some."""
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(CONFIG["repo"], dtype=torch.float32)
    except OSError as error:
        raise RuntimeError(
            f"Could not download {CONFIG['model']} from Hugging Face. Check your internet connection and try again. "
            "After the first download it works offline."
        ) from error
    if adjusted and os.path.exists(ADJUSTED_FILE):
        model.load_state_dict(torch.load(ADJUSTED_FILE, weights_only=True), strict=False)
    return model.to(device).eval()


def adjustable(model):
    """Fixes every weight except the last few layers', and returns those."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    body = model.base_model
    for part in [*body.layers[-CONFIG["adjust_layers"]:], body.norm]:
        for parameter in part.parameters():
            parameter.requires_grad = True
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def write(model, tokenizer, prompt, chat=False, history=(), new_tokens=160, seed=0):
    """Continues the prompt, or with chat=True answers it. The same seed gives the same choices."""
    if chat:
        messages = [*history, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        numbers = tokenizer.encode(text, add_special_tokens=False)
    else:
        numbers = tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.bos_token_id or tokenizer.eos_token_id]
    device = next(model.parameters()).device
    start = torch.tensor([numbers], device=device)
    torch.manual_seed(seed)
    with torch.no_grad():
        result = model.generate(
            start, attention_mask=torch.ones_like(start), max_new_tokens=new_tokens, do_sample=True,
            temperature=0.7, top_k=40, repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    words = tokenizer.decode(result[0, len(numbers):], skip_special_tokens=True)
    return words.strip() if chat else prompt + words
'''

_TRAIN_PY = '''"""Adjusts the writer to your text. Put your text in data/corpus.txt, then run: python train.py

Each line of the file is one piece of text. About 1 in 10 pieces is held back to check the writer on.
"""
import copy
import hashlib
import json
import math
import os
import random

import torch
from torch.nn import functional

from model import ADJUSTED_FILE, CONFIG, adjustable, best_device, load_tokenizer, load_writer, write

DATA_FILE = "data/corpus.txt"
WINDOW = CONFIG["window"]  # tokens it reads at a time while training
BATCH = 8
LEARNING_RATE = 1e-4  # small, so it keeps what it already knows
MAX_CHECK_WINDOWS = 64
SAMPLES = 3
SEED = 42


def held_back(text):
    """Picks the same pieces every time, by a fingerprint of their text."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 10 == 0


def tokens(tokenizer, pieces):
    return [number for piece in pieces for number in tokenizer.encode(piece + "\\n", add_special_tokens=False)]


def windows(numbers):
    """Pieces of WINDOW + 1 tokens: the model reads the first WINDOW and guesses each next one."""
    return [numbers[start:start + WINDOW + 1] for start in range(0, len(numbers) - WINDOW, WINDOW)]


def scores(model, batch):
    logits = model(input_ids=batch[:, :-1]).logits
    return functional.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten()), logits


def check(model, held, device):
    """Average loss on held-back text, and how often its top guess for the next token was right."""
    model.eval()
    total, right, count = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, len(held), BATCH):
            batch = torch.tensor(held[start:start + BATCH], device=device)
            loss, logits = scores(model, batch)
            total += loss.item() * batch[:, 1:].numel()
            right += (logits.argmax(dim=-1) == batch[:, 1:]).sum().item()
            count += batch[:, 1:].numel()
    return total / count, right / count


def samples(model, tokenizer, pieces, original, best):
    """A few held-back pieces: how the original and the adjusted writer carry on from their first words."""
    chosen = [piece for piece in pieces if len(piece.split()) >= 16][:SAMPLES]
    rows = []
    for number, piece in enumerate(chosen):
        words = piece.split()
        prompt = " ".join(words[:8])
        row = {"prompt": prompt, "real": " ".join(words[8:48])}
        for key, state in (("before", original), ("after", best)):
            model.load_state_dict(state, strict=False)
            row[key] = write(model, tokenizer, prompt, new_tokens=60, seed=number)[len(prompt):].strip()
        rows.append(row)
    return rows


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = best_device()
    with open(DATA_FILE, encoding="utf-8") as file:
        pieces = [line.strip() for line in file if line.strip()]
    tokenizer = load_tokenizer()
    train_pieces = [piece for piece in pieces if not held_back(piece)]
    check_pieces = [piece for piece in pieces if held_back(piece)]
    train_ids, check_ids = tokens(tokenizer, train_pieces), tokens(tokenizer, check_pieces)
    if len(check_ids) < 2 * WINDOW + 2:
        # Too few pieces to hold some back whole, so the last tenth of the text is used instead.
        everything = tokens(tokenizer, pieces)
        cut = int(len(everything) * 0.9)
        train_ids, check_ids, check_pieces = everything[:cut], everything[cut:], []
    train, held = windows(train_ids), windows(check_ids)[:MAX_CHECK_WINDOWS]
    if len(train) < 4 or not held:
        raise SystemExit(f"There are only {len(train_ids) + len(check_ids):,} tokens of text. Add more text before training.")

    model = load_writer(adjusted=False, device=device)
    trainable = adjustable(model)
    original = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    print(f"Training on {len(train)} samples, checking on {len(held)}, using {device}.", flush=True)
    loss_before, accuracy_before = check(model, held, device)

    epochs = CONFIG["epochs"]
    per_epoch = math.ceil(len(train) / BATCH)
    optimizer = torch.optim.AdamW(trainable.values(), lr=LEARNING_RATE, weight_decay=0.0)
    best, best_loss, best_accuracy, best_epoch, step = copy.deepcopy(original), loss_before, accuracy_before, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train)
        total = 0.0
        for start in range(0, len(train), BATCH):
            batch = torch.tensor(train[start:start + BATCH], device=device)
            loss, _ = scores(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            optimizer.step()
            total += loss.item()
            step += 1
            print(f"Step {step}/{epochs * per_epoch}", flush=True)
        check_loss, accuracy = check(model, held, device)
        print(f"Epoch {epoch}/{epochs}  training loss {total / per_epoch:.4f}  check loss {check_loss:.4f}  check accuracy {accuracy * 100:.1f}%", flush=True)
        if check_loss < best_loss:
            best = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
            best_loss, best_accuracy, best_epoch = check_loss, accuracy, epoch

    os.makedirs(os.path.dirname(ADJUSTED_FILE), exist_ok=True)
    torch.save(best, ADJUSTED_FILE + ".partial")
    os.replace(ADJUSTED_FILE + ".partial", ADJUSTED_FILE)
    warnings = []
    if best_epoch == 0:
        warnings.append("No adjusted version did better on held-back text than the original writer, so the original is kept. "
                        "Add more text, or train for longer.")
    metrics = {
        "task": "writer", "model": CONFIG["model"], "adjustLayers": CONFIG["adjust_layers"], "pieces": len(pieces),
        "words": sum(len(piece.split()) for piece in pieces), "trainTokens": len(train_ids), "checkTokens": len(check_ids),
        "lossBefore": loss_before, "lossAfter": best_loss, "accuracyBefore": accuracy_before, "accuracyAfter": best_accuracy,
        "bestEpoch": best_epoch, "epochs": epochs, "warnings": warnings,
        "samples": samples(model, tokenizer, check_pieces, original, best),
    }
    with open("data/metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file)
    print(f"Saved the adjusted layers to {ADJUSTED_FILE}.")


if __name__ == "__main__":
    main()
'''

_CHAT_PY = '''"""Writes with your adjusted writer. Train first (or not: it then uses the writer as downloaded).

  python chat.py "Once upon a time"            continues your words
  python chat.py --chat "Suggest a bakery name"  answers, like a chat
  python chat.py --chat                          a conversation, until you type quit
"""
import contextlib
import sys

from model import best_device, load_tokenizer, load_writer, write


def main():
    chat = "--chat" in sys.argv[1:]
    prompt = " ".join(word for word in sys.argv[1:] if word != "--chat")
    with contextlib.redirect_stdout(sys.stderr):
        tokenizer, model = load_tokenizer(), load_writer(device=best_device())
    if prompt or not chat:
        print(write(model, tokenizer, prompt or "Once upon a time", chat=chat))
        return
    history = []
    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            break
        if line.lower() in ("quit", "exit"):
            break
        answer = write(model, tokenizer, line, chat=True, history=history[-12:])
        print(f"Writer: {answer}")
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
'''

_README = """# {name}

A small open language model, {model}, adjusted to write like your text. Made in AI Bricks.
Everything runs on your computer. The model downloads once from Hugging Face ({download}), then works offline.

## Use it

```
pip install -r requirements.txt
python chat.py "Once upon a time"
python chat.py --chat "Suggest a name for a bakery"
python chat.py --chat
```

The last one is a conversation. Type `quit` to stop.

## Train it again

Put your text in `data/corpus.txt`, one piece per line, then run `python train.py`.
About 1 in 10 pieces is held back to check it on. Only its last {layers} layers change, so it keeps what
it already knows, and `model/adjusted.pt` holds just those layers.

## What to expect

It's a small model. It writes in your style and can chat, but it can be wrong, repeat itself or wander off
the subject. Check anything it says before relying on it.
"""


def project_files(name: str, config: dict) -> dict[str, str]:
    spec = WRITERS[config["model"]]
    return {
        "config.json": json.dumps(config, indent=2) + "\n", "model.py": _MODEL_PY, "train.py": _TRAIN_PY, "chat.py": _CHAT_PY,
        "README.md": _README.format(name=name, model=config["model"], download=spec["download"], layers=config["adjust_layers"]),
        "requirements.txt": "torch>=2.2\ntransformers>=4.56\n",
    }


def start(dataset: str, model: str, strength: str, length: str) -> training.Job:
    if model not in WRITERS or strength not in STRENGTHS or length not in LENGTHS:
        raise TrainingError(422, "Choose a writer, how much it changes and how long to train.")
    with training._lock:
        meta = _require_writing(dataset)
        found = analyse(dataset)
        if found["problems"]:
            raise TrainingError(422, " ".join(found["problems"]))
        config = {"model": model, "repo": WRITERS[model]["repo"], "adjust_layers": STRENGTHS[strength], "epochs": LENGTHS[length], "window": 128}
        files = project_files(meta["name"], config)
        return training.start_run(f"{meta['name']}: {model} writer"[:120], codegen.slug(meta["name"]), codegen.build_hash(files), files,
                                  {"Words": found["words"]}, dataset, versions.freeze_corpus, "writer", 0.0)


def _writer_job(job_id: str) -> training.Job:
    job = training.stored(job_id)
    if job.task != "writer":
        raise TrainingError(404, "That model doesn't write.")
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This model's file is no longer saved.")
    return job


LIVE_RUNNER = """
import contextlib, json, sys
sys.path.insert(0, '.')
with contextlib.redirect_stdout(sys.stderr):
    import torch
    from model import ADJUSTED_FILE, adjustable, best_device, load_tokenizer, load_writer, write
    tokenizer = load_tokenizer()
    model = load_writer(adjusted=False, device=best_device())
    original = {name: value.detach().cpu().clone() for name, value in adjustable(model).items()}
    adjusted = torch.load(ADJUSTED_FILE, weights_only=True)
    model.eval()
print('ready', flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        reply = {}
        with contextlib.redirect_stdout(sys.stderr):
            for key, state in (('original', original), ('adjusted', adjusted)):
                if key == 'original' and not request.get('compare'):
                    continue
                model.load_state_dict(state, strict=False)
                reply[key] = write(model, tokenizer, request['prompt'], chat=request['chat'], history=request['history'], seed=request['seed'])
        print(json.dumps(reply), flush=True)
    except Exception as problem:
        print(json.dumps({'error': f'{type(problem).__name__}: {problem}'[:300]}), flush=True)
"""


def ask(job_id: str, prompt: str, chat: bool, history: list[dict], compare: bool, seed: int = 0) -> dict:
    """The adjusted writer's answer and, when asked, the original's, both from the same random choices."""
    job = _writer_job(job_id)
    prompt = prompt.strip()
    if not prompt:
        raise TrainingError(422, "Type something for it to answer or carry on from.")
    if len(prompt) > MAX_PROMPT:
        raise TrainingError(422, f"Keep it under {MAX_PROMPT:,} characters.")
    turns = [{"role": turn["role"], "content": turn["content"][:MAX_PROMPT]} for turn in history[-MAX_TURNS:]
             if turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str)]
    request = {"prompt": prompt, "chat": chat, "history": turns if chat else [], "compare": compare, "seed": seed}
    return training.ask_live_json(job, LIVE_RUNNER, request, ANSWER_SECONDS, "The writer couldn't answer that.")


def details(job_id: str) -> dict:
    job = _writer_job(job_id)
    config = json.loads((training._folder(job) / "config.json").read_text(encoding="utf-8"))
    return {"metrics": job.metrics, "config": config}
