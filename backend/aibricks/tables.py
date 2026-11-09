"""Spreadsheet datasets, and models that predict one column from the others.

Checks here catch columns that would make a model's score misleading, such as row IDs and
columns that give away the answer. The generated train.py does the training, so the downloaded
project behaves the same way.
"""
from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import math
import os
import re
import subprocess
import uuid

from . import codegen, datasets, training, versions
from .training import TrainingError

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_ROWS = 200_000
MAX_COLUMNS = 500
MAX_CLASSES = 50
MIN_ROWS = 20
MIN_PER_CLASS = 5
CHECK_ROWS = 5000
NUMBER_SHARE = 0.95
LEAK_LIMIT = 0.98
COMMON_CLASS_LIMIT = 0.95
_MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-"}
_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T].*)?$")


def _is_missing(value: str) -> bool:
    return value.strip().lower() in _MISSING


_DECIMAL_COMMA = re.compile(r"^[+-]?(\d{1,3}(\.\d{3})+|\d+)(,\d+)?$")


def _number(value: str) -> float | None:
    # Kept the same as number() in the generated model.py, so training and predicting agree.
    value = value.strip().replace(",", "")
    value = value[1:] if value[:1] in "$€£¥" else value
    value = value[:-1] if value.endswith("%") else value
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


# --------------------------------------------------------------- reading a file

def _parse(data: bytes) -> tuple[list[str], list[list[str]], int]:
    """Reads a CSV file, returning its header, its rows and how many damaged rows were skipped."""
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise TrainingError(422, "This file isn't UTF-8 text. Save it as CSV UTF-8 and try again.") from None
    first_line = content[:content.find("\n") if "\n" in content else len(content)]
    delimiter = max(",;\t", key=first_line.count)
    csv.field_size_limit(MAX_FILE_BYTES)
    try:
        reader = csv.reader(io.StringIO(content, newline=""), delimiter=delimiter)
        header = [name.strip() for name in next(reader, [])]
        if len(header) < 2:
            raise TrainingError(422, "This file needs a header row and at least 2 columns.")
        if len(header) > MAX_COLUMNS:
            raise TrainingError(422, f"This file has more than {MAX_COLUMNS} columns.")
        if any(not name for name in header):
            raise TrainingError(422, "Every column needs a name in the header row.")
        if len({name.casefold() for name in header}) != len(header):
            raise TrainingError(422, "Two columns have the same name.")
        rows, damaged = [], 0
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                damaged += 1
                continue
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise TrainingError(413, f"This file has more than {MAX_ROWS:,} rows.")
    except csv.Error:
        raise TrainingError(422, "This CSV file couldn't be read.") from None
    if not rows:
        raise TrainingError(422, "This file has no rows of data.")
    if delimiter == ";":
        _convert_decimal_commas(rows)
    return header, rows, damaged


def _convert_decimal_commas(rows: list[list[str]]) -> None:
    """Semicolon files usually come from places that write 1.234,5 for 1234.5, so those columns are rewritten."""
    for column in range(len(rows[0])):
        present = [row[column].strip() for row in rows if not _is_missing(row[column])]
        if present and any("," in value for value in present) and all(_DECIMAL_COMMA.match(value) for value in present):
            for row in rows:
                if not _is_missing(row[column]):
                    row[column] = row[column].strip().replace(".", "").replace(",", ".")


def read_table(dataset: str) -> tuple[list[str], list[list[str]]]:
    try:
        with open(datasets.table_file(dataset), encoding="utf-8", newline="") as file:
            csv.field_size_limit(MAX_FILE_BYTES)
            reader = csv.reader(file)
            header = next(reader)
            return header, list(reader)
    except (OSError, StopIteration):
        raise TrainingError(409, "Upload a CSV file first.") from None


def upload(dataset: str, data: bytes) -> dict:
    """Replaces the dataset's table. Earlier data is saved as a version first, so it's never lost."""
    if len(data) > MAX_FILE_BYTES:
        raise TrainingError(413, "This file is larger than 50 MB.")
    header, rows, damaged = _parse(data)
    with training._lock:
        meta = datasets._read(dataset)
        if meta["kind"] != "table":
            raise TrainingError(422, "Only spreadsheet datasets can hold a CSV table.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        saved_first = versions.save_before_change(dataset, "Saved automatically before replacing the data") if meta.get("rows") else None
        path = datasets.table_file(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{uuid.uuid4().hex}.partial")
        with open(temporary, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary, path)
        meta["rows"], meta["columns"] = len(rows), header
        datasets._write(meta)
        return {"dataset": datasets.get(dataset), "damagedRows": damaged, "savedFirst": saved_first}


# --------------------------------------------------------------- checking columns

def _column_type(values: list[str]) -> str:
    present = [value for value in values if not _is_missing(value)]
    if not present:
        return "empty"
    sample = present[:CHECK_ROWS]
    if sum(_number(value) is not None for value in sample) >= NUMBER_SHARE * len(sample):
        return "number"
    if sum(bool(_DATE.match(value.strip())) for value in sample) >= NUMBER_SHARE * len(sample):
        return "date"
    return "category"


def _is_counter(numbers: list[float]) -> bool:
    """True for a column like 1, 2, 3..., which identifies rows rather than describing them."""
    if len(numbers) < MIN_ROWS or len(set(numbers)) != len(numbers) or any(not number.is_integer() for number in numbers):
        return False
    ordered = sorted(numbers)
    return all(after - before == 1 for before, after in zip(ordered, ordered[1:]))


def _bucket(values: list[str], kind: str) -> list[str]:
    """Groups values so a column's link to the answer can be measured simply."""
    if kind != "number":
        return [value.strip().casefold() for value in values]
    numbers = sorted(number for number in map(_number, values) if number is not None)
    if not numbers:
        return ["" for _ in values]
    edges = [numbers[len(numbers) * step // 20] for step in range(1, 20)]

    def place(value: str) -> str:
        number = _number(value)
        return "" if number is None else str(sum(number >= edge for edge in edges))

    return [place(value) for value in values]


def _gives_away_answer(values: list[str], target: list[str], kind: str, task: str) -> bool:
    """Tests whether one column alone nearly predicts the answer on rows it hasn't seen."""
    usable = [(value, answer) for value, answer in zip(values, target) if not _is_missing(value)]
    if len(usable) < MIN_ROWS or len(usable) < 0.5 * len(target):
        return False
    values, target = [row[0] for row in usable], [row[1] for row in usable]
    if task == "number":
        answers = [_number(value) for value in target]
        numbers = [_number(value) for value in values] if kind == "number" else None
        if numbers and all(number is not None for number in numbers) and all(answer is not None for answer in answers):
            return abs(_correlation(numbers, answers)) >= LEAK_LIMIT
        return False
    common = max(target.count(name) for name in set(target)) / len(target)
    if common >= COMMON_CLASS_LIMIT:
        return False  # Nearly every row has the same answer, so any column looks predictive.
    groups = _bucket(values, kind)
    half = list(zip(groups, target))
    right = 0
    for learn, check in ((half[::2], half[1::2]), (half[1::2], half[::2])):
        seen: dict[str, dict[str, int]] = {}
        for group, answer in learn:
            seen.setdefault(group, {}).setdefault(answer, 0)
            seen[group][answer] += 1
        guesses = {group: max(counts, key=counts.get) for group, counts in seen.items()}
        right += sum(guesses.get(group) == answer for group, answer in check)
    return right / len(half) >= LEAK_LIMIT


def _correlation(first: list[float], second: list[float]) -> float:
    count = len(first)
    mean_first, mean_second = sum(first) / count, sum(second) / count
    top = sum((a - mean_first) * (b - mean_second) for a, b in zip(first, second))
    bottom = math.sqrt(sum((a - mean_first) ** 2 for a in first) * sum((b - mean_second) ** 2 for b in second))
    return top / bottom if bottom else 0.0


def _guess_task(values: list[str], kind: str) -> str:
    present = [value for value in values if not _is_missing(value)]
    return "number" if kind == "number" and len(set(present)) > 20 else "category"


def analyse(dataset: str, target: str | None = None, task: str | None = None) -> dict:
    """Describes each column for predicting the target, and which ones to leave out by default."""
    meta = datasets._read(dataset)
    if meta["kind"] != "table":
        raise TrainingError(422, "This is for spreadsheet datasets.")
    header, rows = read_table(dataset)
    target = target or header[-1]
    if target not in header:
        raise TrainingError(422, "That column isn't in this table.")
    position = header.index(target)
    target_values = [row[position] for row in rows]
    target_kind = _column_type(target_values)
    task = task or _guess_task(target_values, target_kind)

    usable = [row for row in rows if not _is_missing(row[position])]
    if task == "number":
        usable = [row for row in usable if _number(row[position]) is not None]
    problems, notes = [], []
    counts: dict[str, int] = {}
    if task == "category":
        for row in usable:
            counts[row[position].strip()] = counts.get(row[position].strip(), 0) + 1
        rare = [name for name, amount in counts.items() if amount < MIN_PER_CLASS]
        if len(counts) > MAX_CLASSES:
            problems.append(f"This column has {len(counts):,} different values, which is too many to predict as categories. If it holds numbers, predict it as a number instead.")
        elif rare and len(counts) - len(rare) >= 2:
            notes.append(f"Left out {sum(counts[name] for name in rare):,} rows whose answer appears fewer than {MIN_PER_CLASS} times.")
            usable = [row for row in usable if row[position].strip() not in rare]
            counts = {name: amount for name, amount in counts.items() if name not in rare}
        if len(counts) < 2 and not problems:
            problems.append(f"This column needs at least 2 different values that each appear {MIN_PER_CLASS} times or more.")
    elif target_kind != "number":
        problems.append("This column doesn't hold numbers, so predict it as a category instead.")
    elif len({_number(row[position]) for row in usable}) < 2:
        problems.append("Every row has the same value in this column, so there's nothing to predict.")
    left = len(rows) - len(usable)
    if left and not (task == "category" and notes):
        notes.append(f"Left out {left:,} rows with no usable answer.")
    if len(usable) < MIN_ROWS and not problems:
        problems.append(f"This needs at least {MIN_ROWS} rows with an answer. It has {len(usable):,}.")

    # A fixed spread of rows, so checks on a big table stay quick and repeatable.
    stride = max(1, len(usable) // CHECK_ROWS)
    sample = usable[::stride][:CHECK_ROWS]
    sample_target = [row[position].strip() for row in sample]
    columns = []
    for index, name in enumerate(header):
        if index == position:
            continue
        values = [row[index] for row in usable]
        kind = _column_type(values)
        present = [value.strip() for value in values if not _is_missing(value)]
        distinct = len(set(present))
        use, warning = True, None
        if kind == "empty" or distinct < 2:
            use, warning = False, "Every row has the same value, so it can't help."
        elif kind == "date":
            use, warning = False, "Dates aren't used yet."
        elif kind == "number" and _is_counter([_number(value) for value in present if _number(value) is not None]):
            use, warning = False, "Looks like a row number, not something that explains the answer."
        elif kind == "category" and distinct > 50 and distinct >= 0.5 * len(present):
            use, warning = False, "Almost every value is different, like an ID or free text, so it would only memorise rows."
        elif sample_target and _gives_away_answer([row[index] for row in sample], sample_target, kind, task):
            use, warning = False, "This almost gives away the answer. Only use it if you'd really know it before the answer is known."
        columns.append({"name": name, "kind": kind, "missing": len(values) - len(present), "distinct": distinct, "use": use, "warning": warning})
    if not any(column["use"] for column in columns) and not problems:
        notes.append("No columns are selected. Choose at least one to use.")
    return {
        "target": target, "task": task, "targetKind": target_kind, "rows": len(usable), "totalRows": len(rows),
        "columns": columns, "classes": dict(sorted(counts.items(), key=lambda item: -item[1])[:MAX_CLASSES]) if task == "category" else None,
        "problems": problems, "notes": notes, "headers": header,
    }


# ---------------------------------------------------------------- the generated project

_MODEL_PY = '''"""Shared code for the spreadsheet model: reading rows in the form the model expects."""
import csv
import json
import math

import numpy as np

MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-"}


def number(value):
    value = str(value).strip()
    if value.lower() in MISSING:
        return math.nan
    value = value.replace(",", "")
    value = value[1:] if value[:1] in "$€£¥" else value
    value = value[:-1] if value.endswith("%") else value
    try:
        parsed = float(value)
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def category(value):
    value = str(value).strip()
    return "" if value.lower() in MISSING else value


def to_array(rows, features):
    """Turns rows, each a dictionary of column names to values, into the model's input."""
    return np.array(
        [[number(row.get(item["name"], "")) if item["kind"] == "number" else category(row.get(item["name"], "")) for item in features] for row in rows],
        dtype=object,
    )


def read_csv(path):
    csv.field_size_limit(100_000_000)
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_config():
    with open("config.json", encoding="utf-8") as file:
        return json.load(file)
'''

_TRAIN_PY = '''"""Trains the model. Put your table in data/table.csv and choose columns in config.json, then run: python train.py"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from model import category, number, read_config, read_csv, to_array

DATA_FILE = "data/table.csv"
SEED = 42
HOLD_BACK = 0.2
IMPORTANCE_ROWS = 2000
CHOICES_SHOWN = 50
SUSPICIOUS_SCORE = 0.98
SUSPICIOUS_SHARE = 0.8


def fingerprint(row):
    """Identifies a row by its values, so the same rows are held back every time."""
    return hashlib.sha256("\\x1f".join(value or "" for value in row.values()).encode("utf-8")).hexdigest()[:32]


def split(keys, answers, task):
    """Holds back HOLD_BACK of the rows, picked by fingerprint rather than at random, so models
    trained on the same rows are checked on the same ones and can be compared fairly."""
    groups = {}
    for index, answer in enumerate(answers):
        groups.setdefault(answer if task == "category" else "", []).append(index)
    train, check = [], []
    for members in groups.values():
        members.sort(key=lambda index: keys[index])
        held = max(1, round(len(members) * HOLD_BACK)) if len(members) > 1 else 0
        check += members[:held]
        train += members[held:]
    return np.array(sorted(train)), np.array(sorted(check))


def build(features, task):
    categories = [index for index, item in enumerate(features) if item["kind"] != "number"]
    numbers = [index for index, item in enumerate(features) if item["kind"] == "number"]
    prepare = ColumnTransformer([
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=250), categories),
        ("numbers", "passthrough", numbers),
    ])
    settings = dict(random_state=SEED, categorical_features=list(range(len(categories))) or None)
    learner = HistGradientBoostingClassifier(**settings) if task == "category" else HistGradientBoostingRegressor(**settings)
    return Pipeline([("prepare", prepare), ("learn", learner)])


def save(model, info):
    """Saves in one step, so stopping early can't leave a half-written model."""
    partial = Path("model.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    joblib.dump(model, partial / "model.joblib")
    (partial / "info.json").write_text(json.dumps(info), encoding="utf-8")
    shutil.rmtree("model", ignore_errors=True)
    os.replace(partial, "model")


def main():
    config = read_config()
    target, task, features = config["target"], config["task"], config["features"]
    allowed = config.get("classes")
    rows, seen, repeats = [], set(), 0
    names = [item["name"] for item in features]
    for row in read_csv(DATA_FILE):
        answer = number(row[target]) if task == "number" else category(row[target])
        if answer == "" or answer != answer or (allowed and answer not in allowed):
            continue
        # A row repeated on both sides of the split would make the score look better than it is.
        # Only exact copies count: different people can share the same values in chosen columns.
        key = tuple(row.values())
        if key in seen:
            repeats += 1
            continue
        seen.add(key)
        rows.append((row, answer))
    keys = [fingerprint(row) for row, _ in rows]
    if not rows:
        sys.exit("No rows have a usable answer.")
    inputs = to_array([row for row, _ in rows], features)
    answers = np.array([answer for _, answer in rows], dtype=float if task == "number" else object)
    if task == "category":
        answers = answers.astype(str)
        _, tally = np.unique(answers, return_counts=True)
        if tally.min() < 2:
            sys.exit("Each answer needs at least 2 different rows.")
    train_rows, check_rows = split(keys, answers, task)
    train_x, check_x, train_y, check_y = inputs[train_rows], inputs[check_rows], answers[train_rows], answers[check_rows]
    print(f"Training on {len(train_x)} samples, checking on {len(check_x)}, using cpu.", flush=True)
    print("Step 1/3", flush=True)
    model = build(features, task)
    model.fit(train_x, train_y)
    print("Step 2/3", flush=True)

    guesses = model.predict(check_x)
    metrics = {"task": task, "repeatedRows": repeats, "warnings": []}
    if task == "category":
        classes = list(model.classes_)
        chances = model.predict_proba(check_x)
        accuracy = float(accuracy_score(check_y, guesses))
        check_loss = float(log_loss(check_y, chances, labels=classes))
        score = accuracy
        metrics |= {"accuracy": accuracy, "correct": int((guesses == check_y).sum()), "total": len(check_y), "baseline": float(np.max(np.unique(train_y, return_counts=True)[1]) / len(train_y))}
        metrics["perClass"] = {name: {"correct": int(((check_y == name) & (guesses == name)).sum()), "total": int((check_y == name).sum())} for name in classes}
    else:
        classes = None
        mae = float(mean_absolute_error(check_y, guesses))
        r2 = float(r2_score(check_y, guesses))
        check_loss = mae
        score = r2
        metrics |= {"mae": mae, "r2": r2, "baselineMae": float(mean_absolute_error(check_y, np.full(len(check_y), train_y.mean()))), "total": len(check_y)}

    sample = np.random.default_rng(SEED).permutation(len(check_x))[:IMPORTANCE_ROWS]
    effect = permutation_importance(model, check_x[sample], check_y[sample], n_repeats=5, random_state=SEED).importances_mean
    effect = np.clip(effect, 0, None)
    total = float(effect.sum())
    importance = sorted(({"name": name, "share": float(value / total) if total else 0.0} for name, value in zip(names, effect)), key=lambda item: -item["share"])
    metrics["importance"] = importance
    if score >= SUSPICIOUS_SCORE and importance and importance[0]["share"] >= SUSPICIOUS_SHARE:
        metrics["warnings"].append(f'Nearly all of this score comes from "{importance[0]["name"]}". Check that column isn\\'t a copy of the answer, or only known after it.')
    elif score >= SUSPICIOUS_SCORE:
        metrics["warnings"].append("This score is unusually high. Check that no column gives away the answer.")

    described = []
    for index, item in enumerate(features):
        entry = {"name": item["name"], "kind": item["kind"]}
        if item["kind"] != "number":
            values, amounts = np.unique([str(value) for value in inputs[:, index] if value != ""], return_counts=True)
            entry["choices"] = [str(value) for value in values[np.argsort(-amounts)][:CHOICES_SHOWN]]
        described.append(entry)
    save(model, {"task": task, "target": target, "features": described, "classes": classes})
    with open("data/metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file)
    # Each held-back row's answer, under its fingerprint only, so two models can be compared row by row.
    held = [[keys[index], answer.item() if hasattr(answer, "item") else answer, guess.item() if hasattr(guess, "item") else guess]
            for index, answer, guess in zip(check_rows, check_y, guesses)]
    with open("data/held_out.json", "w", encoding="utf-8") as file:
        json.dump(held, file)
    print("Step 3/3", flush=True)
    accuracy_text = f"{100 * score:.1f}" if task == "category" else "nan"
    print(f"Epoch 1/1  training loss nan  check loss {check_loss:.4f}  check accuracy {accuracy_text}%", flush=True)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''"""Asks the trained model about rows. Train first, then run: python predict.py new_rows.csv

With --json, it reads one row from standard input as a JSON object and prints the answer as JSON.
"""
import json
import sys
from pathlib import Path

import joblib

from model import read_csv, to_array


def load():
    info = json.loads(Path("model/info.json").read_text(encoding="utf-8"))
    return joblib.load("model/model.joblib"), info


def answer(model, info, rows):
    inputs = to_array(rows, info["features"])
    if info["task"] == "number":
        return [{"value": float(value)} for value in model.predict(inputs)]
    return [
        {"answers": sorted(({"category": str(name), "chance": float(chance)} for name, chance in zip(model.classes_, row)), key=lambda item: -item["chance"])}
        for row in model.predict_proba(inputs)
    ]


def main():
    model, info = load()
    if "--json" in sys.argv:
        print(json.dumps(answer(model, info, [json.loads(sys.stdin.read())])[0]))
        return
    if len(sys.argv) < 2:
        sys.exit("Give a CSV file with the same column names, for example: python predict.py new_rows.csv")
    for row in answer(model, info, read_csv(sys.argv[1])):
        print(row["value"] if "value" in row else row["answers"][0]["category"])


if __name__ == "__main__":
    main()
'''

_README = """# {name}

A spreadsheet model trained in AI Bricks. It predicts `{target}` as {task_text}.

It uses gradient-boosted trees from scikit-learn. Your table isn't included in this download.
The model does store your column names and category values, so treat it as private if those are.

## Use it

```
pip install -r requirements.txt
python predict.py new_rows.csv
```

`new_rows.csv` needs the same column names as the data this model learned from.

`model/model.joblib` is a Python pickle, so only open model files from people you trust. It
works best with the scikit-learn version in `requirements.txt`.

## Train it again

Put your table in `data/table.csv`, choose columns in `config.json`, then run `python train.py`.
"""


def _requirements() -> str:
    return "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in ("scikit-learn", "joblib", "numpy")) + "\n"


def project_files(name: str, target: str, task: str, features: list[dict], classes: list[str] | None) -> dict[str, str]:
    return {
        "model.py": _MODEL_PY,
        "train.py": _TRAIN_PY,
        "predict.py": _PREDICT_PY,
        "config.json": json.dumps({"target": target, "task": task, "features": features, "classes": classes}, indent=2),
        "README.md": _README.format(name=name, target=target, task_text="a number" if task == "number" else "a category"),
        "requirements.txt": _requirements(),
    }


# --------------------------------------------------------------------- training

def start(dataset: str, target: str, task: str, chosen: list[str]) -> training.Job:
    with training._lock:
        info = datasets.get(dataset)
        analysis = analyse(dataset, target, task)
        if analysis["problems"]:
            raise TrainingError(422, " ".join(analysis["problems"]))
        known = {column["name"]: column for column in analysis["columns"]}
        unknown = [name for name in chosen if name not in known]
        if unknown:
            raise TrainingError(422, f'"{unknown[0][:60]}" can\'t be used to predict this column.')
        features = [{"name": name, "kind": known[name]["kind"]} for name in chosen if known[name]["kind"] != "empty"]
        if not features:
            raise TrainingError(422, "Choose at least one column to learn from.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        classes = sorted(analysis["classes"]) if task == "category" else None
        amounts = analysis["classes"] if task == "category" else {target: analysis["rows"]}
        files = project_files(info["name"], target, task, features, classes)
        baseline = max(amounts.values()) / sum(amounts.values()) if task == "category" else 0.0
        project = f"{info['name']}: predict {target}"[:120]
        return training.start_run(project, codegen.slug(info["name"]), codegen.build_hash(files), files, amounts, dataset, versions.freeze, "table", baseline)


def _table_job(job_id: str) -> training.Job:
    job = training.stored(job_id)
    if job.task != "table":
        raise TrainingError(404, "That model doesn't use a spreadsheet.")
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This model's file is no longer saved.")
    return job


def schema(job_id: str) -> dict:
    """The columns a model needs, with its known choices, stored only with the model itself."""
    job = _table_job(job_id)
    return json.loads((training._model_path(job) / "info.json").read_text(encoding="utf-8"))


def try_row(job_id: str, values: dict[str, str]) -> dict:
    job = _table_job(job_id)
    folder = training._folder(job)
    try:
        with training._predictions:
            done = subprocess.run(
                training._python_command("predict.py", "--json"), cwd=folder, env=training._environment(folder) | {"OMP_NUM_THREADS": "2"},
                input=json.dumps(values), capture_output=True, text=True, errors="replace", timeout=training.PREDICT_LIMIT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        raise TrainingError(504, "The model took too long to answer.") from None
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise TrainingError(500, lines[-1][:600] if lines else "The model could not read these values.")
    return json.loads(done.stdout.strip().splitlines()[-1])
