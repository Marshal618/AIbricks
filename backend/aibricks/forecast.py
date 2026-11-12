"""Forecasting: predicting the next readings of a column from the readings before it.

A spreadsheet with a date column (or rows in time order) becomes an evenly spaced series of
readings. A small network learns from windows of past readings to guess the next ones. It's
checked on the most recent stretch of readings, which it never learns from, against two simple
guesses: "the same as last time" and "the same as this time last season". If a simple guess
does better, forecasts use that instead, and the results say so.

The generated series.py reads the table into readings. This module runs the same code, so the
app and the downloaded project always see the data the same way.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import subprocess

import numpy as np

from . import codegen, datasets, tables, training, versions
from .training import TrainingError

MIN_READINGS = 40
MAX_HORIZON = 500
MAX_MISSING_SHARE = 0.3
HISTORY_SHOWN = 300
MAX_VALUES_SENT = 5000

_SERIES_PY = '''"""Reads a table into an evenly spaced series of readings. Shared by train.py and predict.py."""
import calendar
import csv
import math
import re
from datetime import datetime, timedelta

import numpy as np

MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-"}
MAX_POINTS = 100_000
CHECK_SHARE = 0.2
VAL_SHARE = 0.15
MAX_CONTEXT = 1024
MIN_CONTEXT = 8
MIN_WINDOWS = 20
_DATES = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%Y.%m.%d", "%d/%m/%y", "%m/%d/%y"]
_TIMES = ["", " %H:%M", " %H:%M:%S", "T%H:%M", "T%H:%M:%S"]
FORMATS = [date + time for date in _DATES for time in _TIMES] + ["%Y-%m", "%Y/%m", "%b %Y", "%B %Y", "%b-%Y", "%Y"]
_EXTRA = re.compile(r"([T ]\\d{1,2}:\\d{2}(?::\\d{2})?)(?:\\.\\d+)?(?:Z|[+-]\\d{2}:?\\d{2})?$")
_YEAR = re.compile(r"^\\d{4}$")
_STEPS = [(60, "minute"), (300, "5 minutes"), (600, "10 minutes"), (900, "15 minutes"), (1800, "30 minutes"),
          (3600, "hour"), (86400, "day"), (604800, "week")]
_MONTHS = {1: "month", 3: "quarter", 6: "half-year", 12: "year"}


class SeriesError(Exception):
    pass


def number(value):
    value = str(value).strip()
    if value.lower() in MISSING:
        return math.nan
    value = value.replace(",", "")
    value = value[1:] if value[:1] in "$\\u20ac\\u00a3\\u00a5" else value
    value = value[:-1] if value.endswith("%") else value
    try:
        parsed = float(value)
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def read_csv(path):
    csv.field_size_limit(100_000_000)
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_time(text, form):
    text = str(text).strip()
    if not text or text.lower() in MISSING:
        return None
    if form == "%Y" and not _YEAR.match(text):
        return None
    text = _EXTRA.sub(r"\\1", text[:-1] if text.endswith("Z") and "T" in text else text)
    try:
        return datetime.strptime(text, form)
    except ValueError:
        return None


def guess_format(values):
    """The time format that reads nearly all of these values, preferring the one with the most even gaps."""
    present = [str(value).strip() for value in values if str(value).strip() and str(value).strip().lower() not in MISSING]
    if not present:
        return None
    sample = list(dict.fromkeys(present))[:500]
    best = None
    for order, form in enumerate(FORMATS):
        parsed = [parse_time(value, form) for value in sample]
        if sum(time is not None for time in parsed) < 0.95 * len(sample):
            continue
        if form == "%Y":
            # A column of 4-digit numbers is only years if each appears once, in order.
            years = [time for time in parsed if time is not None]
            if len(set(years)) != len(years) or years not in (sorted(years), sorted(years, reverse=True)):
                continue
        times = sorted({time for time in parsed if time is not None})
        gaps = len({(after - before).total_seconds() for before, after in zip(times, times[1:])})
        if best is None or (gaps, order) < best[0]:
            best = ((gaps, order), form)
    return best[1] if best else None


def guess_step(times):
    times = sorted(set(times))
    if len(times) < 2:
        raise SeriesError("There need to be at least 2 different times.")
    gaps = sorted((after - before).total_seconds() for before, after in zip(times, times[1:]))
    middle = gaps[len(gaps) // 2]
    days = middle / 86400
    for low, high, months in ((27, 32, 1), (88, 93, 3), (180, 186, 6), (364, 367, 12)):
        if low <= days <= high:
            return {"unit": "months", "size": months}
    for seconds, _ in _STEPS:
        if abs(middle - seconds) <= 0.03 * seconds:
            return {"unit": "seconds", "size": seconds}
    return {"unit": "seconds", "size": max(1, round(middle))}


def step_name(step):
    if step["unit"] == "rows":
        return "row"
    if step["unit"] == "months":
        return _MONTHS.get(step["size"], f"{step['size']} months")
    for seconds, name in _STEPS:
        if step["size"] == seconds:
            return name
    size = step["size"]
    if size % 3600 == 0:
        return f"{size // 3600} hours"
    if size % 60 == 0:
        return f"{size // 60} minutes"
    return f"{size} seconds"


def add_months(time, count):
    month = time.month - 1 + count
    year, month = time.year + month // 12, month % 12 + 1
    return time.replace(year=year, month=month, day=min(time.day, calendar.monthrange(year, month)[1]))


def step_after(time, step, count):
    if step["unit"] == "months":
        return add_months(time, step["size"] * count)
    return time + timedelta(seconds=step["size"] * count)


def _position(time, start, step):
    if step["unit"] == "months":
        months = (time.year - start.year) * 12 + time.month - start.month
        return round(months / step["size"]), 0.0
    exact = (time - start).total_seconds() / step["size"]
    return round(exact), abs(exact - round(exact))


def label(time, step, index=None):
    """How a time is shown, so the same step always looks the same."""
    if time is None:
        return f"Row {index + 1}"
    if step["unit"] == "months":
        return time.strftime("%Y" if step["size"] == 12 else "%Y-%m")
    if step["size"] % 86400 == 0 and time.hour == 0 and time.minute == 0:
        return time.strftime("%Y-%m-%d")
    return time.strftime("%Y-%m-%d %H:%M")


def build(rows, config):
    """Turns rows into evenly spaced readings: rows at the same time are combined, and gaps filled in."""
    when, form, value, step = config.get("when") or "", config.get("format"), config["value"], config["step"]
    pairs, dropped = [], 0
    for row in rows:
        reading = number(row.get(value, ""))
        if when:
            time = parse_time(row.get(when, ""), form)
            if time is None:
                dropped += 1
                continue
        else:
            time = None
        pairs.append((time, reading))
    if len(pairs) < 2:
        raise SeriesError("There aren't enough rows with a readable time.")
    off_grid = 0
    if when:
        start = min(time for time, _ in pairs)
        positions = []
        for time, _ in pairs:
            position, off = _position(time, start, step)
            positions.append(position)
            off_grid += off > 0.25
    else:
        start = None
        positions = list(range(len(pairs)))
    size = max(positions) + 1
    if size > MAX_POINTS:
        raise SeriesError(f"From the first time to the last is more than {MAX_POINTS:,} steps of one {step_name(step)}. Check for a stray date far from the rest.")
    sums, counts = np.zeros(size), np.zeros(size)
    for position, (_, reading) in zip(positions, pairs):
        if not math.isnan(reading):
            sums[position] += reading
            counts[position] += 1
    combined = len(pairs) - len(set(positions))
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(counts > 0, sums if config.get("combine") == "sum" else sums / np.maximum(counts, 1), np.nan)
    missing = int(np.isnan(values).sum())
    if missing == size:
        raise SeriesError(f"No rows have a number in \\u201c{value}\\u201d.")
    if missing:
        index = np.arange(size)
        known = ~np.isnan(values)
        values = np.interp(index, index[known], values[known])
    times = [step_after(start, step, index) for index in range(size)] if when else None
    return {"values": values.astype(float), "times": times, "filled": missing, "combined": combined, "dropped": dropped, "offGrid": off_grid}


def plan(count, horizon, season):
    """How the readings are split: the newest are held back to check it, the ones before that pick when to stop training."""
    check = max(horizon, round(CHECK_SHARE * count))
    train = count - check
    val = max(horizon, round(VAL_SHARE * train))
    fit = train - val
    context = min(max(2 * season, 3 * horizon, 24), MAX_CONTEXT, fit - horizon - MIN_WINDOWS + 1)
    if context < MIN_CONTEXT:
        return None
    return {"check": check, "val": val, "fit": fit, "train": train, "context": int(context)}
'''

_series: dict = {}
exec(compile(_SERIES_PY, "series.py", "exec"), _series)  # noqa: S102 - our own template, run so the app reads data as the project does
SeriesError = _series["SeriesError"]

_SEASONS = {
    "minute": [60, 1440], "5 minutes": [12, 288], "10 minutes": [6, 144], "15 minutes": [4, 96], "30 minutes": [48, 336],
    "hour": [24, 168], "day": [7, 365], "week": [52], "month": [12], "quarter": [4], "half-year": [2], "row": [4, 7, 12, 24, 52],
}
_SEASON_NAMES = {
    ("minute", 60): "hourly", ("minute", 1440): "daily", ("5 minutes", 12): "hourly", ("5 minutes", 288): "daily",
    ("10 minutes", 6): "hourly", ("10 minutes", 144): "daily", ("15 minutes", 4): "hourly", ("15 minutes", 96): "daily",
    ("30 minutes", 48): "daily", ("30 minutes", 336): "weekly", ("hour", 24): "daily", ("hour", 168): "weekly",
    ("day", 7): "weekly", ("day", 365): "yearly", ("week", 52): "yearly", ("month", 12): "yearly", ("quarter", 4): "yearly",
    ("half-year", 2): "yearly",
}
_HORIZONS = {"minute": 60, "hour": 24, "day": 14, "week": 8, "month": 12, "quarter": 4, "half-year": 2, "year": 3}
_SEASON_LINK = 0.3


def _season(values: np.ndarray, name: str) -> int:
    """The repeating cycle in the readings, if there's a clear one, judged on the changes between readings so a trend doesn't fool it."""
    changes = np.diff(values)
    best, strength = 0, _SEASON_LINK
    for period in _SEASONS.get(name, []):
        if len(values) < 3 * period or len(changes) <= period:
            continue
        first, second = changes[period:], changes[:-period]
        if first.std() == 0 or second.std() == 0:
            continue
        link = float(np.corrcoef(first, second)[0, 1])
        if link > strength:
            best, strength = period, link
    return best


def season_text(name: str, season: int) -> str:
    if not season:
        return "No repeating pattern found"
    named = _SEASON_NAMES.get((name, season))
    return f"A {named} pattern, every {season} {name}s" if named else f"A pattern every {season} {name}s"


def _needed(horizon: int, season: int) -> int:
    count = MIN_READINGS
    while count < _series["MAX_POINTS"] and _series["plan"](count, horizon, season) is None:
        count += max(1, count // 20)
    return count


def analyse(dataset: str, when: str | None = None, value: str | None = None, horizon: int | None = None, combine: str | None = None) -> dict:
    """Finds the time and number columns, and describes the series a forecast would learn from."""
    meta = datasets._read(dataset)
    if meta["kind"] != "table":
        raise TrainingError(422, "Forecasting is for spreadsheet datasets.")
    header, rows = tables.read_table(dataset)
    records = [dict(zip(header, row)) for row in rows]
    times = []
    for name in header:
        form = _series["guess_format"]([row[header.index(name)] for row in rows[:5000]])
        if form:
            times.append({"name": name, "format": form})
    numbers = [name for name in header if name not in {item["name"] for item in times}
               and tables._column_type([row[header.index(name)] for row in rows]) == "number"]
    problems, notes = [], []
    if when is None:
        when = next((item["name"] for item in times if item["format"] != "%Y"), times[0]["name"] if times else "")
    if when and when not in {item["name"] for item in times}:
        raise TrainingError(422, "That column doesn't hold times.")
    form = next((item["format"] for item in times if item["name"] == when), None)
    if value is None:
        value = numbers[-1] if numbers else None
    if value is not None and value not in numbers:
        raise TrainingError(422, "That column doesn't hold numbers.")
    reply = {"timeColumns": [item["name"] for item in times], "valueColumns": numbers, "when": when, "value": value,
             "combine": combine or "mean", "problems": problems, "notes": notes, "readings": 0, "horizon": horizon or 0,
             "maxHorizon": 0, "season": 0, "seasonText": "", "step": None, "stepName": "", "duplicates": False,
             "start": None, "end": None, "recent": {"times": [], "values": []}, "needed": MIN_READINGS}
    if value is None:
        problems.append("This table has no column of numbers to forecast.")
        return reply

    if when:
        parsed = [_series["parse_time"](row.get(when, ""), form) for row in records]
        try:
            step = _series["guess_step"]([time for time in parsed if time is not None])
        except SeriesError as error:
            problems.append(str(error))
            return reply
    else:
        step = {"unit": "rows", "size": 1}
        notes.append("There's no date column, so each row is one step, in the order they are in the file.")
    config = {"when": when, "format": form, "value": value, "step": step, "combine": combine or "mean"}
    try:
        series = _series["build"](records, config)
    except SeriesError as error:
        problems.append(str(error))
        return reply
    values = series["values"]
    name = _series["step_name"](step)
    count = len(values)
    season = _season(values, name)
    max_horizon = int(max(1, min(MAX_HORIZON, count // 5)))
    chosen = horizon or (season if season and season <= 2 * _HORIZONS.get(name, 10) else _HORIZONS.get(name, 10))
    chosen = int(max(1, min(chosen, max_horizon)) if not horizon else horizon)
    reply |= {"readings": count, "horizon": chosen, "maxHorizon": max_horizon, "season": season, "seasonText": season_text(name, season),
              "step": step, "stepName": name, "duplicates": series["combined"] > 0, "needed": _needed(chosen, season)}
    labels = [_series["label"](time, step, index) for index, time in enumerate(series["times"] or [None] * count)]
    reply["start"], reply["end"] = labels[0], labels[-1]
    reply["recent"] = {"times": labels[-HISTORY_SHOWN:], "values": [float(item) for item in values[-HISTORY_SHOWN:]]}

    if series["dropped"]:
        notes.append(f"Left out {series['dropped']:,} rows whose time couldn't be read.")
    if series["combined"]:
        notes.append(f"{series['combined']:,} rows share a time with another row, so readings at the same time are "
                     f"{'added together' if config['combine'] == 'sum' else 'averaged'}.")
    if series["offGrid"] > 0.2 * count:
        notes.append(f"The times aren't evenly spaced, so each reading is placed at the nearest {name}.")
    if series["filled"] > MAX_MISSING_SHARE * count:
        problems.append(f"{round(100 * series['filled'] / count)}% of the {name}s from {labels[0]} to {labels[-1]} have no reading, "
                        "too many to fill in. Try a column with fewer gaps.")
    elif series["filled"]:
        notes.append(f"{series['filled']:,} {name}s had no reading, so they were filled in from the readings either side.")
    if float(np.ptp(values)) == 0:
        problems.append("Every reading is the same, so there's nothing to forecast.")
    if chosen > max_horizon:
        problems.append(f"With {count:,} readings it can forecast at most {max_horizon} {name}s ahead.")
    elif count < MIN_READINGS or _series["plan"](count, chosen, season) is None:
        problems.append(f"Forecasting {chosen} {name}s ahead needs at least {reply['needed']:,} readings, and this has {count:,}. "
                        "Forecast fewer steps ahead, or add more readings.")
    return reply


# ---------------------------------------------------------------- the generated project

_MODEL_PY = '''"""The forecasting network, and the simple guesses it's compared with."""
import numpy as np
import torch
from torch import nn

SEED = 42
HIDDEN = 128


class Forecaster(nn.Module):
    """Reads a window of past readings and guesses the next ones.

    Each window is scaled by its own average and spread first, as in RevIN (Kim et al., 2022),
    so it learns the shape of what comes next rather than the level. A straight-line part, as
    in DLinear (Zeng et al., 2023), and a small fully connected part add their guesses together.
    """

    def __init__(self, context, horizon):
        super().__init__()
        self.straight = nn.Linear(context, horizon)
        self.curved = nn.Sequential(
            nn.Linear(context, HIDDEN), nn.GELU(), nn.Dropout(0.1), nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, horizon),
        )

    def scaled(self, window):
        middle = window.mean(dim=1, keepdim=True)
        spread = window.std(dim=1, keepdim=True) + 1e-5 * (middle.abs() + 1)
        shape = (window - middle) / spread
        return self.straight(shape) + self.curved(shape), middle, spread

    def forward(self, window):
        guess, middle, spread = self.scaled(window)
        return guess * spread + middle


def simple_guess(method, history, horizon, season):
    """\\u201cThe same as last time\\u201d, or \\u201cthe same as this time last season\\u201d."""
    if method == "season":
        return np.array([history[len(history) - season + (step % season)] for step in range(horizon)], dtype=float)
    return np.full(horizon, float(history[-1]))


def guess(method, net, history, horizon, season, context):
    if method == "model":
        with torch.no_grad():
            window = torch.tensor(np.asarray(history[-context:], dtype=np.float32)).unsqueeze(0)
            return net(window)[0].double().numpy()
    return simple_guess(method, history, horizon, season)
'''

_TRAIN_PY = '''"""Learns to forecast. Put your table in data/table.csv and choose columns in config.json, then run: python train.py"""
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from model import SEED, Forecaster, guess, simple_guess
from series import SeriesError, build, label, plan, read_csv, step_after

DATA_FILE = "data/table.csv"
MAX_EPOCHS = 150
PATIENCE = 15
WINDOWS_PER_EPOCH = 4000
BATCH = 128
MAX_ORIGINS = 200
MAX_HELD_ROWS = 20000
LOW, HIGH = 0.1, 0.9
HISTORY_KEPT = 2000


def origins(start, end, horizon):
    """Points to forecast from, spread evenly through a stretch, each with a full horizon after it."""
    last = end - horizon
    if last < start:
        return []
    step = max(1, math.ceil((last - start + 1) / MAX_ORIGINS))
    return list(range(start, last + 1, step))


def errors(method, net, values, points, horizon, season, context):
    """For each starting point, the real readings minus the guess, using only readings before that point."""
    if method == "model" and points:
        with torch.no_grad():
            windows = torch.tensor(np.stack([values[point - context:point] for point in points]), dtype=torch.float32)
            guesses = net(windows).double().numpy()
    else:
        guesses = np.stack([simple_guess(method, values[:point], horizon, season) for point in points])
    real = np.stack([values[point:point + horizon] for point in points])
    return real - guesses, guesses


def train(values, end, horizon, context, val=None, epochs=MAX_EPOCHS, season=0):
    """Trains on readings before end. With val, stops when guesses on that stretch stop improving."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = Forecaster(context, horizon)
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    series = torch.tensor(values[:end], dtype=torch.float32)
    starts = np.arange(0, end - context - horizon + 1)
    ahead = torch.arange(context + horizon)
    best, best_epoch, best_state, waited = math.inf, epochs, None, 0
    for epoch in range(1, epochs + 1):
        net.train()
        chosen = rng.permutation(starts)[:WINDOWS_PER_EPOCH]
        total = 0.0
        for first in range(0, len(chosen), BATCH):
            rows = series[torch.tensor(chosen[first:first + BATCH]).unsqueeze(1) + ahead]
            window, target = rows[:, :context], rows[:, context:]
            guessed, middle, spread = net.scaled(window)
            loss = (guessed - (target - middle) / spread).abs().mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(window)
        if val is None:
            continue
        net.eval()
        check = float(np.abs(errors("model", net, values, val, horizon, season, context)[0]).mean())
        print(f"Epoch {epoch}/{epochs}  training loss {total / len(chosen):.4f}  check loss {check:.6g}  check accuracy nan%", flush=True)
        if check < best - 1e-9:
            best, best_epoch, waited = check, epoch, 0
            best_state = {key: item.clone() for key, item in net.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, best_epoch


def band(residuals):
    """How far off it was on readings it hadn't seen, as a band per step ahead. Few examples are pooled together."""
    if len(residuals) >= 10:
        return np.quantile(residuals, LOW, axis=0), np.quantile(residuals, HIGH, axis=0)
    pooled = residuals.reshape(-1)
    count = residuals.shape[1]
    return np.full(count, np.quantile(pooled, LOW)), np.full(count, np.quantile(pooled, HIGH))


def save(net, info):
    """Saves in one step, so stopping early can't leave a half-written model."""
    partial = Path("model.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    torch.save(net.state_dict(), partial / "model.pt")
    (partial / "info.json").write_text(json.dumps(info), encoding="utf-8")
    shutil.rmtree("model", ignore_errors=True)
    os.replace(partial, "model")


def main():
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    try:
        series = build(read_csv(DATA_FILE), config)
    except SeriesError as error:
        sys.exit(str(error))
    values, times = series["values"], series["times"]
    count, horizon, season, step = len(values), config["horizon"], config["season"], config["step"]
    split = plan(count, horizon, season)
    if split is None:
        sys.exit(f"There aren't enough readings to forecast {horizon} steps ahead.")
    context, fit, train_end = split["context"], split["fit"], split["train"]
    val_points = origins(fit, train_end, horizon)
    check_points = origins(train_end, count, horizon)
    print(f"Training on {fit - context - horizon + 1} samples, checking on {len(check_points)}, using cpu.", flush=True)
    print("Step 1/3", flush=True)

    net, epochs = train(values, fit, horizon, context, val=val_points, season=season)
    methods = ["model", "last"] + (["season"] if season else [])
    val_errors = {method: errors(method, net, values, val_points, horizon, season, context)[0] for method in methods}
    chosen = min(methods, key=lambda method: (float(np.abs(val_errors[method]).mean()), methods.index(method)))
    low, high = band(val_errors[chosen])
    print("Step 2/3", flush=True)

    checked = {method: errors(method, net, values, check_points, horizon, season, context) for method in methods}
    wrong, guesses = checked[chosen]
    real = wrong + guesses
    mae = {method: float(np.abs(checked[method][0]).mean()) for method in methods}
    inside = float(((real >= guesses + low) & (real <= guesses + high)).mean())
    typical = float(np.abs(values[train_end:]).mean())
    names = {"last": "the same as last time", "season": "the same as last season"}
    warnings = []
    best_simple = min((method for method in methods if method != "model"), key=lambda method: mae[method])
    if chosen != "model":
        warnings.append(f"A simple guess, {names[chosen]}, did better than the model before the check, so forecasts use that guess.")
    elif mae["model"] >= mae[best_simple]:
        warnings.append(f"On the most recent readings, the model did no better than guessing {names[best_simple]}.")
    if len(check_points) < 10:
        warnings.append(f"It was checked from only {len(check_points)} starting points, so treat the score as a rough guide.")

    # The final model learns from every reading, for the same number of passes that worked best before.
    final, _ = train(values, count, horizon, context, epochs=epochs)
    print("Step 3/3", flush=True)

    def labelled(index):
        return label(times[index] if times else None, step, index)

    def future(index):
        return label(step_after(times[-1], step, index + 1), step) if times else f"Row {count + index + 1}"

    ahead = guess(chosen, final, values, horizon, season, context)
    last_point = count - horizon
    back = guess(chosen, net, values[:last_point], horizon, season, context)
    kept = max(context, season, HISTORY_KEPT)
    info = {
        "value": config["value"], "when": config.get("when") or "", "step": step, "stepName": config["stepName"],
        "horizon": horizon, "context": context, "season": season, "method": chosen, "readings": count,
        "history": {"times": [labelled(index) for index in range(max(0, count - kept), count)], "values": [float(item) for item in values[-kept:]]},
        "forecast": {"times": [future(index) for index in range(horizon)], "values": ahead.tolist(),
                     "low": (ahead + low).tolist(), "high": (ahead + high).tolist()},
        "backtest": {"from": labelled(last_point), "times": [labelled(index) for index in range(last_point, count)], "values": back.tolist(),
                     "real": values[last_point:].tolist(), "low": (back + low).tolist(), "high": (back + high).tolist()},
        "band": {"low": low.tolist(), "high": high.tolist()},
    }
    save(final, info)
    metrics = {
        "task": "forecast", "readings": count, "horizon": horizon, "stepName": config["stepName"], "season": season,
        "method": chosen, "mae": mae[chosen], "maeModel": mae["model"], "maeLast": mae["last"], "maeSeason": mae.get("season"),
        "maePerStep": np.abs(wrong).mean(axis=0).tolist(), "inside": inside, "typical": typical, "origins": len(check_points),
        "filled": series["filled"], "epochs": epochs, "warnings": warnings,
    }
    Path("data/metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    every = max(1, math.ceil(len(check_points) * horizon / MAX_HELD_ROWS))
    held = [[f"{labelled(point)}|{ahead_step + 1}", float(real[row, ahead_step]), float(guesses[row, ahead_step])]
            for row, point in enumerate(check_points) for ahead_step in range(0, horizon, every)]
    Path("data/held_out.json").write_text(json.dumps(held), encoding="utf-8")
    print(f"Epoch {epochs}/{epochs}  training loss nan  check loss {mae[chosen]:.6g}  check accuracy nan%", flush=True)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''"""Forecasts the next readings.

python predict.py                shows the forecast made when it was trained
python predict.py recent.csv     forecasts what comes after the readings in recent.csv,
                                 which needs the same column names as the training table
With --json, it reads {"values": [...]} from standard input and prints a forecast as JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from model import Forecaster, guess
from series import SeriesError, build, read_csv


def load():
    info = json.loads(Path("model/info.json").read_text(encoding="utf-8"))
    net = Forecaster(info["context"], info["horizon"])
    net.load_state_dict(torch.load("model/model.pt", weights_only=True))
    net.eval()
    return net, info


def needed(info):
    return info["context"] if info["method"] == "model" else info["season"] if info["method"] == "season" else 1


def forecast(net, info, values):
    values = np.asarray(values, dtype=float)
    if len(values) < needed(info):
        raise ValueError(f"This model needs at least {needed(info)} readings to forecast from. It got {len(values)}.")
    ahead = guess(info["method"], net, values, info["horizon"], info["season"], info["context"])
    return {"values": ahead.tolist(), "low": (ahead + np.array(info["band"]["low"])).tolist(), "high": (ahead + np.array(info["band"]["high"])).tolist()}


def main():
    net, info = load()
    if "--json" in sys.argv:
        try:
            values = [float(item) for item in json.loads(sys.stdin.read())["values"]]
            print(json.dumps(forecast(net, info, values)))
        except ValueError as error:
            sys.exit(str(error))
        return
    if len(sys.argv) > 1:
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))
        try:
            values = build(read_csv(sys.argv[1]), config)["values"]
            reply = forecast(net, info, values)
        except (SeriesError, ValueError) as error:
            sys.exit(str(error))
        for step, (value, low, high) in enumerate(zip(reply["values"], reply["low"], reply["high"]), 1):
            print(f"{step} {info['stepName']}{'s' if step != 1 else ''} after the last reading: {value:.6g} (likely {low:.6g} to {high:.6g})")
        return
    saved = info["forecast"]
    for time, value, low, high in zip(saved["times"], saved["values"], saved["low"], saved["high"]):
        print(f"{time}: {value:.6g} (likely {low:.6g} to {high:.6g})")


if __name__ == "__main__":
    main()
'''

_README = """# {name}

A forecasting model trained in AI Bricks. It forecasts `{value}` {horizon} {step}s ahead.

It's a small PyTorch network that reads the last {context} readings and guesses the next ones. It was
checked on the most recent readings, which it never learned from, against simple guesses such as
"the same as last time". `model/info.json` says which one forecasts are made with.

Your table isn't included, but `model/info.json` keeps your most recent readings so it can forecast
from them. Treat it as private if they are.

## Use it

```
pip install -r requirements.txt
python predict.py               # the forecast made when it was trained
python predict.py recent.csv    # forecast after newer readings, with the same columns
```

## Train it again

Put your table in `data/table.csv`, choose columns in `config.json`, then run `python train.py`.
"""


def _requirements() -> str:
    return "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in ("torch", "numpy")) + "\n"


def project_files(name: str, config: dict) -> dict[str, str]:
    return {
        "series.py": _SERIES_PY,
        "model.py": _MODEL_PY,
        "train.py": _TRAIN_PY,
        "predict.py": _PREDICT_PY,
        "config.json": json.dumps(config, indent=2),
        "README.md": _README.format(name=name, value=config["value"], horizon=config["horizon"], step=config["stepName"], context=config["context"]),
        "requirements.txt": _requirements(),
    }


# --------------------------------------------------------------------- training

def start(dataset: str, when: str, value: str, horizon: int, combine: str) -> training.Job:
    with training._lock:
        info = datasets.get(dataset)
        analysis = analyse(dataset, when, value, horizon, combine)
        if analysis["problems"]:
            raise TrainingError(422, " ".join(analysis["problems"]))
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        split = _series["plan"](analysis["readings"], horizon, analysis["season"])
        form = None
        if analysis["when"]:
            header, rows = tables.read_table(dataset)
            position = header.index(analysis["when"])
            form = _series["guess_format"]([row[position] for row in rows[:5000]])
        config = {
            "when": analysis["when"], "format": form, "value": value, "step": analysis["step"], "stepName": analysis["stepName"],
            "combine": combine, "horizon": horizon, "season": analysis["season"], "context": split["context"],
        }
        files = project_files(info["name"], config)
        project = f"{info['name']}: forecast {value}"[:120]
        return training.start_run(project, codegen.slug(info["name"]), codegen.build_hash(files), files, {value: analysis["readings"]},
                                  dataset, versions.freeze, "forecast", 0.0)


def _forecast_job(job_id: str) -> training.Job:
    job = training.stored(job_id)
    if job.task != "forecast":
        raise TrainingError(404, "That model doesn't forecast.")
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This model's file is no longer saved.")
    return job


def details(job_id: str) -> dict:
    """The forecast, the recent readings and the check, stored only with the model itself."""
    job = _forecast_job(job_id)
    info = json.loads((training._model_path(job) / "info.json").read_text(encoding="utf-8"))
    shown = min(max(5 * info["horizon"], 35), HISTORY_SHOWN)
    info["history"] = {key: items[-shown:] for key, items in info["history"].items()}
    return info


def forecast_from(job_id: str, values: list[float]) -> dict:
    job = _forecast_job(job_id)
    if not values or any(not math.isfinite(value) for value in values):
        raise TrainingError(422, "Give a list of numbers to forecast from.")
    folder = training._folder(job)
    try:
        with training._predictions:
            done = subprocess.run(
                training._python_command("predict.py", "--json"), cwd=folder, env=training._environment(folder) | {"OMP_NUM_THREADS": "2"},
                input=json.dumps({"values": values}), capture_output=True, text=True, errors="replace", timeout=training.PREDICT_LIMIT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        raise TrainingError(504, "The model took too long to answer.") from None
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise TrainingError(422, lines[-1][:600] if lines else "The model couldn't forecast from these numbers.")
    return json.loads(done.stdout.strip().splitlines()[-1])
