"""Forecasting: reading a table as a series of readings, and a model that's checked against simple guesses."""
import io
import math
import random
import subprocess
import sys
import time
import zipfile
from datetime import date, timedelta

from fastapi.testclient import TestClient
from test_tables import create, to_csv, upload
from test_training import workspace  # noqa: F401

from aibricks import forecast
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


def shop(days=420, seed=3, start=date(2024, 1, 1)):
    """Daily sales that rise slowly and peak every weekend, written day/month/year."""
    rng = random.Random(seed)
    rows = []
    for day in range(days):
        when = start + timedelta(days=day)
        sales = 100 + 0.1 * day + (40 if when.weekday() >= 5 else 0) + 10 * math.sin(day / 3) + rng.gauss(0, 3)
        rows.append([when.strftime("%d/%m/%Y"), f"Shop {day % 3}", round(sales, 2)])
    return ["day", "note", "sales"], rows


def analysis(dataset, **params):
    reply = client.get(f"/api/datasets/{dataset}/forecast/analysis", params=params)
    assert reply.status_code == 200, reply.text
    return reply.json()


def wait(limit=300):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": "forecast"}).json()["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("training did not finish")


def test_a_table_becomes_evenly_spaced_readings(workspace):  # noqa: F811
    dataset = create("Shop")
    header, rows = shop()
    assert upload(dataset, to_csv(header, rows)).status_code == 200
    found = analysis(dataset)
    assert found["timeColumns"] == ["day"] and found["valueColumns"] == ["sales"] and found["when"] == "day"
    assert found["stepName"] == "day" and found["readings"] == 420 and found["problems"] == []
    assert found["season"] == 7 and "weekly" in found["seasonText"] and found["horizon"] == 7
    assert found["start"] == "2024-01-01" and found["end"] == (date(2024, 1, 1) + timedelta(days=419)).isoformat()
    assert len(found["recent"]["values"]) == forecast.HISTORY_SHOWN

    # Missing days are filled in, and rows on the same day are combined the way chosen.
    gappy = [row for index, row in enumerate(rows) if index % 10 != 4] + [rows[0][:2] + [1000]]
    upload(dataset, to_csv(header, gappy))
    found = analysis(dataset, combine="sum")
    assert found["readings"] == 420 and found["duplicates"]
    assert any("filled in" in note for note in found["notes"]) and any("added together" in note for note in found["notes"])
    assert found["recent"]["values"][-1] == float(rows[-1][2])

    # Too far ahead for the readings there are, or too few readings, is explained.
    assert any("at most 84 days" in text for text in analysis(dataset, horizon=200)["problems"])
    upload(dataset, to_csv(header, rows[:30]))
    assert any("needs at least" in text for text in analysis(dataset)["problems"])
    assert client.post(f"/api/datasets/{dataset}/train-forecast", json={"when": "day", "value": "sales", "horizon": 7}).status_code == 422
    assert client.get(f"/api/datasets/{dataset}/forecast/analysis", params={"value": "note"}).status_code == 422

    # Months written as year-month, and rows with no date at all, work too.
    months = [[f"{1990 + index // 12}-{index % 12 + 1:02d}", 100 + index + 30 * math.sin(2 * math.pi * index / 12)] for index in range(144)]
    upload(dataset, to_csv(["month", "passengers"], months))
    found = analysis(dataset)
    assert found["stepName"] == "month" and found["season"] == 12 and found["horizon"] == 12 and found["end"] == "2001-12"
    assert upload(dataset, to_csv(["reading", "site"], [[index % 5, "North"] for index in range(100)])).status_code == 200
    found = analysis(dataset)
    assert found["when"] == "" and found["stepName"] == "row" and any("no date column" in note for note in found["notes"])

    # Formats that can't be told apart by one date are told apart by which gives even gaps.
    american = [[(date(2024, 1, 1) + timedelta(days=day)).strftime("%m/%d/%Y"), day % 7] for day in range(120)]
    upload(dataset, to_csv(["date", "visits"], american))
    assert analysis(dataset)["end"] == (date(2024, 1, 1) + timedelta(days=119)).isoformat()


def test_it_learns_to_forecast_and_beats_simple_guesses(workspace, tmp_path):  # noqa: F811
    dataset = create("Shop")
    header, rows = shop()
    upload(dataset, to_csv(header, rows))
    started = client.post(f"/api/datasets/{dataset}/train-forecast", json={"when": "day", "value": "sales", "horizon": 7})
    assert started.status_code == 200, started.text
    job = wait()
    assert job["status"] == "finished", job["error"]
    metrics = job["metrics"]
    assert metrics["task"] == "forecast" and metrics["horizon"] == 7 and metrics["season"] == 7
    assert metrics["method"] == "model" and metrics["mae"] < 0.6 * metrics["maeLast"] and metrics["mae"] < metrics["maeSeason"], metrics
    assert 0.5 <= metrics["inside"] <= 1 and len(metrics["maePerStep"]) == 7

    info = client.get(f"/api/models/{job['id']}/forecast").json()
    assert info["forecast"]["times"][0] == (date(2024, 1, 1) + timedelta(days=420)).isoformat() and len(info["forecast"]["values"]) == 7
    assert all(low <= value <= high for low, value, high in zip(info["forecast"]["low"], info["forecast"]["values"], info["forecast"]["high"]))
    # Next week's weekend should be forecast busier than its weekdays.
    weekend = [value for when, value in zip(info["forecast"]["times"], info["forecast"]["values"]) if date.fromisoformat(when).weekday() >= 5]
    weekdays = [value for when, value in zip(info["forecast"]["times"], info["forecast"]["values"]) if date.fromisoformat(when).weekday() < 5]
    assert min(weekend) > max(weekdays)
    assert info["backtest"]["times"][-1] == info["history"]["times"][-1] and len(info["backtest"]["real"]) == 7

    # It forecasts from other readings too, if there are enough of them.
    recent = [float(row[2]) for row in rows[-info["context"]:]]
    reply = client.post(f"/api/models/{job['id']}/forecast-from", json={"values": recent}, headers=JSON)
    assert reply.status_code == 200, reply.text
    assert [round(value, 3) for value in reply.json()["values"]] == [round(value, 3) for value in info["forecast"]["values"]]
    too_few = client.post(f"/api/models/{job['id']}/forecast-from", json={"values": recent[:3]}, headers=JSON)
    assert too_few.status_code == 422 and "at least 24" in too_few.json()["detail"]["message"]

    card = client.get(f"/api/models/{job['id']}/card").json()
    assert "forecasts “sales” 7 days ahead" in card["markdown"] and "same as last time" in card["markdown"]

    # The download forecasts the same way.
    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        archive.extractall(tmp_path)
    folder = next(path.parent for path in tmp_path.rglob("predict.py"))
    done = subprocess.run([sys.executable, "predict.py"], cwd=folder, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0].startswith(info["forecast"]["times"][0] + ": ")


def test_a_simple_guess_is_used_when_the_model_cant_beat_it(workspace):  # noqa: F811
    """A random walk can't be forecast better than "the same as last time", and the results should say so."""
    dataset = create("Walk")
    rng = random.Random(8)
    level, rows = 50.0, []
    for hour in range(600):
        level += rng.gauss(0, 1)
        rows.append([f"{(date(2025, 3, 1) + timedelta(days=hour // 24)).isoformat()} {hour % 24:02d}:00", round(level, 3)])
    upload(dataset, to_csv(["time", "price"], rows))
    found = analysis(dataset)
    assert found["stepName"] == "hour" and found["season"] == 0 and found["horizon"] == 24
    assert client.post(f"/api/datasets/{dataset}/train-forecast", json={"when": "time", "value": "price", "horizon": 6}).status_code == 200
    job = wait()
    assert job["status"] == "finished", job["error"]
    metrics = job["metrics"]
    assert metrics["mae"] <= 1.1 * metrics["maeLast"], metrics
    assert metrics["warnings"] and ("simple guess" in metrics["warnings"][0] or "no better" in metrics["warnings"][0])
    info = client.get(f"/api/models/{job['id']}/forecast").json()
    assert info["forecast"]["times"][0] == "2025-03-26 00:00"
