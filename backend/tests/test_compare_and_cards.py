"""Comparing two models fairly, and the model card that describes each one."""
import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from test_tables import create, houses, to_csv, upload, wait
from test_training import build, fill, mark_tested, wait_for_end, workspace  # noqa: F401

from aibricks import cards, compare, training
from aibricks.main import app

client = TestClient(app)


def train_table(dataset, columns, target="band", task="category"):
    reply = client.post(f"/api/datasets/{dataset}/train-table", json={"target": target, "task": task, "columns": columns})
    assert reply.status_code == 200, reply.text
    job = wait()
    assert job["status"] == "finished", job
    return job


def compared(dataset, first, second):
    return client.get(f"/api/datasets/{dataset}/compare", params={"first": first, "second": second})


# --------------------------------------------------------------------------- the maths

@pytest.mark.parametrize("wins, losses, expected", [(10, 0, 0.001953125), (12, 3, 0.03515625), (5, 5, 1.0), (0, 0, 1.0), (1, 0, 1.0)])
def test_the_chance_of_luck_matches_the_exact_binomial_test(wins, losses, expected):
    assert compare.chance_of_luck(wins, losses) == pytest.approx(expected)
    assert compare.chance_of_luck(losses, wins) == pytest.approx(expected)


def test_the_likely_range_is_the_wilson_interval():
    low, high = cards.likely_range(8, 10)
    assert low == pytest.approx(0.490, abs=0.001) and high == pytest.approx(0.943, abs=0.001)
    assert cards.likely_range(0, 0) == (0.0, 1.0)
    assert cards.likely_range(10, 10)[1] == 1.0


# ------------------------------------------------------------------- comparing models

def test_a_better_spreadsheet_model_wins_on_the_same_rows(workspace):
    dataset = create()
    assert upload(dataset, to_csv(*houses(400))).status_code == 200
    strong = train_table(dataset, ["size", "area"])
    weak = train_table(dataset, ["area"])
    assert strong["checkCount"] == weak["checkCount"]

    reply = compared(dataset, strong["id"], weak["id"])
    assert reply.status_code == 200, reply.text
    result = reply.json()
    # The same rows were held back both times, so every one of them is shared.
    assert result["shared"] == strong["checkCount"]
    assert result["verdict"] == "first" and result["chanceOfLuck"] < 0.05
    assert result["firstScore"] > result["secondScore"]
    assert result["summary"].startswith("Model A is better")
    assert result["onlyFirst"] + result["onlySecond"] + result["bothRight"] + result["neitherRight"] == result["shared"]

    flipped = compared(dataset, weak["id"], strong["id"]).json()
    assert flipped["verdict"] == "second" and flipped["chanceOfLuck"] == result["chanceOfLuck"]


def test_number_models_are_compared_by_which_was_closer(workspace):
    dataset = create()
    upload(dataset, to_csv(*houses(400)))
    strong = train_table(dataset, ["size", "area"], target="price", task="number")
    weak = train_table(dataset, ["area"], target="price", task="number")
    result = compared(dataset, strong["id"], weak["id"]).json()
    assert result["task"] == "number" and result["verdict"] == "first"
    assert result["firstScore"] < result["secondScore"]  # average error, so lower is better
    assert result["onlyFirst"] + result["onlySecond"] + result["ties"] == result["shared"]
    assert "closer on" in result["summary"]

    category = train_table(dataset, ["size"])
    assert compared(dataset, strong["id"], category["id"]).status_code == 422


def test_held_back_rows_stay_held_back_as_data_grows(workspace):
    dataset = create()
    header, rows = houses(400)
    upload(dataset, to_csv(header, rows[:300]))
    before = train_table(dataset, ["size", "area"])
    upload(dataset, to_csv(header, rows))
    after = train_table(dataset, ["size", "area"])
    first, second = (training.held_out(training.stored(job["id"]))["rows"] for job in (before, after))
    # Rows held back from the bigger table are mostly the ones held back before, plus new rows.
    old_rows_in_second = len(first.keys() & second.keys())
    assert old_rows_in_second >= 0.7 * len(first)
    assert compared(dataset, before["id"], after["id"]).json()["shared"] == old_rows_in_second


def test_comparisons_explain_what_they_cant_do(workspace):
    dataset = create()
    upload(dataset, to_csv(*houses(300)))
    job = train_table(dataset, ["size", "area"])
    assert compared(dataset, job["id"], job["id"]).status_code == 422
    assert compared(dataset, job["id"], "0123456789ab").status_code == 404
    second = train_table(dataset, ["size"])
    other = create("Other")
    reply = compared(other, job["id"], second["id"])
    assert reply.status_code == 404 and "wasn't trained on this dataset" in reply.json()["detail"]["message"]

    (training._folder(training.stored(second["id"])) / training.HELD_OUT_FILE).unlink()
    reply = compared(dataset, job["id"], second["id"])
    assert reply.status_code == 409 and "Train it again" in reply.json()["detail"]["message"]


def test_photo_models_trained_twice_are_checked_on_the_same_photos(workspace):
    fill()
    graph = build()
    mark_tested(graph)
    ids = []
    for _ in range(2):
        assert client.post("/api/train", json=graph).status_code == 200
        ids.append(wait_for_end()["id"])
    first, second = (training.held_out(training.stored(job_id))["rows"] for job_id in ids)
    assert len(first) == 8 and first.keys() == second.keys()


# ------------------------------------------------------------------------- model cards

def test_a_model_card_describes_the_run_and_keeps_notes(workspace):
    dataset = create()
    upload(dataset, to_csv(*houses(300)))
    job = train_table(dataset, ["size", "area"])

    card = client.get(f"/api/models/{job['id']}/card").json()
    titles = [section["title"] for section in card["sections"]]
    assert titles == ["Model details", "Intended use", "Training data", "How it was checked", "Results", "Limitations"]
    text = card["markdown"]
    assert "sorts rows by “band”" in text and "Columns it learned from: size, area" in text
    assert "The true figure is probably between" in text
    assert "Not written yet" in text

    reply = client.put(f"/api/models/{job['id']}/card", json={"intendedUse": "Rough price bands for listings", "notFor": "Valuations"})
    assert reply.status_code == 200
    assert "Use it for: Rough price bands for listings" in reply.json()["markdown"]
    assert client.get(f"/api/models/{job['id']}/card").json()["notes"]["notFor"] == "Valuations"
    assert client.put(f"/api/models/{job['id']}/card", json={"intendedUse": "x" * 2001}).status_code == 422

    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/train/{job['id']}/download").content))
    card_file = next(name for name in archive.namelist() if name.endswith("MODEL_CARD.md"))
    assert "Rough price bands" in archive.read(card_file).decode()


def test_a_card_warns_about_a_small_check(workspace):
    dataset = create()
    upload(dataset, to_csv(*houses(60)))
    job = train_table(dataset, ["size", "area"])
    limitations = next(s for s in client.get(f"/api/models/{job['id']}/card").json()["sections"] if s["title"] == "Limitations")
    assert any("checked on only" in line for line in limitations["lines"])


def test_a_number_model_card_reports_its_error(workspace):
    dataset = create()
    upload(dataset, to_csv(*houses(300)))
    job = train_table(dataset, ["size", "area"], target="price", task="number")
    text = client.get(f"/api/models/{job['id']}/card").json()["markdown"]
    assert "predicts the number in “price”" in text and "Average error:" in text and "R²:" in text


def test_cards_need_a_finished_model(workspace):
    assert client.get("/api/models/0123456789ab/card").status_code == 404
    assert client.get("/api/models/nope/card").status_code == 404
