"""Awkward inputs found by edge-case testing, each of which used to be accepted or misread."""
import types

import pytest
from fastapi.testclient import TestClient
from test_training import BINARY, picture, workspace  # noqa: F401

from aibricks import health, tables
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


def create(kind, categories=()):
    reply = client.post("/api/datasets", json={"name": "Edge", "kind": kind, "categories": list(categories)})
    assert reply.status_code == 200, reply.text
    return reply.json()["id"]


def add_image(dataset, data):
    return client.post("/api/label/images", params={"dataset": dataset}, content=data, headers=BINARY)


def add_text(dataset, data):
    return client.post("/api/label/text", params={"dataset": dataset}, content=data, headers=BINARY)


# ----------------------------------------------------------- the right data in each dataset

def test_photos_only_go_into_photo_datasets(workspace):
    for kind in ("text", "table"):
        reply = add_image(create(kind), picture((200, 0, 0)))
        assert reply.status_code == 422
        assert "can't hold photos" in reply.json()["detail"]["message"]


def test_text_only_goes_into_text_datasets(workspace):
    for kind in ("image", "table"):
        dataset = create(kind)
        reply = add_text(dataset, b"some words")
        assert reply.status_code == 422
        assert "can't hold text" in reply.json()["detail"]["message"]
        if kind == "table":
            assert client.get(f"/api/datasets/{dataset}").json()["rows"] in (0, None)


def test_open_data_import_checks_the_dataset_type_before_downloading(workspace):
    dataset = create("image", ["World"])
    reply = client.post(
        "/api/open-data/import", params={"dataset": dataset},
        json={"dataset": "fancyzhx/ag_news", "kind": "text", "mapping": {"World": "World"}, "perCategory": 5},
    )
    assert reply.status_code == 422
    assert client.get("/api/open-data/import", params={"dataset": dataset}).json()["job"] is None


# ------------------------------------------------------------ labels must be real categories

def test_a_photo_cant_be_filed_under_a_missing_category(workspace):
    dataset = create("image", ["Red", "Blue"])
    assert add_image(dataset, picture((200, 0, 0))).status_code == 200
    image = client.get("/api/label/images", params={"dataset": dataset}).json()["items"][0]["id"]

    reply = client.post(f"/api/label/images/{image}/assign", params={"dataset": dataset}, json={"category": "Green"})
    assert reply.status_code == 422
    # The photo is still waiting to be sorted, rather than hidden in a folder nothing reads.
    assert client.get("/api/label/images", params={"dataset": dataset}).json()["total"] == 1
    assert client.post(f"/api/label/images/{image}/assign", params={"dataset": dataset}, json={"category": "Red"}).status_code == 200
    assert client.get(f"/api/datasets/{dataset}").json()["counts"] == {"Red": 1, "Blue": 0}


def test_a_text_cant_be_given_a_missing_category(workspace):
    dataset = create("text", ["Sport", "Money"])
    assert add_text(dataset, b"a goal in the last minute").status_code == 200
    item = client.get("/api/label/text", params={"dataset": dataset}).json()["items"][0]["id"]

    reply = client.post(f"/api/label/text/{item}/assign", params={"dataset": dataset}, json={"category": "Weather"})
    assert reply.status_code == 422
    assert client.get("/api/label/text", params={"dataset": dataset}).json()["total"] == 1


def test_a_removed_category_cant_be_labelled_from_an_old_page(workspace):
    dataset = create("text", ["Sport", "Money"])
    add_text(dataset, b"shares fell sharply")
    item = client.get("/api/label/text", params={"dataset": dataset}).json()["items"][0]["id"]
    assert client.delete(f"/api/datasets/{dataset}/categories", params={"name": "Money"}).status_code == 200
    reply = client.post(f"/api/label/text/{item}/assign", params={"dataset": dataset}, json={"category": "Money"})
    assert reply.status_code == 422
    assert "Refresh" in reply.json()["detail"]["message"]


# ------------------------------------------------------------------- reviewing duplicates

def test_keeping_a_pair_needs_two_real_examples(workspace):
    dataset = create("text", ["Sport", "Money"])
    add_text(dataset, b"first\nsecond")
    items = [row["id"] for row in client.get("/api/label/text", params={"dataset": dataset}).json()["items"]]
    for item, name in zip(items, ("Sport", "Money")):
        client.post(f"/api/label/text/{item}/assign", params={"dataset": dataset}, json={"category": name})

    def keep(pair):
        return client.post(f"/api/datasets/{dataset}/health/review", json={"action": "keep-pair", "pair": pair})

    assert keep([items[0], items[0]]).status_code == 422
    assert keep([items[0]]).status_code == 422
    assert keep([items[0], "f" * 32]).status_code == 404
    assert keep(list(reversed(items))).status_code == 200


def test_a_kept_pair_matches_either_way_round():
    assert health._pair_key("b", "a") == health._pair_key("a", "b") == "a|b"


# ------------------------------------------------------------------ reading spreadsheet numbers

def test_semicolon_files_with_decimal_commas_are_read_as_numbers():
    header, rows, _ = tables._parse("name;price;code\nA;1.234,5;1,2\nB;3,25;x\nC;7;1.5\n".encode())
    assert [row[1] for row in rows] == ["1234.5", "3.25", "7"]
    # A column that mixes formats is left alone rather than guessed at.
    assert [row[2] for row in rows] == ["1,2", "x", "1.5"]


def test_comma_files_keep_commas_as_thousands():
    _, rows, _ = tables._parse(b'a,b\n"1,500",2\n')
    assert rows == [["1,500", "2"]]
    assert tables._number("1,500") == 1500


@pytest.mark.parametrize("text, expected", [
    ("$1,200", 1200), ("€9", 9), ("12%", 12), ("-3", -3), ("1e3", 1000),
    ("$", None), ("%", None), ("inf", None), ("nan", None), ("twelve", None),
])
def test_money_and_percentages_count_as_numbers(text, expected):
    assert tables._number(text) == expected


def test_the_model_reads_numbers_the_same_way_as_the_analysis():
    """If these disagreed, a column checked as numbers would reach the model as blanks."""
    pytest.importorskip("numpy")
    model = types.ModuleType("generated_model")
    exec(tables.project_files("T", "y", "number", [], None)["model.py"], model.__dict__)  # noqa: S102
    for text in ["$1,200", "€9", "12%", "-3", "1e3", "1,500", "", "NA", "inf", "twelve", " 4 "]:
        ours, theirs = tables._number(text), model.number(text)
        assert (ours is None and theirs != theirs) or ours == theirs, text


def test_a_number_column_with_one_value_is_refused(workspace):
    dataset = create("table")
    client.post(f"/api/datasets/{dataset}/table", content=b"x,y\n" + b"".join(f"{i % 7},5\n".encode() for i in range(60)), headers=BINARY)
    analysis = client.get(f"/api/datasets/{dataset}/table/analysis", params={"target": "y", "task": "number"}).json()
    assert any("same value" in problem for problem in analysis["problems"])
    reply = client.post(f"/api/datasets/{dataset}/train-table", json={"target": "y", "task": "number", "columns": ["x"]})
    assert reply.status_code == 422


# ----------------------------------------------------------------------- one job at a time

def test_training_waits_for_a_label_check(workspace, monkeypatch):
    dataset = create("table")
    rows = b"".join(f"{i % 9},{(i % 9) * 3 + i % 2}\n".encode() for i in range(80))
    client.post(f"/api/datasets/{dataset}/table", content=b"x,y\n" + rows, headers=BINARY)
    monkeypatch.setattr(health, "_job", {"status": "running", "dataset": "someotherset"})
    reply = client.post(f"/api/datasets/{dataset}/train-table", json={"target": "y", "task": "number", "columns": ["x"]})
    assert reply.status_code == 409
    assert "label check" in reply.json()["detail"]["message"]
