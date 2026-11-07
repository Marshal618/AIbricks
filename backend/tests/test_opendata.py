"""Importing open datasets, using a fake Hugging Face so that the tests run offline."""
import csv
import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from test_training import BLUE, RED, build, fill, mark_tested, picture, wait_for_end, workspace  # noqa: F401

from aibricks import opendata, training
from aibricks.main import app

client = TestClient(app)
GOOD_LINK = "https://datasets-server.huggingface.co/cached-assets/x/{}.png"
TOTAL = 400


def image_rows(offset, length, bad_every=0):
    # Stored by category, as many real datasets are: all of label 0, then all of label 1.
    rows = []
    for index in range(offset, min(offset + length, TOTAL)):
        rows.append({"row_idx": index, "row": {"image": {"src": GOOD_LINK.format(index)}, "labels": 0 if index < TOTAL // 2 else 1}, "truncated_cells": []})
    return rows


FEATURES = [
    {"name": "image", "type": {"_type": "Image"}},
    {"name": "labels", "type": {"_type": "ClassLabel", "names": ["cat", "dog"]}},
]


@pytest.fixture
def fake_hub(monkeypatch):
    state = {"features": FEATURES, "rows": image_rows, "tags": ["license:mit"], "gated": False, "bytes": lambda index: picture(RED if index < TOTAL // 2 else BLUE, index)}

    def get_json(url, params=None):
        params = params or {}
        if url == opendata.HUB:
            return [
                {"id": "open/cats", "downloads": 9, "tags": ["license:mit"]},
                {"id": "secret/gated", "downloads": 8, "tags": [], "gated": "auto"},
                {"id": "open/nolicence", "downloads": 7, "tags": []},
            ]
        if url.startswith(opendata.HUB + "/"):
            return {"tags": state["tags"], "gated": state["gated"]}
        if url.endswith("/splits"):
            return {"splits": [{"config": "default", "split": "test"}, {"config": "default", "split": "train"}]}
        assert params["split"] == "train"
        return {"features": state["features"], "rows": state["rows"](params["offset"], params["length"]), "num_rows_total": TOTAL}

    def fetch_image(link):
        return state["bytes"](int(link.rsplit("/", 1)[1].split(".")[0]))

    monkeypatch.setattr(opendata, "_get_json", get_json)
    monkeypatch.setattr(opendata, "_fetch_image", fetch_image)
    opendata._job = None
    yield state
    opendata.cancel()
    deadline = time.time() + 20
    while opendata._job and opendata._job["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
    opendata._job = None


def wait_for_import():
    deadline = time.time() + 60
    while time.time() < deadline:
        job = client.get("/api/open-data/import").json()["job"]
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("import did not finish")


def counts(*names):
    return client.get("/api/dataset", params={"category": list(names)}).json()["counts"]


def start(mapping=None, per=10, kind="image", dataset="open/cats"):
    return client.post("/api/open-data/import", json={"dataset": dataset, "kind": kind, "mapping": mapping or {"cat": "Red", "dog": "Blue"}, "perCategory": per})


def test_search_hides_gated_and_shows_licences(fake_hub):
    results = client.get("/api/open-data/search", params={"kind": "image", "q": "cats"}).json()["results"]
    assert [r["id"] for r in results] == ["open/cats", "open/nolicence"]
    assert results[0]["licence"] == "mit" and results[1]["licence"] == "unknown"


def test_inspect_reads_the_training_split_and_categories(fake_hub):
    info = client.get("/api/open-data/inspect", params={"kind": "image", "dataset": "open/cats"}).json()
    assert info["labels"] == ["cat", "dog"] and info["split"] == "train" and info["licence"] == "mit"
    assert info["namesUsable"] and info["totalRows"] == TOTAL and info["preview"][0]["image"].startswith("https://")
    assert "firstRows" not in info


@pytest.mark.parametrize("change, fragment", [
    ({"features": [FEATURES[0]]}, "no category column"),
    ({"features": [FEATURES[1]]}, "no image column"),
    ({"gated": "manual"}, "needs an account"),
])
def test_unusable_datasets_say_why(fake_hub, change, fragment):
    fake_hub.update(change)
    reply = client.get("/api/open-data/inspect", params={"kind": "image", "dataset": "open/cats"})
    assert reply.status_code == 422 and fragment in reply.json()["detail"]["message"]


def test_bad_dataset_names_never_reach_the_network(monkeypatch):
    monkeypatch.setattr(opendata, "_get_json", lambda *a, **k: pytest.fail("reached the network"))
    for bad in ["../../etc/passwd", "a/b/c", "http://evil.example/x", "a b", ""]:
        assert client.get("/api/open-data/inspect", params={"kind": "image", "dataset": bad}).status_code == 422


def test_import_balances_categories_even_when_stored_in_order(fake_hub):
    assert start(per=15).status_code == 200
    job = wait_for_import()
    assert job["status"] == "finished" and job["counts"] == {"Blue": 15, "Red": 15}, job
    assert counts("Red", "Blue") == {"Red": 15, "Blue": 15}
    sources = client.get("/api/open-data/import").json()["sources"]
    assert sources == [{"dataset": "open/cats", "licence": "mit", "added": 30, "url": "https://huggingface.co/datasets/open/cats"}]


def test_importing_twice_counts_duplicates_and_credits_once(fake_hub):
    start(per=5)
    wait_for_import()
    start(per=5)
    job = wait_for_import()
    assert job["added"] == 0 and job["duplicates"] == 10 and job["status"] == "finished"
    assert client.get("/api/open-data/import").json()["sources"][0]["added"] == 10


def test_skipping_a_category_and_merging_two(fake_hub):
    start(mapping={"cat": "Red"}, per=4)
    job = wait_for_import()
    assert job["counts"] == {"Red": 4} and counts("Red", "Blue") == {"Red": 4, "Blue": 0}
    start(mapping={"cat": "Pets", "dog": "Pets"}, per=6)
    assert wait_for_import()["counts"] == {"Pets": 6}
    assert start(mapping={"parrot": "Red"}).status_code == 422
    assert start(mapping={"cat": ""}).status_code == 422
    assert start(per=0).status_code == 422
    assert start(per=501).status_code == 422


def test_damaged_and_foreign_images_are_skipped(fake_hub):
    fake_hub["bytes"] = lambda index: b"not an image" if index % 2 else picture(RED if index < TOTAL // 2 else BLUE, index)
    start(per=5)
    job = wait_for_import()
    assert job["status"] == "finished" and job["counts"] == {"Blue": 5, "Red": 5} and job["skipped"] > 0

    def foreign(offset, length):
        rows = image_rows(offset, length)
        for row, host in zip(rows, ["evil.example", "huggingface.co.evil.example", "169.254.169.254"]):
            row["row"]["image"]["src"] = f"https://{host}/x/1.png"
        return rows

    assert opendata._safe_image_link({"src": "https://evil.example/1.png"}) is None
    assert opendata._safe_image_link({"src": "https://huggingface.co.evil.example/1.png"}) is None
    assert opendata._safe_image_link({"src": "http://huggingface.co/1.png"}) is None
    assert opendata._safe_image_link({"src": "https://cdn-lfs.huggingface.co/1.png"})
    assert opendata._safe_image_link("https://huggingface.co/1.png") is None
    fetched = []
    fake_hub["rows"] = foreign
    fake_hub["bytes"] = lambda index: fetched.append(index) or picture(BLUE, index + 9000)
    start(mapping={"cat": "Other"}, per=200)
    wait_for_import()
    assert all(index >= 3 for index in fetched)


def test_text_import_labels_and_exports(fake_hub):
    fake_hub["features"] = [
        {"name": "id", "type": {"_type": "Value", "dtype": "string"}},
        {"name": "sentence", "type": {"_type": "Value", "dtype": "string"}},
        {"name": "label", "type": {"_type": "ClassLabel", "names": ["ham", "spam"]}},
    ]

    def rows(offset, length):
        out = []
        for index in range(offset, min(offset + length, TOTAL)):
            out.append({"row": {"id": str(index), "sentence": f"<b>message</b> {index}", "label": index % 2}, "truncated_cells": ["sentence"] if index == 5 else []})
        out[0]["row"]["label"] = None
        out[-1]["row"]["sentence"] = "x" * 5001
        return out

    fake_hub["rows"] = rows
    info = client.get("/api/open-data/inspect", params={"kind": "text", "dataset": "open/cats"}).json()
    assert info["contentColumn"] == "sentence" and info["preview"][1]["text"].startswith("<b>")
    assert start(mapping={"ham": "Good", "spam": "Junk"}, per=8, kind="text").status_code == 200
    job = wait_for_import()
    assert job["counts"] == {"Good": 8, "Junk": 8} and job["status"] == "finished"
    queue = client.get("/api/label/text", params={"category": ["Good", "Junk"]}).json()
    assert queue["counts"] == {"Good": 8, "Junk": 8} and queue["total"] == 0
    rows_out = list(csv.DictReader(io.StringIO(client.get("/api/label/text/export", params={"category": ["Good", "Junk"]}).text)))
    assert len(rows_out) == 16 and {row["label"] for row in rows_out} == {"Good", "Junk"}
    assert all(len(row["text"]) <= 5000 for row in rows_out)


def test_import_can_be_cancelled(fake_hub):
    original = fake_hub["bytes"]

    def slow(index):
        time.sleep(0.05)
        return original(index)

    fake_hub["bytes"] = slow
    start(per=200)
    assert start(per=5).status_code == 409
    assert client.delete("/api/open-data/import").status_code == 200
    job = wait_for_import()
    assert job["status"] == "cancelled" and sum(job["counts"].values()) < 400


def test_label_names_that_cannot_be_categories_are_flagged():
    assert opendata._names_usable(["cat", "dog"])
    assert not opendata._names_usable(["a, b", "c"])
    assert not opendata._names_usable(["a/b", "c"])
    assert not opendata._names_usable(["same", "SAME"])
    assert not opendata._names_usable(["only one"])
    assert not opendata._names_usable(["<script>", "x"])


def test_download_credits_imported_data(fake_hub):
    start(per=12)
    wait_for_import()
    graph = build(classes="Red, Blue")
    mark_tested(graph)
    client.post("/api/train", json=graph)
    job = wait_for_end()
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/train/{job['id']}/download").content))
    credits = archive.read("colours/DATA_SOURCES.md").decode()
    assert "open/cats" in credits and "licence mit" in credits
    assert job["best"]["accuracy"] > 0.5


def test_failures_become_messages(monkeypatch, fake_hub):
    def broken(*a, **k):
        raise training.TrainingError(502, "Hugging Face is busy right now. Try again in a minute.")

    monkeypatch.setattr(opendata, "_get_json", broken)
    assert client.get("/api/open-data/search", params={"kind": "image"}).status_code == 502
    assert start().status_code == 502
