"""Training a text sorter from a saved dataset, linked to its version like an image model."""
import hashlib
import io
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_training import BINARY, RED, build, mark_tested, picture, wait_for_end, workspace  # noqa: F401

from aibricks import textmodels, training
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}
CACHE = Path(__file__).resolve().parents[1] / ".test-cache"

GOOD = ["what a wonderful and happy day", "I love this, it is fantastic", "great work, really impressive", "such a lovely kind person",
        "this made me smile all day", "brilliant result, well done", "the best meal I have ever had", "a delightful and charming film"]
BAD = ["what a terrible and awful day", "I hate this, it is horrible", "poor work, really disappointing", "such a rude nasty person",
       "this made me angry all day", "dreadful result, badly done", "the worst meal I have ever had", "a boring and painful film"]


@pytest.fixture(autouse=True)
def shared_cache(monkeypatch):
    # The base model is downloaded once and shared, rather than once per test.
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


def fill(good=GOOD, bad=BAD, name="Moods"):
    dataset = client.post("/api/datasets", json={"name": name, "kind": "text", "categories": ["Good", "Bad"]}).json()["id"]
    content = "\n".join([*good, *bad]).encode()
    assert client.post("/api/label/text", params={"dataset": dataset, "kind": "lines"}, content=content, headers=BINARY).status_code == 200
    for row in client.get("/api/label/text", params={"dataset": dataset}).json()["items"]:
        label = "Good" if row["text"] in good else "Bad"
        assert client.post(f"/api/label/text/{row['id']}/assign", params={"dataset": dataset}, json={"category": label}).status_code == 200
    return dataset


def train(dataset):
    reply = client.post(f"/api/datasets/{dataset}/train-text", headers=JSON)
    assert reply.status_code == 200, reply.text
    return reply.json()["job"]


def wait(limit=300):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": "text"}).json()["job"]
        if job["status"] != "running":
            return job
        time.sleep(0.3)
    raise AssertionError("training did not finish")


def test_training_needs_enough_labelled_text():
    too_few = fill(GOOD[:3], BAD[:6])
    reply = client.post(f"/api/datasets/{too_few}/train-text", headers=JSON)
    assert reply.status_code == 422 and "Good" in reply.json()["detail"]["message"]
    single = client.post("/api/datasets", json={"name": "One", "kind": "text", "categories": ["Only"]}).json()["id"]
    assert client.post(f"/api/datasets/{single}/train-text", headers=JSON).status_code == 422
    images = client.post("/api/datasets", json={"name": "Pics", "kind": "image", "categories": ["A", "B"]}).json()["id"]
    assert client.post(f"/api/datasets/{images}/train-text", headers=JSON).status_code == 422
    assert client.post(f"/api/datasets/{too_few}/train-text").status_code == 415
    assert client.post("/api/datasets/nope/train-text", headers=JSON).status_code == 404
    assert client.post("/api/models/nope/try", json={"text": "hi"}).status_code == 404
    assert client.get("/api/models/nope/mistakes").status_code == 404


def test_train_try_review_download_and_link_to_version(workspace):  # noqa: F811
    dataset = fill()
    started = train(dataset)
    assert started["task"] == "text" and started["datasetVersion"] == 1
    # Labelling during training doesn't change what this run uses.
    client.post("/api/label/text", params={"dataset": dataset, "kind": "lines"}, content=b"added while training\n", headers=BINARY)
    job = wait()
    assert job["status"] == "finished", job["error"]
    assert job["trainCount"] == 14 and job["checkCount"] == 2 and job["stepsDone"] == job["stepsTotal"] > 0
    assert job["categories"] == {"Good": 8, "Bad": 8} and job["review"]["total"] == 2
    assert job["best"]["accuracy"] is not None and job["hasModel"] is True

    run = workspace / "runs" / job["id"]
    # The run keeps no copy of the text once it ends.
    assert not (run / "data").exists()
    assert GOOD[0].encode() not in (run / "job.json").read_bytes()

    versions = client.get(f"/api/datasets/{dataset}/versions").json()
    assert versions["versions"][0]["automatic"] is True and versions["versions"][0]["labelled"] == 16
    models = client.get(f"/api/datasets/{dataset}/models").json()["models"]
    assert [row["id"] for row in models] == [job["id"]] and models[0]["task"] == "text" and models[0]["modelBytes"] > 1_000_000

    answers = client.post(f"/api/models/{job['id']}/try", json={"text": "what a wonderful, fantastic and lovely film"}).json()["answers"]
    assert answers[0]["category"] == "Good" and abs(sum(row["chance"] for row in answers) - 1) < 0.01
    assert client.post(f"/api/models/{job['id']}/try", json={"text": ""}).status_code == 422
    assert client.post(f"/api/models/{job['id']}/try", json={"text": "x" * 5001}).status_code == 422
    assert client.post(f"/api/train/{job['id']}/predict", content=picture(RED), headers=BINARY).status_code == 422

    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        names = archive.namelist()
        assert "moods/train.py" in names and "moods/model/head.pt" in names and "moods/model/encoder/model.safetensors" in names
        assert not [name for name in names if name.startswith("moods/data")]
        assert not any(GOOD[0].encode() in archive.read(name) for name in names if not name.endswith("safetensors"))
        archive.extractall(workspace / "download")

    # The download works on its own, away from the app.
    project = workspace / "download" / "moods"
    done = subprocess.run(
        [sys.executable, "-I", "-c", "import runpy,sys; sys.path.insert(0, '.'); sys.argv=['predict.py','this is awful and horrible','--json']; runpy.run_path('predict.py', run_name='__main__')"],
        cwd=project, capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert done.returncode == 0, done.stderr
    assert '"Bad"' in done.stdout


def test_mistakes_read_live_text_and_forget_deleted_text(workspace):  # noqa: F811
    dataset = fill()
    train(dataset)
    job = wait()
    assert job["status"] == "finished", job["error"]
    # A chosen mistake, so this doesn't depend on which texts the run held back or got wrong.
    item = hashlib.sha256(GOOD[0].encode()).hexdigest()[:32]
    stored = training.stored(job["id"])
    stored.review = {**job["review"], "mistakeCount": 1, "mistakes": [{"actual": "Good", "guess": "Bad", "chance": 0.9, "file": item}]}
    training._save(stored)
    rows = client.get(f"/api/models/{job['id']}/mistakes").json()
    assert rows["total"] == 1 and rows["mistakes"][0]["text"] == GOOD[0] and rows["mistakes"][0]["label"] == "Good"
    client.post(f"/api/label/text/{item}/remove", params={"dataset": dataset}, headers=JSON)
    assert client.get(f"/api/models/{job['id']}/mistakes").json()["mistakes"] == []
    assert client.delete(f"/api/datasets/{dataset}").status_code == 200
    assert client.get(f"/api/models/{job['id']}/mistakes").status_code == 404
    assert not (workspace / "collections" / dataset).exists()


def test_text_and_image_models_dont_hide_each_other():
    images = client.post("/api/datasets", json={"name": "Pics", "kind": "image", "categories": ["Red", "Blue"]}).json()["id"]
    for seed in range(10):
        for name, colour in (("Red", RED), ("Blue", (20, 20, 200))):
            client.post("/api/dataset/images", params={"dataset": images, "category": name}, content=picture(colour, seed), headers=BINARY)
    graph = build()
    mark_tested(graph)
    assert client.post("/api/train", params={"dataset": images}, json=graph).status_code == 200
    # Only one model trains at a time, of either kind.
    text = fill()
    assert client.post(f"/api/datasets/{text}/train-text", headers=JSON).status_code == 409
    image_job = wait_for_end()
    assert image_job["status"] == "finished"
    client.post("/api/label/images", params={"dataset": images}, content=picture(RED, 500), headers=BINARY)

    train(text)
    text_job = wait()
    assert text_job["status"] == "finished", text_job["error"]
    assert client.get("/api/train/current").json()["job"]["id"] == text_job["id"]
    assert client.get("/api/train/current", params={"kind": "image"}).json()["job"]["id"] == image_job["id"]
    # The image model can still be used and still makes suggestions.
    assert client.post(f"/api/train/{image_job['id']}/predict", content=picture(RED, 99), headers=BINARY).status_code == 200
    assert client.get("/api/label/images", params={"dataset": images}).json()["modelCategories"] == ["Blue", "Red"]


def test_stopping_and_failing_leave_no_model(monkeypatch):
    dataset = fill()
    job = train(dataset)
    assert client.delete(f"/api/train/{job['id']}").status_code == 200
    stopped = wait()
    assert stopped["status"] == "stopped" and stopped["hasModel"] is False
    assert client.post(f"/api/models/{job['id']}/try", json={"text": "hi"}).status_code == 409

    monkeypatch.setattr(textmodels, "BASE_MODEL", "./no-such-model")
    train(dataset)
    failed = wait()
    assert failed["status"] == "failed" and "base model" in failed["error"]
