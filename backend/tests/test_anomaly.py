"""Spot what's unusual: learning from normal photos only."""
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from test_training import BINARY, workspace  # noqa: F401

from aibricks import anomaly, training
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    from test_text_models import CACHE
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


def photo(seed, spot=None):
    """A wooden-looking board. An unusual one has a dark stain at `spot`, as (x, y) in 0-1."""
    rng = np.random.default_rng(seed)
    grain = np.sin(np.linspace(0, 30, 128)[None, :] + rng.uniform(0, 6)) * 18 + rng.normal(0, 6, (128, 128))
    base = np.stack([170 + grain, 125 + grain * 0.8, 80 + grain * 0.5], axis=-1).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(base)
    if spot:
        draw = ImageDraw.Draw(image)
        x, y = int(spot[0] * 128), int(spot[1] * 128)
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(30, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def create(normal=16, unusual=4):
    reply = client.post("/api/datasets", json={"name": "Boards", "kind": "image", "goal": "unusual", "categories": ["ignored"]})
    assert reply.status_code == 200, reply.text
    dataset = reply.json()
    assert dataset["categories"] == ["Normal", "Unusual"] and dataset["goal"] == "unusual"
    for seed in range(normal):
        assert client.post("/api/dataset/images", params={"dataset": dataset["id"], "category": "Normal"}, content=photo(seed), headers=BINARY).status_code == 200
    for seed in range(unusual):
        spot = (0.25 + 0.5 * (seed % 2), 0.3 + 0.4 * (seed // 2))
        client.post("/api/dataset/images", params={"dataset": dataset["id"], "category": "Unusual"}, content=photo(100 + seed, spot), headers=BINARY)
    return dataset["id"]


def wait(limit=600):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": "anomaly"}).json()["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.3)
    raise AssertionError("training did not finish")


def test_categories_are_fixed_and_photos_only(workspace):  # noqa: F811
    dataset = create(normal=0, unusual=0)
    assert client.post(f"/api/datasets/{dataset}/categories", json={"names": ["Other"]}).status_code == 422
    assert client.delete(f"/api/datasets/{dataset}/categories", params={"name": "Unusual"}).status_code == 422
    assert client.post("/api/datasets", json={"name": "Words", "kind": "text", "goal": "unusual"}).status_code == 422
    assert "at least 10 normal" in client.post(f"/api/datasets/{dataset}/train-unusual", headers=JSON).json()["detail"]["message"]
    assert client.get(f"/api/datasets/{dataset}/health").json()["level"] == 0
    assert client.post(f"/api/datasets/{dataset}/experiments", json={"minutes": 5}).status_code == 422


def test_it_learns_normal_and_spots_the_stains(workspace):  # noqa: F811
    dataset = create()
    started = client.post(f"/api/datasets/{dataset}/train-unusual", headers=JSON)
    assert started.status_code == 200, started.text
    job = wait()
    assert job["status"] == "finished", job["error"]
    assert job["task"] == "anomaly" and job["trainCount"] == 13 and job["checkCount"] == 7
    metrics = job["metrics"]
    assert metrics["task"] == "unusual" and metrics["checkedNormal"] == 3 and metrics["checkedUnusual"] == 4
    assert metrics["caught"] >= 3 and metrics["auroc"] >= 0.9, metrics
    assert len(metrics["scores"]) == 7

    # A new stained board, checked live: it's unusual, and the heatmap is brightest at the stain.
    reply = client.post(f"/api/train/{job['id']}/explain", content=photo(999, (0.75, 0.25)), headers=BINARY)
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["answers"][0]["category"] == "Unusual" and body["unusualness"] > 1
    heat = body["heatmap"]
    grid = np.array(heat["values"]).reshape(heat["height"], heat["width"])
    y, x = np.unravel_index(grid.argmax(), grid.shape)
    assert abs(x / heat["width"] - 0.75) < 0.2 and abs(y / heat["height"] - 0.25) < 0.2
    clean = client.post(f"/api/train/{job['id']}/predict-live", params={"explain": "true"}, content=photo(998), headers=BINARY).json()
    assert clean["answers"][0]["category"] == "Normal"

    # Moving the line: a higher factor flags less.
    settings = client.get(f"/api/models/{job['id']}/unusual").json()
    assert settings["factor"] == 1.0
    assert client.put(f"/api/models/{job['id']}/unusual", json={"factor": 3.0}).json()["factor"] == 3.0
    raised = client.post(f"/api/train/{job['id']}/predict-live", content=photo(999, (0.75, 0.25)), headers=BINARY).json()
    assert client.put(f"/api/models/{job['id']}/unusual", json={"factor": 9}).status_code == 422
    assert raised["answers"][0]["chance"] != body["answers"][0]["chance"]

    card = client.get(f"/api/models/{job['id']}/card").json()["markdown"]
    assert "PatchCore" in card
    training.close_live()
