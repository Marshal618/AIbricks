"""Find things in photos: drawing boxes, versions of them, and training a detector that draws its own."""
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from test_training import BINARY, workspace  # noqa: F401

from aibricks import finder
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}
SIZE = 160


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    from test_text_models import CACHE
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


def photo(seed):
    """A grey, speckled table with one to three red squares on it, and where each square is."""
    rng = np.random.default_rng(seed)
    base = (rng.normal(150, 12, (SIZE, SIZE, 1)) + rng.normal(0, 4, (SIZE, SIZE, 3))).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    boxes = []
    for index in range(1 + seed % 3):
        side = int(rng.integers(26, 40))
        x = 8 + index * 50 + int(rng.integers(0, 12))
        y = int(rng.integers(8, SIZE - side - 8))
        draw.rectangle((x, y, x + side, y + side), fill=(200, 30, 30))
        boxes.append({"category": "Square", "x": x / SIZE, "y": y / SIZE, "w": side / SIZE, "h": side / SIZE})
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue(), boxes


def create(count=0):
    reply = client.post("/api/datasets", json={"name": "Squares", "kind": "image", "goal": "find", "categories": ["Square"]})
    assert reply.status_code == 200, reply.text
    dataset = reply.json()
    assert dataset["goal"] == "find" and dataset["categories"] == ["Square"]
    truth = {}
    for seed in range(count):
        data, boxes = photo(seed)
        assert client.post("/api/label/images", params={"dataset": dataset["id"]}, content=data, headers=BINARY).status_code == 200
        truth[seed] = (data, boxes)
    return dataset["id"], truth


def ids(dataset):
    return [row["id"] for row in client.get(f"/api/datasets/{dataset}/boxes").json()["photos"]]


def box_all(dataset, truth):
    """Draws every square's box, as a person would."""
    import hashlib
    by_id = {hashlib.sha256(data).hexdigest()[:32] + ".png": boxes for data, boxes in truth.values()}
    for image_id in ids(dataset):
        reply = client.put(f"/api/datasets/{dataset}/boxes/{image_id}", json={"boxes": by_id[image_id]})
        assert reply.status_code == 200, reply.text


def wait(limit=900):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": "detect"}).json()["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.5)
    raise AssertionError("training did not finish")


def test_boxes_are_saved_counted_and_checked(workspace):  # noqa: F811
    dataset, truth = create(3)
    assert client.post("/api/datasets", json={"name": "Words", "kind": "text", "goal": "find"}).status_code == 422
    first, second, third = ids(dataset)
    summary = client.get(f"/api/datasets/{dataset}").json()
    assert summary["labelled"] == 0 and summary["unsorted"] == 3

    # Boxes are kept inside the photo, and a slip of the mouse is ignored.
    saved = client.put(f"/api/datasets/{dataset}/boxes/{first}", json={"boxes": [
        {"category": "Square", "x": 0.9, "y": -0.2, "w": 0.5, "h": 0.5},
        {"category": "Square", "x": 0.1, "y": 0.1, "w": 0.001, "h": 0.2},
    ]}).json()["boxes"]
    assert saved == [{"category": "Square", "x": 0.9, "y": 0.0, "w": 0.1, "h": 0.5}]
    assert client.put(f"/api/datasets/{dataset}/boxes/{second}", json={"boxes": [{"category": "Circle", "x": 0, "y": 0, "w": 0.5, "h": 0.5}]}).status_code == 422
    assert client.put(f"/api/datasets/{dataset}/boxes/{second}", json={"boxes": []}).status_code == 200  # nothing here
    listing = client.get(f"/api/datasets/{dataset}/boxes").json()
    assert listing["done"] == 2 and listing["photos"][0]["id"] == third and listing["photos"][0]["boxes"] is None
    summary = client.get(f"/api/datasets/{dataset}").json()
    assert summary["counts"] == {"Square": 1} and summary["labelled"] == 2 and summary["unsorted"] == 1

    # Photos get boxes, not categories.
    assert client.post(f"/api/label/images/{third}/assign", params={"dataset": dataset}, json={"category": "Square"}).status_code == 422
    assert client.post("/api/dataset/images", params={"dataset": dataset, "category": "Square"}, content=truth[0][0], headers=BINARY).status_code == 422

    # Renaming a thing renames its boxes.
    assert client.put(f"/api/datasets/{dataset}/categories", json={"old": "Square", "new": "Block"}).status_code == 200
    boxes = finder.read_boxes(dataset)
    assert boxes[first][0]["category"] == "Block"

    # Removing a photo hides its boxes; it isn't counted any more.
    assert client.post(f"/api/label/images/{first}/remove", params={"dataset": dataset}, headers=JSON).status_code == 200
    assert client.get(f"/api/datasets/{dataset}").json()["counts"] == {"Block": 0}

    health = client.get(f"/api/datasets/{dataset}/health").json()
    assert health["level"] == 0
    assert "at least 10 photos" in client.post(f"/api/datasets/{dataset}/train-find", headers=JSON).json()["detail"]["message"]
    assert client.post(f"/api/datasets/{dataset}/experiments", json={"minutes": 5}).status_code == 422


def test_versions_keep_boxes_and_restore_them(workspace):  # noqa: F811
    dataset, truth = create(2)
    first, second = ids(dataset)
    client.put(f"/api/datasets/{dataset}/boxes/{first}", json={"boxes": [{"category": "Square", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]})
    assert client.post(f"/api/datasets/{dataset}/versions", json={"message": "First boxes"}).status_code == 200
    # Moving a box is a change worth saving, though the photo still has the same thing in it.
    client.put(f"/api/datasets/{dataset}/boxes/{first}", json={"boxes": [{"category": "Square", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}]})
    overview = client.get(f"/api/datasets/{dataset}/versions").json()
    assert overview["canSave"] is True
    number = overview["versions"][0]["number"]
    # Removing the thing saves a version first, and restoring brings back the boxes.
    assert client.delete(f"/api/datasets/{dataset}/categories", params={"name": "Square"}).status_code == 200
    assert finder.read_boxes(dataset)[first] == []
    restored = client.post(f"/api/datasets/{dataset}/versions/{number}/restore", headers=JSON)
    assert restored.status_code == 200, restored.text
    assert restored.json()["dataset"]["categories"] == ["Square"]
    assert finder.read_boxes(dataset)[first][0]["x"] == 0.1
    assert set(ids(dataset)) == {first, second}


def test_it_learns_to_find_the_squares(workspace):  # noqa: F811
    dataset, truth = create(16)
    box_all(dataset, truth)
    assert client.get(f"/api/datasets/{dataset}").json()["counts"]["Square"] == sum(len(b) for _, b in truth.values())
    started = client.post(f"/api/datasets/{dataset}/train-find", headers=JSON)
    assert started.status_code == 200, started.text
    job = wait()
    assert job["status"] == "finished", job["error"]
    assert job["task"] == "detect" and job["trainCount"] == 13 and job["checkCount"] == 3
    metrics = job["metrics"]
    assert metrics["task"] == "detect" and metrics["photos"] == 3
    assert metrics["found"] >= 0.8 * (metrics["found"] + metrics["missed"]), metrics
    assert metrics["map50"] is not None and metrics["map50"] > 0.6, metrics
    assert job["review"]["total"] == 3

    # Live answers draw boxes where the squares are.
    data, boxes = photo(1001)  # three squares, never seen
    reply = client.post(f"/api/train/{job['id']}/predict-live", content=data, headers=BINARY)
    assert reply.status_code == 200, reply.text
    found = reply.json()
    assert found["counts"]["Square"] == 3, found
    for real in boxes:
        assert any(abs(box["x"] - real["x"]) < 0.06 and abs(box["y"] - real["y"]) < 0.06 for box in found["boxes"])

    # Asking for more sureness shows fewer boxes or the same; the setting is kept.
    assert client.put(f"/api/models/{job['id']}/find", json={"sure": 2}).status_code == 422
    assert client.put(f"/api/models/{job['id']}/find", json={"sure": 0.9}).json()["sure"] == 0.9
    assert client.get(f"/api/models/{job['id']}/find").json()["sure"] == 0.9

    # The model's card and download work like any other model's.
    assert client.get(f"/api/models/{job['id']}/card").status_code == 200
    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200 and len(download.content) > 1_000_000

    # The accurate detector is too big for a web page.
    app_page = client.get(f"/api/train/{job['id']}/app")
    assert app_page.status_code == 422 and "fast detector" in app_page.json()["detail"]["message"]


def test_the_fast_detector_learns_and_becomes_an_app(workspace):  # noqa: F811
    dataset, truth = create(16)
    box_all(dataset, truth)
    assert client.post(f"/api/datasets/{dataset}/train-find", params={"detector": "tiny"}, headers=JSON).status_code == 422
    started = client.post(f"/api/datasets/{dataset}/train-find", params={"detector": "fast"}, headers=JSON)
    assert started.status_code == 200, started.text
    job = wait()
    assert job["status"] == "finished", job["error"]
    metrics = job["metrics"]
    assert metrics["found"] >= 0.8 * (metrics["found"] + metrics["missed"]) and metrics["map50"] > 0.6, metrics
    assert client.get(f"/api/models/{job['id']}/find").json()["detector"] == "fast"

    data, boxes = photo(1001)
    found = client.post(f"/api/train/{job['id']}/predict-live", content=data, headers=BINARY).json()
    assert found["counts"]["Square"] == 3, found
    for real in boxes:
        assert any(abs(box["x"] - real["x"]) < 0.06 and abs(box["y"] - real["y"]) < 0.06 for box in found["boxes"])
    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200 and len(download.content) < 20_000_000  # about a fifth of the accurate one

    # Its web app carries the detector, which answers the page's check exactly as PyTorch did.
    import base64
    import json
    import math
    import re
    import onnxruntime
    from aibricks import webapp
    assert client.put(f"/api/models/{job['id']}/find", json={"sure": 0.7}).status_code == 200
    page = client.get(f"/api/train/{job['id']}/app")
    assert page.status_code == 200, page.text
    settings = json.loads(re.search(r'<script id="settings" type="application/json">(.*?)</script>', page.text, re.S).group(1))
    assert settings["kind"] == "detect" and settings["labels"] == ["Square"] and settings["sure"] == 0.7
    assert settings["about"] == "Finds and counts: Square." and settings["nothing"] == "Nothing found."
    session = onnxruntime.InferenceSession(base64.b64decode(re.search(r'<script id="model" type="application/octet-stream">(.*?)</script>', page.text, re.S).group(1)))
    pattern = np.array([math.sin(i * 0.37) * 0.5 + 0.5 for i in range(3 * 320 * 320)], dtype=np.float32).reshape(3, 320, 320)
    got_boxes, got_scores, _ = session.run(None, {"image": pattern})
    top = len(settings["check"]["scores"])
    assert np.allclose(got_scores[:top], settings["check"]["scores"], atol=1e-4)
    assert np.allclose(got_boxes[:top].reshape(-1), settings["check"]["boxes"], atol=1e-2)
    assert webapp.RUNTIME_INTEGRITY in page.text and webapp.VISION_URL not in page.text
