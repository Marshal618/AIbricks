"""Outline things in photos: outlines saved and checked, click to outline, and a model that outlines by itself."""
import base64
import hashlib
import io
import math
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
WIDTH, HEIGHT = 240, 180


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    from test_text_models import CACHE
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


def photo(seed):
    """A grey, speckled table with one or two green leaves on it, and each leaf's outline."""
    rng = np.random.default_rng(seed)
    base = (rng.normal(150, 14, (HEIGHT, WIDTH, 1)) + rng.normal(0, 4, (HEIGHT, WIDTH, 3))).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    shapes = []
    for index in range(1 + seed % 2):
        cx = 60 + index * 110 + int(rng.integers(-10, 10))
        cy = int(rng.integers(55, HEIGHT - 55))
        r = int(rng.integers(28, 45))
        corners = [(cx + r * (0.7 + 0.3 * rng.random()) * math.cos(2 * math.pi * k / 7),
                    cy + r * (0.7 + 0.3 * rng.random()) * math.sin(2 * math.pi * k / 7)) for k in range(7)]
        draw.polygon(corners, fill=(40, 150, 60))
        shapes.append({"category": "Leaf", "points": [[x / WIDTH, y / HEIGHT] for x, y in corners]})
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue(), shapes


def filled(points, width=WIDTH, height=HEIGHT):
    canvas = Image.new("L", (width, height))
    ImageDraw.Draw(canvas).polygon([(x * width, y * height) for x, y in points], fill=1)
    return np.array(canvas) > 0


def overlap(a, b):
    return (a & b).sum() / (a | b).sum()


def create(count=0):
    reply = client.post("/api/datasets", json={"name": "Leaves", "kind": "image", "goal": "outline", "categories": ["Leaf"]})
    assert reply.status_code == 200, reply.text
    dataset = reply.json()
    assert dataset["goal"] == "outline"
    truth = {}
    for seed in range(count):
        data, shapes = photo(seed)
        assert client.post("/api/label/images", params={"dataset": dataset["id"]}, content=data, headers=BINARY).status_code == 200
        truth[hashlib.sha256(data).hexdigest()[:32] + ".png"] = (data, shapes)
    return dataset["id"], truth


def wait(limit=900):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": "outline"}).json()["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.5)
    raise AssertionError("training did not finish")


def test_outlines_are_saved_counted_and_checked(workspace):  # noqa: F811
    dataset, truth = create(3)
    assert client.post("/api/datasets", json={"name": "Words", "kind": "text", "goal": "outline"}).status_code == 422
    first, second, third = sorted(truth)

    # Corners are kept inside the photo, and each outline gets the box around it.
    saved = client.put(f"/api/datasets/{dataset}/boxes/{first}", json={"boxes": [
        {"category": "Leaf", "points": [[0.5, 0.5], [1.4, 0.5], [0.5, 0.9]]},
        {"category": "Leaf", "points": [[0.1, 0.1], [0.1001, 0.1], [0.1, 0.1001]]},  # a slip of the mouse
        {"category": "Leaf", "points": [[0.1, 0.1], [0.2, 0.2]]},  # not a shape
    ]}).json()["boxes"]
    assert saved == [{"category": "Leaf", "points": [[0.5, 0.5], [1.0, 0.5], [0.5, 0.9]], "x": 0.5, "y": 0.5, "w": 0.5, "h": 0.4}]
    assert client.put(f"/api/datasets/{dataset}/boxes/{second}", json={"boxes": [{"category": "Stone", "points": [[0, 0], [1, 0], [1, 1]]}]}).status_code == 422
    assert client.put(f"/api/datasets/{dataset}/boxes/{second}", json={"boxes": [{"category": "Leaf", "points": [[0, 0]] * 500}]}).status_code == 422
    assert client.put(f"/api/datasets/{dataset}/boxes/{second}", json={"boxes": []}).status_code == 200  # nothing here
    summary = client.get(f"/api/datasets/{dataset}").json()
    assert summary["counts"] == {"Leaf": 1} and summary["labelled"] == 2 and summary["unsorted"] == 1

    # Photos get outlines, not categories.
    assert client.post(f"/api/label/images/{third}/assign", params={"dataset": dataset}, json={"category": "Leaf"}).status_code == 422

    # Renaming a thing renames its outlines, and a version keeps them.
    assert client.put(f"/api/datasets/{dataset}/categories", json={"old": "Leaf", "new": "Petal"}).status_code == 200
    assert finder.read_boxes(dataset)[first][0]["category"] == "Petal"
    assert client.post(f"/api/datasets/{dataset}/versions", json={"message": "First outlines"}).status_code == 200
    client.put(f"/api/datasets/{dataset}/boxes/{first}", json={"boxes": []})
    versions = client.get(f"/api/datasets/{dataset}/versions").json()["versions"]
    assert client.post(f"/api/datasets/{dataset}/versions/{versions[0]['number']}/restore", headers=JSON).status_code == 200
    assert finder.read_boxes(dataset)[first][0]["points"] == [[0.5, 0.5], [1.0, 0.5], [0.5, 0.9]]

    # The data check says how far it is from training.
    health = client.get(f"/api/datasets/{dataset}/health").json()
    assert health["level"] == 0 and any("Outline at least 10 photos" in step for step in health["nextSteps"])
    assert client.post(f"/api/datasets/{dataset}/train-outline", headers=JSON).status_code == 422


def test_a_click_outlines_the_thing_under_it(workspace):  # noqa: F811
    dataset, truth = create(2)
    for image_id, (_, shapes) in truth.items():
        leaf = shapes[0]["points"]
        x, y = np.mean([p[0] for p in leaf]), np.mean([p[1] for p in leaf])
        reply = client.post(f"/api/datasets/{dataset}/outline/{image_id}", json={"clicks": [{"x": x, "y": y, "inside": True}]}, headers=JSON)
        assert reply.status_code == 200, reply.text
        found = reply.json()
        assert 3 <= len(found["points"]) <= 200 and found["box"]["w"] > 0
        assert overlap(filled(found["points"]), filled(leaf)) > 0.85
        # A second click on the table, marked as not part of it, keeps the leaf.
        again = client.post(f"/api/datasets/{dataset}/outline/{image_id}", json={"clicks": [
            {"x": x, "y": y, "inside": True}, {"x": 0.02, "y": 0.02, "inside": False}]}, headers=JSON).json()
        assert overlap(filled(again["points"]), filled(leaf)) > 0.85
    assert client.post(f"/api/datasets/{dataset}/outline/{image_id}", json={"clicks": [{"x": 0.5, "y": 0.5, "inside": False}]}, headers=JSON).status_code == 422
    assert client.post(f"/api/datasets/{dataset}/outline/nope.png", json={"clicks": [{"x": 0.5, "y": 0.5}]}, headers=JSON).status_code == 404
    words = client.post("/api/datasets", json={"name": "Sorted", "kind": "image", "categories": ["A", "B"]}).json()["id"]
    assert client.post(f"/api/datasets/{words}/outline/{image_id}", json={"clicks": [{"x": 0.5, "y": 0.5}]}, headers=JSON).status_code == 422


def test_it_learns_to_outline(workspace, tmp_path):  # noqa: F811
    dataset, truth = create(16)
    for image_id, (_, shapes) in truth.items():
        assert client.put(f"/api/datasets/{dataset}/boxes/{image_id}", json={"boxes": shapes}).status_code == 200
    started = client.post(f"/api/datasets/{dataset}/train-outline", headers=JSON)
    assert started.status_code == 200, started.text
    job = wait()
    assert job["status"] == "finished", job["error"]
    assert job["task"] == "outline" and job["trainCount"] + job["checkCount"] == 16
    metrics = job["metrics"]
    assert metrics["task"] == "outline" and metrics["meanIoU"] > 0.75, metrics
    assert metrics["perCategory"]["Leaf"]["photos"] == metrics["photos"] == 3
    assert len(metrics["photosChecked"]) == 3 and {"file", "score", "ious", "real", "mask"} <= set(metrics["photosChecked"][0])
    assert job["review"]["total"] == 3

    # It outlines a new photo, and says how much of it each thing covers.
    data, shapes = photo(1001)
    reply = client.post(f"/api/train/{job['id']}/predict-live", content=data, headers=BINARY)
    assert reply.status_code == 200, reply.text
    found = reply.json()
    real = sum(filled(shape["points"]).mean() for shape in shapes)
    assert abs(found["areas"]["Leaf"] - real) < 0.04 and found["answers"][0]["category"] == "Leaf"
    mask = np.array(Image.open(io.BytesIO(base64.b64decode(found["mask"]))))
    assert mask.shape == (320, 320, 4)
    covered = np.array(Image.fromarray(mask[:, :, 3] > 0).resize((WIDTH, HEIGHT)))
    assert overlap(covered, np.logical_or.reduce([filled(shape["points"]) for shape in shapes])) > 0.75

    # Its held-back outlines can be seen, and it works like any other model.
    assert client.get(f"/api/models/{job['id']}/outline-review/0").headers["content-type"] == "image/png"
    assert client.get(f"/api/models/{job['id']}/outline-review/99").status_code == 404
    assert client.get(f"/api/models/{job['id']}/outline").json()["categories"] == ["Leaf"]
    card = client.get(f"/api/models/{job['id']}/card")
    assert card.status_code == 200 and "LRASPP" in card.text
    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200

    # It becomes a web app whose model answers the page's check as PyTorch did, and a Raspberry Pi kit.
    import json
    import re
    import onnxruntime
    import pi_kit
    page = client.get(f"/api/train/{job['id']}/app")
    assert page.status_code == 200, page.text
    settings = json.loads(re.search(r'<script id="settings" type="application/json">(.*?)</script>', page.text, re.S).group(1))
    assert settings["kind"] == "outline" and settings["labels"] == ["Leaf"] and settings["rule"]["min"] == 1
    assert settings["about"].startswith("Outlines Leaf") and len(settings["colours"]) >= 1
    session = onnxruntime.InferenceSession(base64.b64decode(re.search(r'<script id="model" type="application/octet-stream">(.*?)</script>', page.text, re.S).group(1)))
    pattern = np.array([math.sin(i * 0.37) * 0.5 + 0.5 for i in range(3 * 320 * 320)], dtype=np.float32).reshape(1, 3, 320, 320)
    got = session.run(None, {"image": pattern})[0].reshape(-1)[::settings["check"]["step"]]
    assert np.allclose(got, settings["check"]["values"], atol=1e-3)

    folder, files = pi_kit.unpack(client.get(f"/api/train/{job['id']}/pi"), tmp_path)
    assert "points.task" not in files and "--when Leaf --pin 17" in (folder / "README.md").read_text()
    code, out, err = pi_kit.run(folder, "--check")
    assert code == 0 and "✓" in out, err
    (tmp_path / "leaves.png").write_bytes(data)
    answers = pi_kit.answers(folder, tmp_path / "leaves.png")
    assert answers[0]["name"] == "Leaf" and abs(answers[0]["chance"] - found["areas"]["Leaf"]) < 0.01
    assert pi_kit.run(folder, "--photo", tmp_path / "leaves.png")[1].strip().split(": ")[1].startswith("Leaf ")
