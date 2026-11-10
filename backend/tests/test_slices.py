"""Find where it fails: groups of held-back photos a model gets wrong, named in words."""
import hashlib
import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_training import BINARY, build, fill, mark_tested, wait_for_end, workspace  # noqa: F401

from aibricks import slices, training
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    from test_text_models import CACHE
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


@pytest.fixture
def trained(workspace):  # noqa: F811
    fill()
    graph = build()
    mark_tested(graph)
    client.post("/api/train", json=graph)
    job = wait_for_end()
    assert job["status"] == "finished"
    return job["id"]


def scene(seed, dark=False):
    rng = np.random.default_rng(seed)
    image = np.zeros((128, 128, 3)) + rng.uniform(90, 210, 3)
    yy, xx = np.mgrid[:128, :128]
    image[(xx - 64) ** 2 + (yy - 64) ** 2 < (24 + rng.integers(0, 16)) ** 2] = rng.uniform(0, 255, 3)
    if dark:
        image *= 0.15
    buffer = io.BytesIO()
    Image.fromarray(image.clip(0, 255).astype(np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def plant(job_id, normal=30, dark=10):
    """Puts photos in the dataset and records the dark ones as the model's mistakes."""
    job = training.stored(job_id)
    rows = {}
    for index in range(normal + dark):
        is_dark = index >= normal
        category = "Red" if index % 2 else "Blue"
        data = scene(500 + index, is_dark)
        assert client.post("/api/dataset/images", params={"category": category}, content=data, headers=BINARY).status_code == 200
        name = hashlib.sha256(data).hexdigest()[:32] + ".png"  # photos are stored under their contents' fingerprint
        rows[name] = [category, ("Blue" if category == "Red" else "Red") if is_dark else category]
    (training._folder(job) / training.HELD_OUT_FILE).write_text(json.dumps({"task": "category", "rows": rows}))


def wait(job_id, limit=900):
    deadline = time.time() + limit
    while time.time() < deadline:
        state = client.get(f"/api/models/{job_id}/slices").json()
        if state["status"] not in ("running",):
            return state
        time.sleep(0.5)
    raise AssertionError("never finished")


def test_it_finds_and_names_the_dark_photos(trained):
    plant(trained)
    assert client.get(f"/api/models/{trained}/slices").json()["status"] == "none"
    assert client.post(f"/api/models/{trained}/slices", headers=JSON).status_code == 200
    result = wait(trained)
    assert result["status"] == "finished", result
    assert result["enough"] and result["mistakes"] == 10 and result["held"] == 40
    top = result["slices"][0]
    assert top["kind"] == "light", top
    assert top["wrong"] >= 6 and top["rate"] >= 0.7  # most dark photos, nearly all wrong
    assert top["confusion"]["times"] >= 4 and top["examples"] and "light" in top["advice"]
    # The result is kept with the model.
    assert client.get(f"/api/models/{trained}/slices").json()["slices"][0]["name"] == top["name"]


def test_too_few_mistakes_says_so(trained):
    plant(trained, normal=20, dark=2)
    result = client.post(f"/api/models/{trained}/slices", headers=JSON).json()
    assert result["enough"] is False and result["mistakes"] == 2
    assert client.post("/api/models/0123456789ab/slices", headers=JSON).status_code == 404
    assert client.post(f"/api/models/{trained}/slices").status_code == 415
