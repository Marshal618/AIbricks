"""Web apps made from trained photo models: one page that runs the model by itself."""
import base64
import io
import json
import math
import re

import numpy as np
import onnxruntime
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_training import BINARY, BLUE, RED, build, fill, mark_tested, picture, wait_for_end, workspace  # noqa: F401

from aibricks import training, webapp
from aibricks.main import app

client = TestClient(app)


@pytest.fixture
def trained(workspace):  # noqa: F811
    fill()
    graph = build()
    mark_tested(graph)
    assert client.post("/api/train", json=graph).status_code == 200
    job = wait_for_end()
    assert job["status"] == "finished"
    return job["id"]


def parts(page):
    settings = json.loads(re.search(r'<script id="settings" type="application/json">(.*?)</script>', page, re.S).group(1))
    model = base64.b64decode(re.search(r'<script id="model" type="application/octet-stream">(.*?)</script>', page, re.S).group(1))
    return settings, onnxruntime.InferenceSession(model)


def softmax(values):
    top = max(values)
    exps = [math.exp(value - top) for value in values]
    return [value / sum(exps) for value in exps]


def test_the_page_runs_the_same_model(trained):
    reply = client.get(f"/api/train/{trained}/app")
    assert reply.status_code == 200 and reply.headers["content-type"].startswith("text/html")
    assert "script-src 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net" in reply.headers["content-security-policy"]
    page = reply.text
    assert f'integrity="{webapp.RUNTIME_INTEGRITY}"' in page and "<title>Colours</title>" in page
    settings, session = parts(page)
    assert settings["labels"] == ["Red", "Blue"] and settings["size"] == 32 and settings["channels"] == 3 and settings["mean"] is None

    # The page's self-check pattern gives PyTorch's answer.
    count = 3 * 32 * 32
    pattern = np.array([math.sin(i * 0.37) for i in range(count)], np.float32).reshape(1, 3, 32, 32)
    got = session.run(None, {"image": pattern})[0][0]
    assert np.allclose(got, settings["check"], atol=1e-4)

    # A real photo, prepared as the page does, gets the chances the app itself gives.
    for colour, seed in ((RED, 400), (BLUE, 401)):
        data = picture(colour, seed)
        image = Image.open(io.BytesIO(data)).convert("RGB").resize((32, 32), Image.BILINEAR)
        tensor = (np.asarray(image, np.float32) / 255).transpose(2, 0, 1)[None]
        chances = dict(zip(settings["labels"], softmax(session.run(None, {"image": tensor})[0][0].tolist())))
        expected = client.post(f"/api/train/{trained}/predict", content=data, headers=BINARY).json()["answers"]
        for answer in expected:
            assert chances[answer["category"]] == pytest.approx(answer["chance"], abs=1e-4)


def test_downloads_are_named_and_the_export_is_reused(trained):
    first = client.get(f"/api/train/{trained}/app", params={"download": "true"})
    assert first.headers["content-disposition"] == 'attachment; filename="colours-app.html"'
    exported = training._folder(training.stored(trained)) / webapp.APP_FILE
    made = exported.stat().st_mtime_ns
    assert client.get(f"/api/train/{trained}/app").status_code == 200
    assert exported.stat().st_mtime_ns == made


def test_only_finished_photo_models_become_apps(trained, monkeypatch):
    assert client.get("/api/train/0123456789ab/app").status_code == 404
    assert client.get("/api/train/nope/app").status_code == 404
    job = training.stored(trained)
    monkeypatch.setattr(job, "task", "text")
    monkeypatch.setattr(training, "stored", lambda job_id: job)
    reply = client.get(f"/api/train/{trained}/app")
    assert reply.status_code == 422 and "photo models" in reply.json()["detail"]["message"]


def test_names_cant_break_out_of_the_page(trained):
    job = training.stored(trained)
    job.projectName = "</script><script>alert(1)</script>"
    training._save(job)
    page = client.get(f"/api/train/{trained}/app").text
    assert "<script>alert(1)" not in page
    settings, _ = parts(page)
    assert settings["name"] == job.projectName
