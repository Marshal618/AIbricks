"""A writer: a small open language model adjusted to a dataset's text, then asked to write and chat."""
import io
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_training import BINARY, workspace  # noqa: F401

from aibricks import training, writer
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}
CACHE = Path(__file__).resolve().parents[1] / ".test-cache"
BREADS = ["sourdough", "rye loaf", "cinnamon scroll", "baguette", "olive focaccia", "seeded roll"]


@pytest.fixture(autouse=True)
def shared_cache(monkeypatch):
    # SmolLM2 135M (270 MB) is downloaded once and shared, rather than once per test.
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


def notes(count, seed=1):
    rng = random.Random(seed)
    return [f"Baker's note, day {day + 1}: Today the {rng.choice(BREADS)} came out golden. We sold {rng.randint(10, 60)} by noon, "
            f"and the {rng.choice(BREADS)} ran out first, so tomorrow we bake more." for day in range(count)]


def create(name="Bakery notes", **extra):
    return client.post("/api/datasets", json={"name": name, "kind": "text", "goal": "write"} | extra)


def add(dataset, lines):
    reply = client.post("/api/label/text", params={"dataset": dataset, "kind": "lines"}, content="\n".join(lines).encode(), headers=BINARY)
    assert reply.status_code == 200, reply.text
    return reply.json()


def wait(limit=600):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": "writer"}).json()["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.5)
    raise AssertionError("training did not finish")


def test_a_writing_dataset_holds_text_without_categories(workspace):  # noqa: F811
    assert create(categories=["A", "B"]).status_code == 422
    assert client.post("/api/datasets", json={"name": "x", "kind": "image", "goal": "write"}).status_code == 422
    dataset = create().json()["id"]
    assert client.post(f"/api/datasets/{dataset}/categories", json={"names": ["Good"]}).status_code == 422
    add(dataset, notes(20))
    found = client.get(f"/api/datasets/{dataset}/writer/analysis").json()
    assert found["pieces"] == 20 and found["words"] > 400 and "at least 1,500 words" in found["problems"][0]
    assert [model["name"] for model in found["models"]] == ["SmolLM2 135M", "SmolLM2 360M", "Granite 4.0 350M", "Qwen2.5 0.5B"]
    assert client.post(f"/api/datasets/{dataset}/train-writer", json={}, headers=JSON).status_code == 422
    summary = client.get(f"/api/datasets/{dataset}").json()
    assert summary["labelled"] == 20 and summary["words"] == found["words"]

    # A long paragraph is cut at sentence ends rather than skipped.
    reply = add(dataset, [" ".join(["This is one long sentence about bread."] * 300)])
    assert reply["added"] >= 2 and reply["tooLong"] == 0


def test_it_learns_your_text_then_writes_and_chats(workspace, tmp_path):  # noqa: F811
    dataset = create().json()["id"]
    add(dataset, notes(160))
    started = client.post(f"/api/datasets/{dataset}/train-writer", json={"length": "quick"}, headers=JSON)
    assert started.status_code == 200, started.text
    job = wait()
    assert job["status"] == "finished", job["error"]
    metrics = job["metrics"]
    assert metrics["task"] == "writer" and metrics["model"] == "SmolLM2 135M" and metrics["adjustLayers"] == 2
    # Its guesses about held-back text get better than the downloaded writer's.
    assert metrics["lossAfter"] < metrics["lossBefore"] and metrics["accuracyAfter"] > metrics["accuracyBefore"], metrics
    assert metrics["samples"] and all(sample["before"] and sample["after"] for sample in metrics["samples"])
    assert job["best"]["accuracy"] == pytest.approx(metrics["accuracyAfter"], abs=0.001)
    assert job["modelBytes"] < 40_000_000 if "modelBytes" in job else True

    try:
        reply = client.post(f"/api/models/{job['id']}/write", json={"prompt": "Baker's note, day 200:", "chat": False, "compare": True}, headers=JSON)
        assert reply.status_code == 200, reply.text
        answer = reply.json()
        assert answer["adjusted"].startswith("Baker's note, day 200:") and answer["original"].startswith("Baker's note")
        chat = client.post(f"/api/models/{job['id']}/write", headers=JSON, json={
            "prompt": "And what should I bake tomorrow?", "chat": True,
            "history": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello! How can I help?"}],
        })
        assert chat.status_code == 200, chat.text
        assert chat.json()["adjusted"] and "original" not in chat.json()
        assert client.post(f"/api/models/{job['id']}/write", json={"prompt": " "}, headers=JSON).status_code == 422
    finally:
        with training._live_lock:
            if training._live is not None:
                training._live.close()
                training._live = None

    assert client.get(f"/api/models/{job['id']}/writer").json()["config"]["adjust_layers"] == 2
    card = client.get(f"/api/models/{job['id']}/card").json()["markdown"]
    assert "SmolLM2 135M" in card and "held-back" in card

    # The download writes the same way, offline.
    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        archive.extractall(tmp_path)
    folder = next(path.parent for path in tmp_path.rglob("chat.py"))
    env = {"PATH": "/usr/bin:/bin", "HF_HOME": str(CACHE / "huggingface"), "HF_HUB_OFFLINE": "1", "HOME": str(tmp_path)}
    done = subprocess.run([sys.executable, "chat.py", "--chat", "Say hello."], cwd=folder, capture_output=True, text=True, timeout=180, env=env)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip()
