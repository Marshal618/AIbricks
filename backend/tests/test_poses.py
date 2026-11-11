"""Sort poses and hand signs: MediaPipe finds the points, and a small network learns the categories."""
import base64
import hashlib
import importlib.util
import io
import json
import math
import re
import time
import urllib.request

import numpy as np
import onnxruntime
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageEnhance
from test_text_models import CACHE
from test_training import BINARY, workspace  # noqa: F401

from aibricks import poses, webapp
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}

# MediaPipe's own sample photos (Apache 2.0), downloaded once into the test cache.
SAMPLES = {
    "thumbs_up.jpg": ("gesture_recognizer", "2aee0e3a69ba5f0d3287597e61d265f4f3ac2a44ccec198dddd2639b0c8ef7ba"),
    "thumbs_down.jpg": ("gesture_recognizer", "080b589bf3b91ba10cc6c03645be3b5b491a8ca8c8f7d65b5f32c563ae266af9"),
    "victory.jpg": ("gesture_recognizer", "6ac265f3ace6a6c4ac4a9079b63fcce4ab6517272afb1e430857f55ef324fde6"),
    "pointing_up.jpg": ("gesture_recognizer", "f4a701316b63dd8fa56e622f2b3042766369ccc189f0d89513f803cd985b993b"),
}
SIGNS = {"Up": "thumbs_up.jpg", "Down": "thumbs_down.jpg", "Victory": "victory.jpg", "Point": "pointing_up.jpg"}


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


def sample(name):
    folder = CACHE / "pose-samples"
    path = folder / name
    task, sha = SAMPLES[name]
    if not path.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        import ssl

        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urllib.request.urlopen(f"https://storage.googleapis.com/mediapipe-tasks/{task}/{name}", timeout=60, context=context) as reply:
                data = reply.read()
        except OSError:
            pytest.skip("The sample photos couldn't be downloaded.")
        path.write_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    return Image.open(path).convert("RGB")


def variant(image, seed):
    """The same hand sign in a slightly different photo: framed, tilted and lit differently."""
    rng = np.random.default_rng(seed)
    width, height = image.size
    share = rng.uniform(0.82, 0.97)
    left, top = rng.uniform(0, 1 - share) * width, rng.uniform(0, 1 - share) * height
    image = image.crop((int(left), int(top), int(left + share * width), int(top + share * height)))
    image = image.rotate(float(rng.uniform(-10, 10)), fillcolor=(128, 128, 128))
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.75, 1.25)))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def create(points="hands", categories=tuple(SIGNS)):
    reply = client.post("/api/datasets", json={"name": "Hand signs", "kind": "image", "goal": "pose", "points": points, "categories": list(categories)})
    assert reply.status_code == 200, reply.text
    return reply.json()


def add(dataset, category, data):
    reply = client.post("/api/dataset/images", params={"dataset": dataset, "category": category}, content=data, headers=BINARY)
    assert reply.status_code == 200, reply.text


def wait(limit=600):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": "pose"}).json()["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.5)
    raise AssertionError("training did not finish")


def model_module(points, tmp_path):
    """The generated model.py, imported on its own, as a downloaded project would use it."""
    folder = tmp_path / f"project-{points}"
    folder.mkdir()
    for name, text in poses.project_files("Test", points).items():
        (folder / name).write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"pose_model_{points}", folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_points_dont_depend_on_where_or_how_big(tmp_path):
    model = model_module("body", tmp_path)
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.2, 0.8, (33, 3)).astype(np.float32)
    seen = np.ones(33, dtype=np.float32)
    seen[25:] = 0  # legs out of view, as in most webcam photos
    moved = body_features(model, xyz * 0.5 + 0.1, seen)
    assert np.allclose(body_features(model, xyz, seen), moved, atol=1e-5)
    assert body_features(model, xyz, seen).shape == (model.FEATURES,)
    assert not body_features(model, xyz, seen)[25 * 3:33 * 3].any()  # unseen points carry nothing

    hands = model_module("hands", tmp_path)
    hand = rng.uniform(0.3, 0.6, (21, 3)).astype(np.float32)
    one = hands.hand_features({"Right": hand})
    assert one.shape == (hands.FEATURES,) and one[63] == 0 and one[-1] == 1  # no left hand, a right one
    assert np.allclose(one, hands.hand_features({"Right": hand * 2 + 0.3}), atol=1e-5)


def body_features(model, xyz, seen):
    return model.body_features(xyz.copy(), seen)


def test_pose_datasets_are_photo_datasets(workspace):  # noqa: F811
    assert client.post("/api/datasets", json={"name": "Words", "kind": "text", "goal": "pose"}).status_code == 422
    assert client.post("/api/datasets", json={"name": "Feet", "kind": "image", "goal": "pose", "points": "feet"}).status_code == 422
    dataset = create("body", ["Standing", "Arms up"])
    assert dataset["goal"] == "pose" and dataset["points"] == "body"
    assert client.post(f"/api/datasets/{dataset['id']}/train-pose", headers=JSON).status_code == 422
    assert client.post(f"/api/datasets/{dataset['id']}/experiments", json={"minutes": 5}).status_code == 422
    assert client.post(f"/api/datasets/{dataset['id']}/health/check", headers=JSON).status_code == 422
    sort = client.post("/api/datasets", json={"name": "Cats", "kind": "image", "categories": ["A", "B"]}).json()
    assert client.post(f"/api/datasets/{sort['id']}/train-pose", headers=JSON).status_code == 422


def test_it_learns_hand_signs(workspace, tmp_path):  # noqa: F811
    images = {category: sample(name) for category, name in SIGNS.items()}
    dataset = create()["id"]
    for category, image in images.items():
        for seed in range(10):
            add(dataset, category, variant(image, seed))
    # A photo with no hands teaches nothing, and it's said so rather than guessed.
    blank = io.BytesIO()
    Image.new("RGB", (320, 240), (90, 120, 160)).save(blank, "PNG")
    add(dataset, "Up", blank.getvalue())

    started = client.post(f"/api/datasets/{dataset}/train-pose", headers=JSON)
    assert started.status_code == 200, started.text
    job = wait()
    assert job["status"] == "finished", job["error"]
    assert job["task"] == "pose" and job["trainCount"] + job["checkCount"] == 41
    metrics = job["metrics"]
    assert metrics["task"] == "pose" and metrics["points"] == "hands"
    assert metrics["nothingFoundTraining"] + metrics["nothingFoundChecking"] >= 1
    assert job["review"]["correct"] >= 0.75 * job["review"]["total"], job["review"]

    for category, image in images.items():
        reply = client.post(f"/api/train/{job['id']}/predict-live", content=variant(image, 500), headers=BINARY)
        assert reply.status_code == 200, reply.text
        answer = reply.json()
        assert answer["found"] is True and len(answer["points"]) == 1 and len(answer["points"][0]) == 21
        assert answer["answers"][0]["category"] == category, (category, answer["answers"])
    nothing = client.post(f"/api/train/{job['id']}/predict-live", content=blank.getvalue(), headers=BINARY).json()
    assert nothing == {"found": False, "answers": [], "points": []}

    # Unsorted photos get suggestions from the pose model; a photo with no hands gets none.
    client.post("/api/label/images", params={"dataset": dataset}, content=variant(images["Victory"], 700), headers=BINARY)
    empty = io.BytesIO()
    Image.new("RGB", (320, 240), (200, 180, 150)).save(empty, "PNG")
    client.post("/api/label/images", params={"dataset": dataset}, content=empty.getvalue(), headers=BINARY)
    assert client.post("/api/label/suggest", params={"dataset": dataset}, headers=JSON).status_code == 200
    deadline = time.time() + 120
    while client.get("/api/label/images", params={"dataset": dataset}).json()["suggesting"] and time.time() < deadline:
        time.sleep(0.3)
    queue = client.get("/api/label/images", params={"dataset": dataset}).json()
    assert queue["suggestError"] is None and queue["unsuggested"] == 0 and queue["modelCategories"] == sorted(SIGNS)
    suggestions = [item["suggestion"] for item in queue["items"]]
    assert suggestions[0]["category"] == "Victory" and suggestions[1] is None

    card = client.get(f"/api/models/{job['id']}/card")
    assert card.status_code == 200 and "MediaPipe" in card.text
    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200
    import zipfile
    names = zipfile.ZipFile(io.BytesIO(download.content)).namelist()
    assert any(name.endswith("model/points.task") for name in names) and any(name.endswith("model/weights.pt") for name in names)

    # The web app carries both models, and its network answers exactly as PyTorch did.
    page = client.get(f"/api/train/{job['id']}/app")
    assert page.status_code == 200, page.text
    text = page.text
    assert f'integrity="{webapp.VISION_INTEGRITY}"' in text and "<title>Hand signs</title>" in text
    settings = json.loads(re.search(r'<script id="settings" type="application/json">(.*?)</script>', text, re.S).group(1))
    assert settings["kind"] == "pose" and settings["points"] == "hands" and settings["labels"] == sorted(SIGNS) and settings["about"].startswith("Sorts hand signs")
    finder = base64.b64decode(re.search(r'<script id="finder" type="application/octet-stream">(.*?)</script>', text, re.S).group(1))
    assert hashlib.sha256(finder).hexdigest() == "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
    network = onnxruntime.InferenceSession(base64.b64decode(re.search(r'<script id="model" type="application/octet-stream">(.*?)</script>', text, re.S).group(1)))
    pattern = np.array([[math.sin(i * 0.37) for i in range(settings["features"])]], dtype=np.float32)
    assert np.allclose(network.run(None, {"features": pattern})[0][0], settings["check"], atol=1e-4)
    assert len(settings["sample"]["features"]) == settings["features"]

    # The Raspberry Pi kit carries both models too, measures points the same way and names the signs.
    import pi_kit
    folder, files = pi_kit.unpack(client.get(f"/api/train/{job['id']}/pi"), tmp_path)
    assert files == {"model.onnx", "points.task", "settings.json", "run.py", "setup.sh", "requirements.txt", "README.md"}
    assert "mediapipe" in (folder / "requirements.txt").read_text()
    code, out, err = pi_kit.run(folder, "--check")
    assert code == 0 and "✓" in out, err
    for category, image in images.items():
        path = tmp_path / f"{category}.png"
        path.write_bytes(variant(image, 500))
        assert pi_kit.answers(folder, path)[0]["name"] == category
    nothing = tmp_path / "blank.png"
    nothing.write_bytes(blank.getvalue())
    assert pi_kit.answers(folder, nothing) == []
    assert pi_kit.run(folder, "--photo", nothing)[1].strip().endswith("No hands in view.")
