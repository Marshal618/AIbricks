"""Showing where a model looked: Grad-CAM heatmaps for photos, word weights for text."""
import base64
import io
import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_training import BINARY, RED, build, fill, mark_tested, wait_for_end, workspace  # noqa: F401

from aibricks import explain, training
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
    yield job["id"]
    training.close_live()


def half_red(red_side="left", other=(0, 0, 0)):
    """Black holds no red or blue, so the red half is the only evidence in the photo."""
    image = Image.new("RGB", (40, 40), other)
    image.paste(RED, (0, 0, 20, 40) if red_side == "left" else (20, 0, 40, 40))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def side_heat(heatmap):
    width, values = heatmap["width"], heatmap["values"]
    rows = [values[row * width:(row + 1) * width] for row in range(heatmap["height"])]
    left = sum(sum(row[: width // 2]) for row in rows)
    right = sum(sum(row[width - width // 2:]) for row in rows)
    return left, right


def test_a_photo_heatmap_points_at_the_red_half(trained):
    for side in ("left", "right"):
        reply = client.post(f"/api/train/{trained}/explain", content=half_red(side), headers=BINARY)
        assert reply.status_code == 200, reply.text
        body = reply.json()
        heatmap = body["heatmap"]
        assert heatmap["method"] == "gradcam" and heatmap["category"] == body["answers"][0]["category"] == "Red"
        assert len(heatmap["values"]) == heatmap["width"] * heatmap["height"] and max(heatmap["values"]) == 1.0
        assert all(0 <= value <= 1 for value in heatmap["values"])
        left, right = side_heat(heatmap)
        assert (left > right) if side == "left" else (right > left), (side, left, right)


def test_live_frames_carry_a_heatmap_only_when_asked(trained):
    plain = client.post(f"/api/train/{trained}/predict-live", content=half_red(), headers=BINARY).json()
    assert set(plain) == {"answers"}
    explained = client.post(f"/api/train/{trained}/predict-live", params={"explain": "true"}, content=half_red(), headers=BINARY).json()
    assert explained["heatmap"]["width"] > 1
    normal = client.post(f"/api/train/{trained}/predict", content=half_red(), headers=BINARY).json()["answers"]
    assert explained["answers"][0]["chance"] == pytest.approx(normal[0]["chance"], abs=1e-4)


def test_explaining_refuses_bad_requests(trained):
    assert client.post(f"/api/train/{trained}/explain", content=b"nope", headers=BINARY).status_code == 422
    assert client.post("/api/train/0123456789ab/explain", content=half_red(), headers=BINARY).status_code == 404
    assert client.post("/api/models/0123456789ab/explain", json={"text": "hi"}).status_code == 404
    assert client.post(f"/api/models/{trained}/explain", json={"text": "hi"}).status_code == 404  # a photo model


# The runner works on any architecture, so it's checked on stand-in projects too.

_DATA = """
from PIL import Image
from torchvision import transforms
def _image_transform():
    return transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])
def _load_image(path, transform):
    return transform(Image.open(path).convert("RGB"))
"""
_PREDICT = "from model import build_model\ndef load_trained_model():\n    return build_model().eval()\n"
_MODELS = {
    "resnet": (64, "gradcam", "from torchvision.models import resnet18\nimport torch.nn as nn\nCLASS_NAMES = ['A', 'B', 'C']\n"
               "def build_model():\n    model = resnet18(weights=None)\n    model.fc = nn.Linear(512, 3)\n    return model\n"),
    "no image layers": (16, "gradients", "import torch.nn as nn\nCLASS_NAMES = ['A', 'B']\n"
                        "def build_model():\n    return nn.Sequential(nn.Flatten(), nn.Linear(768, 16), nn.ReLU(), nn.Linear(16, 2))\n"),
    "vision transformer": (224, "gradients", "from torchvision.models import vit_b_16\nimport torch.nn as nn\nCLASS_NAMES = ['A', 'B']\n"
                           "def build_model():\n    model = vit_b_16(weights=None)\n    model.heads = nn.Linear(768, 2)\n    return model\n"),
}


@pytest.mark.parametrize("name", list(_MODELS))
def test_every_kind_of_photo_model_gets_a_heatmap(tmp_path, name):
    size, method, code = _MODELS[name]
    (tmp_path / "model.py").write_text(code)
    (tmp_path / "data.py").write_text(_DATA.replace("SIZE", str(size)))
    (tmp_path / "predict.py").write_text(_PREDICT)
    process = subprocess.Popen([sys.executable, "-I", "-u", "-c", explain.LIVE_RUNNER], cwd=tmp_path, text=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdout.readline().strip() == "ready", process.stderr.read()
        frame = base64.b64encode(half_red()).decode()
        for explain_it in (True, False, True):
            process.stdin.write(json.dumps({"image": frame, "explain": explain_it}) + "\n")
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())
            assert "error" not in reply, reply
            assert abs(sum(answer["chance"] for answer in reply["answers"]) - 1) < 1e-4
            if explain_it:
                assert reply["heatmap"]["method"] == method
                assert len(reply["heatmap"]["values"]) == reply["heatmap"]["width"] * reply["heatmap"]["height"]
            else:
                assert reply["heatmap"] is None
    finally:
        process.kill()


# ------------------------------------------------------------------------------- text

def test_text_explanations_weigh_each_word(workspace):  # noqa: F811
    from test_text_models import fill as fill_text, train, wait

    train(fill_text())
    job = wait()
    assert job["status"] == "finished", job["error"]

    text = "the weather was wonderful and lovely"
    reply = client.post(f"/api/models/{job['id']}/explain", json={"text": text})
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["category"] == body["answers"][0]["category"] == "Good"
    words = [text[row["start"]:row["end"]] for row in body["words"]]
    assert words == text.split() and body["wordCount"] == 6
    # Removing the warm words lowers its confidence more than removing the plain ones.
    weights = {word: row["weight"] for word, row in zip(words, body["words"])}
    assert max(weights["wonderful"], weights["lovely"]) > max(weights["the"], weights["was"])

    long = " ".join(["word"] * (explain.MAX_WORDS + 20))
    body = client.post(f"/api/models/{job['id']}/explain", json={"text": long}).json()
    assert len(body["words"]) == explain.MAX_WORDS and body["wordCount"] == explain.MAX_WORDS + 20
    assert client.post(f"/api/models/{job['id']}/explain", json={"text": ""}).status_code == 422
    assert client.post(f"/api/train/{job['id']}/explain", content=half_red(), headers=BINARY).status_code == 422
