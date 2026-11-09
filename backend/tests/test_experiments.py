"""Find a better model: experiments that compare one change at a time, and the training check."""
import time

import pytest
from fastapi.testclient import TestClient
from test_training import BINARY, BLUE, RED, picture, workspace  # noqa: F401

from aibricks import checkup, codegen, experiments, training
from aibricks.graph import GraphIn, analyse
from aibricks.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    from test_text_models import CACHE
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))
    monkeypatch.setattr(experiments, "SHORT_EPOCHS", 2)
    monkeypatch.setattr(experiments, "LONG_EPOCHS", 3)
    monkeypatch.setattr(experiments, "SCRATCH_SIZE", 32)
    monkeypatch.setattr(experiments, "PRETRAINED_SIZE", 64)


def photos(per_category=15, name="Desk colours"):
    dataset = client.post("/api/datasets", json={"name": name, "kind": "image", "categories": ["Red", "Blue"]}).json()["id"]
    for seed in range(per_category):
        for category, colour in (("Red", RED), ("Blue", BLUE)):
            reply = client.post("/api/dataset/images", params={"dataset": dataset, "category": category}, content=picture(colour, seed), headers=BINARY)
            assert reply.status_code == 200
    return dataset


def wait(dataset, limit=600):
    deadline = time.time() + limit
    while time.time() < deadline:
        state = client.get(f"/api/datasets/{dataset}/experiments").json()["experiment"]
        if state and state["status"] != "running":
            return state
        time.sleep(0.5)
    raise AssertionError("experiment did not finish")


# ------------------------------------------------------------------ the options

@pytest.mark.parametrize("changes", [(), ("augment",), ("wider",), ("pretrained",), ("slower",), ("augment", "pretrained", "slower")])
def test_every_option_is_a_valid_build(changes):
    graph = experiments.graph_for("Desk colours - x", ["Red", "Blue"], changes, 3)
    payload = GraphIn(**graph)
    analysis = analyse(payload)
    assert analysis.ok, analysis.issues
    files = codegen.generate(payload, analysis)
    assert ("AUGMENT = True" in files["data.py"]) == ("augment" in changes)
    assert ("resnet18" in files["bricks.py"]) == ("pretrained" in changes)


def test_varied_photos_only_change_training_photos(workspace, tmp_path):  # noqa: F811
    graph = experiments.graph_for("Vary", ["Red", "Blue"], ("augment",), 1)
    payload = GraphIn(**graph)
    files = codegen.generate(payload, analyse(payload))
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    for category, colour in (("Red", RED), ("Blue", BLUE)):
        (tmp_path / "data" / "images" / category).mkdir(parents=True)
        for seed in range(10):
            (tmp_path / "data" / "images" / category / f"{seed}.png").write_bytes(picture(colour, seed))
    import subprocess
    import sys
    script = (
        "import torch, sys; sys.path.insert(0, '.')\n"
        "from data import load_datasets\n"
        "train, check = load_datasets()\n"
        "a = [train[0][0] for _ in range(4)]; b = [check[0][0] for _ in range(4)]\n"
        "print(any(not torch.equal(a[0], x) for x in a[1:]), all(torch.equal(b[0], x) for x in b[1:]))\n"
    )
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert done.stdout.split() == ["True", "True"], done.stderr[-500:]


# ------------------------------------------------------------------ experiments

def test_an_experiment_compares_changes_and_keeps_only_the_winner(workspace):  # noqa: F811
    dataset = photos()
    reply = client.post(f"/api/datasets/{dataset}/experiments", json={"minutes": 10})
    assert reply.status_code == 200, reply.text
    # Other training waits while it runs.
    from test_training import build, mark_tested
    graph = build()
    mark_tested(graph)
    blocked = client.post("/api/train", params={"dataset": dataset}, json=graph)
    assert blocked.status_code == 409
    assert client.post(f"/api/datasets/{dataset}/experiments", json={"minutes": 10}).status_code == 409

    state = wait(dataset)
    assert state["status"] == "finished", state["error"]
    assert [effect["key"] for effect in state["effects"]] == ["augment", "wider", "pretrained", "slower"]
    for effect in state["effects"]:
        assert effect["meaning"] in {"helped", "hurt", "no clear difference", "too few photos to tell"}
        assert effect["shared"] == 6  # the same held-back photos for every option
    first_round = [option for option in state["options"] if option["round"] == 1]
    assert len(first_round) == 5 and all(option["deleted"] for option in first_round)
    winner = next(option for option in state["options"] if option["round"] == 2 and option["key"] == state["winner"])
    assert winner["epochs"] == 3 and winner["graph"]["nodes"][0]["type"] == "image_folder"

    models = client.get(f"/api/datasets/{dataset}/models").json()["models"]
    assert [model["id"] for model in models] == [winner["jobId"]]
    # Its blocks are a normal build that can be opened and trained again.
    assert analyse(GraphIn(**winner["graph"])).ok


def test_stopping_leaves_nothing_behind(workspace):  # noqa: F811
    dataset = photos()
    assert client.post(f"/api/datasets/{dataset}/experiments", json={"minutes": 5}).status_code == 200
    time.sleep(1.5)
    assert client.delete(f"/api/datasets/{dataset}/experiments").status_code == 200
    state = wait(dataset)
    assert state["status"] == "stopped"
    assert client.get(f"/api/datasets/{dataset}/models").json()["models"] == []
    assert client.delete(f"/api/datasets/{dataset}/experiments").status_code == 404


def test_experiments_explain_what_they_need(workspace):  # noqa: F811
    small = photos(per_category=5, name="Few")
    assert "20 photos" in client.post(f"/api/datasets/{small}/experiments", json={"minutes": 5}).json()["detail"]["message"]
    assert client.post(f"/api/datasets/{small}/experiments", json={"minutes": 7}).status_code == 422
    words = client.post("/api/datasets", json={"name": "Words", "kind": "text", "categories": ["A", "B"]}).json()["id"]
    assert "photo datasets" in client.post(f"/api/datasets/{words}/experiments", json={"minutes": 5}).json()["detail"]["message"]
    reply = client.get(f"/api/datasets/{small}/experiments").json()
    assert reply["experiment"] is None and reply["budgets"] == [5, 10, 20, 30] and set(reply["changes"]) == {"augment", "wider", "pretrained", "slower"}


# ---------------------------------------------------------------- training check

def epochs(train, held, accuracy):
    return [{"epoch": i + 1, "trainLoss": t, "checkLoss": c, "accuracy": a} for i, (t, c, a) in enumerate(zip(train, held, accuracy))]


def kinds(findings):
    return [finding["kind"] for finding in findings]


def test_memorising_is_spotted():
    found = checkup.check(epochs([1.0, 0.6, 0.3, 0.1, 0.05], [0.9, 0.6, 0.55, 0.7, 0.9], [0.5, 0.7, 0.75, 0.72, 0.7]), 0.5)
    assert kinds(found)[0] == "memorising" and "epoch 3" in found[0]["detail"] and "augment" in found[0]["fixes"]


def test_an_unstable_run_suggests_learning_more_slowly():
    assert kinds(checkup.check(epochs([1.0, 0.5, 2.5, 0.9], [1.0, 0.8, 2.0, 1.0], [0.5, 0.6, 0.4, 0.5]), 0.5))[0] == "unstable"
    assert kinds(checkup.check(epochs([1.0, float("nan")], [1.0, float("nan")], [0.5, None]), 0.5)) == ["unstable"]


def test_a_model_near_guessing_is_too_simple():
    found = checkup.check(epochs([0.69, 0.68, 0.68], [0.69, 0.69, 0.69], [0.5, 0.52, 0.55]), 0.5)
    assert kinds(found)[0] == "too_simple" and "pretrained" in found[0]["fixes"]


def test_still_learning_and_healthy_runs():
    assert "still_learning" in kinds(checkup.check(epochs([1.0, 0.6, 0.4], [1.0, 0.7, 0.5], [0.5, 0.7, 0.8]), 0.5))
    assert kinds(checkup.check(epochs([1.0, 0.5, 0.3, 0.28], [1.0, 0.55, 0.4, 0.395], [0.5, 0.8, 0.9, 0.9]), 0.5)) == ["healthy"]
    assert checkup.check(epochs([1.0], [1.0], [0.5]), 0.5) == []


def test_finished_photo_runs_carry_the_check(workspace):  # noqa: F811
    from test_training import build, fill, mark_tested, wait_for_end
    fill()
    graph = build()
    mark_tested(graph)
    client.post("/api/train", json=graph)
    job = wait_for_end()
    assert job["status"] == "finished" and job["checkup"] and job["checkup"][0]["title"]


# ------------------------------------------------------------------ varying photos

def test_previews_show_varied_versions_of_a_real_photo(workspace):  # noqa: F811
    import base64
    import io
    from PIL import Image
    dataset = photos(per_category=3)
    reply = client.post("/api/augment/preview", json={"dataset": dataset, "size": 48, "channels": 3,
                                                     "params": {"vary_flip": True, "vary_crop": 40, "vary_rotate": 20, "vary_light": 30, "vary_auto": True}})
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert len(body["varied"]) == 8
    images = [Image.open(io.BytesIO(base64.b64decode(item.split(",", 1)[1]))) for item in [body["original"], *body["varied"]]]
    assert all(image.size == (48, 48) for image in images)
    grey = client.post("/api/augment/preview", json={"dataset": dataset, "size": 32, "channels": 1, "params": {}}).json()
    assert Image.open(io.BytesIO(base64.b64decode(grey["original"].split(",", 1)[1]))).size == (32, 32)
    words = client.post("/api/datasets", json={"name": "Words", "kind": "text", "categories": ["A"]}).json()["id"]
    assert client.post("/api/augment/preview", json={"dataset": words, "size": 32, "channels": 3, "params": {}}).status_code == 422
    empty = client.post("/api/datasets", json={"name": "Empty", "kind": "image", "categories": ["A"]}).json()["id"]
    assert "Add some photos" in client.post("/api/augment/preview", json={"dataset": empty, "size": 32, "channels": 3, "params": {}}).json()["detail"]["message"]


def test_vary_settings_reach_the_generated_code():
    graph = experiments.graph_for("Vary", ["Red", "Blue"], ("augment",), 1)
    graph["nodes"][0]["params"].update(vary_flip=False, vary_rotate=10, vary_auto=True)
    payload = GraphIn(**graph)
    data = codegen.generate(payload, analyse(payload))["data.py"]
    assert "VARY = {'flip': False, 'crop': 30, 'rotate': 10, 'light': 25, 'auto': True}" in data
    graph["nodes"][0]["params"]["vary_rotate"] = 90
    assert not analyse(GraphIn(**graph)).ok
