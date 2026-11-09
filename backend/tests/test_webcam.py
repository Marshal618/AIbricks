"""Webcam support: photos captured straight into a category, and quick answers for live frames."""
import time

import pytest
from fastapi.testclient import TestClient
from test_training import BINARY, BLUE, RED, build, fill, mark_tested, picture, wait_for_end, workspace  # noqa: F401

from aibricks import training
from aibricks.main import app

client = TestClient(app)


@pytest.fixture
def trained(workspace):
    fill()
    graph = build()
    mark_tested(graph)
    assert client.post("/api/train", json=graph).status_code == 200
    job = wait_for_end()
    assert job["status"] == "finished"
    yield job["id"]
    training.close_live()


def live(job_id, data):
    return client.post(f"/api/train/{job_id}/predict-live", content=data, headers=BINARY)


# ------------------------------------------------------------------ capturing photos

def capture(dataset, category, data):
    return client.post("/api/dataset/images", params={"dataset": dataset, "category": category}, content=data, headers=BINARY)


def test_captured_photos_go_straight_into_a_category(workspace):
    dataset = client.post("/api/datasets", json={"name": "Cam", "kind": "image", "categories": ["Cup", "Pen"]}).json()["id"]
    for seed in range(3):
        reply = capture(dataset, "Cup", picture(RED, seed, "JPEG"))
        assert reply.status_code == 200 and reply.json()["count"] == seed + 1
    assert client.get(f"/api/datasets/{dataset}").json()["counts"] == {"Cup": 3, "Pen": 0}


def test_captures_need_a_photo_dataset(workspace):
    words = client.post("/api/datasets", json={"name": "Words", "kind": "text", "categories": ["Cup"]}).json()["id"]
    reply = capture(words, "Cup", picture(RED))
    assert reply.status_code == 422 and "can't hold photos" in reply.json()["detail"]["message"]
    table = client.post("/api/datasets", json={"name": "Rows", "kind": "table"}).json()["id"]
    assert capture(table, "Cup", picture(RED)).status_code == 422


# -------------------------------------------------------------------- live answers

def test_live_answers_match_normal_answers_and_come_back_quickly(trained):
    frame = picture(RED, 500, "JPEG")
    normal = client.post(f"/api/train/{trained}/predict", content=frame, headers=BINARY).json()["answers"]
    first = live(trained, frame)
    assert first.status_code == 200, first.text
    answers = first.json()["answers"]
    assert [a["category"] for a in answers] == [a["category"] for a in normal]
    assert answers[0]["chance"] == pytest.approx(normal[0]["chance"], abs=1e-4)

    # Once loaded, each frame skips starting Python and loading the model.
    started = time.perf_counter()
    for seed in range(5):
        assert live(trained, picture(BLUE if seed % 2 else RED, 600 + seed, "JPEG")).status_code == 200
    assert (time.perf_counter() - started) / 5 < 0.5


def test_one_live_model_is_kept_at_a_time(trained):
    assert live(trained, picture(RED, 1)).status_code == 200
    process = training._live.process
    assert live(trained, picture(BLUE, 1)).status_code == 200
    assert training._live.process is process  # reused, not restarted


def test_bad_frames_and_unknown_models_are_refused(trained):
    assert live(trained, b"not a picture").status_code == 422
    assert live("0123456789ab", picture(RED)).status_code == 404
    assert client.post(f"/api/train/{trained}/predict-live", content=picture(RED), headers={"Content-Type": "image/png"}).status_code == 415
    # The live model still works after a bad frame.
    assert live(trained, picture(RED, 2)).status_code == 200


def test_deleting_a_model_stops_its_live_process(trained):
    assert live(trained, picture(RED, 3)).status_code == 200
    process = training._live.process
    assert client.delete(f"/api/models/{trained}").status_code == 200
    assert training._live is None and process.poll() is not None


def test_an_idle_live_model_is_closed(trained, monkeypatch):
    monkeypatch.setattr(training, "LIVE_IDLE_SECONDS", 0)
    monkeypatch.setattr(training, "LIVE_CHECK_SECONDS", 0.1)
    assert live(trained, picture(RED, 4)).status_code == 200
    process = training._live.process
    deadline = time.time() + 10
    while training._live is not None and time.time() < deadline:
        time.sleep(0.1)
    assert training._live is None and process.poll() is not None
