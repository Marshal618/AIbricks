"""Is this like what it learned from? Warnings for photos and text unlike anything a model trained on."""
import random
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from test_text_models import fill as fill_text
from test_text_models import train as train_text
from test_text_models import wait as wait_text
from test_training import BINARY, BLUE, RED, build, fill, mark_tested, picture, wait_for_end, workspace  # noqa: F401

from aibricks import familiar
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}
CACHE = Path(__file__).resolve().parents[1] / ".test-cache"


@pytest.fixture(autouse=True)
def shared_cache(monkeypatch):
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))
    familiar._builds.clear()


def prepare(job_id):
    reply = client.post(f"/api/models/{job_id}/familiar", headers=JSON)
    assert reply.status_code == 200, reply.text
    deadline = time.time() + 300
    while time.time() < deadline:
        state = client.get(f"/api/models/{job_id}/familiar").json()
        if state["status"] != "building":
            assert state["status"] == "ready", state["error"]
            return state
        time.sleep(0.2)
    raise AssertionError("the check was not prepared")


def test_false_alarms_stay_near_one_in_twenty():
    # Points scattered around one centre are "what it learned from". New points from the same place
    # should be flagged about 1 time in 20 at most; points from somewhere else nearly always.
    generator = torch.Generator().manual_seed(0)
    centre = F.normalize(torch.randn(1, 32, generator=generator), dim=1)

    def near(count, spread=0.6):
        return F.normalize(centre + spread * torch.randn(count, 32, generator=generator) / 32 ** 0.5, dim=1)

    flagged, elsewhere = [], []
    for _ in range(30):
        train, held = near(40), near(10)
        usual = familiar.usual_distances(train, held)
        assert len(usual) == 50 and usual == sorted(usual)
        new = familiar.distances(near(200), train)
        flagged += [familiar.p_value(gap, usual) <= familiar.LEVEL for gap in new.tolist()]
        far = familiar.distances(F.normalize(torch.randn(20, 32, generator=generator), dim=1), train)
        elsewhere += [familiar.p_value(gap, usual) <= familiar.LEVEL for gap in far.tolist()]
    assert sum(flagged) / len(flagged) <= 0.07
    assert sum(elsewhere) / len(elsewhere) >= 0.95


def test_p_values_count_the_new_example():
    assert familiar.p_value(0.5, [0.1, 0.2, 0.3]) == 0.25  # farther than all three: (1 + 0) / 4
    assert familiar.p_value(0.0, [0.1, 0.2, 0.3]) == 1.0
    known = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=1)
    # Leaving itself out, each point's distance is to another point, never zero.
    assert all(gap > 0 for gap in familiar.distances(known, known, skip_self=True).tolist())


def test_a_photo_model_warns_about_an_unlike_photo(workspace):  # noqa: F811
    fill()
    graph = build()
    mark_tested(graph)
    assert client.post("/api/train", json=graph).status_code == 200
    job = wait_for_end()
    assert job["status"] == "finished"
    assert client.get(f"/api/models/{job['id']}/familiar").json()["status"] == "missing"
    assert client.post(f"/api/models/{job['id']}/familiar/photo", content=picture(RED, 99), headers=BINARY).status_code == 409

    state = prepare(job["id"])
    assert state["known"] == 32 and state["held"] == 8 and state["usual"] == 40 and state["canWarn"]

    usual = client.post(f"/api/models/{job['id']}/familiar/photo", content=picture(RED, 99), headers=BINARY).json()
    assert usual["unfamiliar"] is False and usual["nearest"][0]["category"] == "Red"
    assert len(usual["nearest"]) == 3 and usual["nearest"][0]["similarity"] >= usual["nearest"][1]["similarity"]

    rng = random.Random(1)
    from io import BytesIO

    from PIL import Image
    stripes = Image.new("RGB", (64, 64))
    stripes.putdata([(255, 255, 0) if (x // 4 + y // 4) % 2 else (rng.randint(0, 80), 160, 40) for y in range(64) for x in range(64)])
    buffer = BytesIO()
    stripes.save(buffer, "PNG")
    odd = client.post(f"/api/models/{job['id']}/familiar/photo", content=buffer.getvalue(), headers=BINARY).json()
    assert odd["unfamiliar"] is True and odd["pValue"] <= 0.05 and odd["farther"] > 0.9

    # The closest training photo can be shown, but nothing outside the check can be reached.
    first = client.get(f"/api/models/{job['id']}/familiar/photos/{usual['nearest'][0]['index']}")
    assert first.status_code == 200 and first.headers["content-type"].startswith("image/")
    assert client.get(f"/api/models/{job['id']}/familiar/photos/999").status_code == 404
    assert client.post(f"/api/models/{job['id']}/familiar/text", json={"text": "hello"}).status_code == 422
    assert client.post(f"/api/models/{job['id']}/familiar/photo", content=b"not a photo", headers=BINARY).status_code == 422


def test_a_text_model_warns_about_off_topic_text(workspace):  # noqa: F811
    good = [f"what a {word} {thing}" for word in ("wonderful", "lovely", "happy", "brilliant", "great")
            for thing in ("day", "film", "meal", "trip")]
    bad = [f"what an {word} {thing}" for word in ("awful", "dreadful", "horrible", "miserable", "terrible")
           for thing in ("day", "film", "meal", "trip")]
    dataset = fill_text(good, bad)
    job = train_text(dataset)
    job = wait_text()
    assert job["status"] == "finished"
    state = prepare(job["id"])
    assert state["canWarn"] and state["known"] + state["held"] == 40

    usual = client.post(f"/api/models/{job['id']}/familiar/text", json={"text": "what a lovely meal"}).json()
    assert usual["unfamiliar"] is False and usual["nearest"][0]["text"] == "what a lovely meal"
    odd = client.post(f"/api/models/{job['id']}/familiar/text", json={"text": "The invoice number is 4471-B, due March 3rd."}).json()
    assert odd["unfamiliar"] is True
    assert client.post(f"/api/models/{job['id']}/familiar/photo", content=picture(RED, 1), headers=BINARY).status_code == 422


def test_only_finished_sorters_are_checked(workspace):  # noqa: F811
    assert client.get("/api/models/nothing/familiar").status_code == 404
    assert client.post("/api/models/nothing/familiar").status_code == 415


def test_a_saved_dataset_is_read_from_the_version_it_trained_on(workspace):  # noqa: F811
    from test_health import image_dataset
    dataset, _ = image_dataset(red=12, blue=12)
    graph = build()
    mark_tested(graph)
    assert client.post("/api/train", params={"dataset": dataset}, json=graph).status_code == 200
    job = wait_for_end()
    assert job["status"] == "finished" and job["datasetVersion"] is not None
    state = prepare(job["id"])
    assert state["known"] + state["held"] == 24
    nearest = client.post(f"/api/models/{job['id']}/familiar/photo", content=picture(RED, 5), headers=BINARY).json()["nearest"]
    assert client.get(f"/api/models/{job['id']}/familiar/photos/{nearest[0]['index']}").status_code == 200
