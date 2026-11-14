"""Find things by name: words for each thing, and OWLv2 drawing the first boxes on a photo."""
import hashlib
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from test_training import BINARY, workspace  # noqa: F401

from aibricks import namefinder
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    from test_text_models import CACHE
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


def create(categories=("Ball",)):
    reply = client.post("/api/datasets", json={"name": "Things", "kind": "image", "goal": "find", "categories": list(categories)})
    assert reply.status_code == 200, reply.text
    return reply.json()["id"]


def add(dataset, data):
    reply = client.post("/api/label/images", params={"dataset": dataset}, content=data, headers=BINARY)
    assert reply.status_code == 200, reply.text
    return hashlib.sha256(data).hexdigest()[:32] + ".png"


def balls_photo():
    """A wide photo of a pale table with two orange balls on it, and where each ball is."""
    width, height = 480, 320
    rng = np.random.default_rng(3)
    base = (rng.normal(215, 8, (height, width, 1)) + rng.normal(0, 3, (height, width, 3))).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    truth = []
    for cx, cy, r in ((120, 170, 60), (350, 150, 50)):
        for step in range(r, 0, -1):  # shaded, so it looks round
            shade = 1 - 0.5 * step / r
            draw.ellipse((cx - step - r * 0.2 * (1 - step / r), cy - step - r * 0.2 * (1 - step / r), cx + step, cy + step),
                         fill=(int(255 * (0.75 + 0.25 * shade)), int(140 * (0.6 + 0.4 * shade)), int(30 * shade)))
        truth.append({"x": (cx - r) / width, "y": (cy - r) / height, "w": 2 * r / width, "h": 2 * r / height})
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue(), truth


def test_each_thing_starts_with_its_own_name_and_can_have_other_words(workspace):  # noqa: F811
    dataset = create(("Ball", "Cup"))
    reply = client.get(f"/api/datasets/{dataset}/find-by-name").json()
    assert reply["words"] == {"Ball": "Ball", "Cup": "Cup"} and reply["download"] == namefinder.DOWNLOAD
    reply = client.put(f"/api/datasets/{dataset}/find-by-name", json={"words": {"Cup": "  coffee   mug, cup "}})
    assert reply.status_code == 200, reply.text
    assert reply.json()["words"] == {"Ball": "Ball", "Cup": "coffee mug, cup"}
    # Clearing the words goes back to the thing's name.
    assert client.put(f"/api/datasets/{dataset}/find-by-name", json={"words": {"Ball": ""}}).json()["words"]["Ball"] == "Ball"
    assert client.put(f"/api/datasets/{dataset}/find-by-name", json={"words": {"Plate": "plate"}}).status_code == 422
    assert client.put(f"/api/datasets/{dataset}/find-by-name", json={"words": {"Cup": "x" * 81}}).status_code == 422

    # Words follow a renamed thing and go with a removed one.
    assert client.put(f"/api/datasets/{dataset}/categories", json={"old": "Cup", "new": "Mug"}).status_code == 200
    assert client.get(f"/api/datasets/{dataset}/find-by-name").json()["words"] == {"Ball": "Ball", "Mug": "coffee mug, cup"}
    assert client.delete(f"/api/datasets/{dataset}/categories", params={"name": "Mug"}).status_code == 200
    assert client.post(f"/api/datasets/{dataset}/categories", json={"names": ["Mug"]}).status_code == 200
    assert client.get(f"/api/datasets/{dataset}/find-by-name").json()["words"]["Mug"] == "Mug"

    sorted_photos = client.post("/api/datasets", json={"name": "Sorted", "kind": "image", "categories": ["A", "B"]}).json()["id"]
    assert client.get(f"/api/datasets/{sorted_photos}/find-by-name").status_code == 422


def test_words_split_at_commas_and_overlapping_boxes_are_tidied():
    assert namefinder.searches({"Cup": "mug, cup,, mug", "Ball": "ball"}) == [("Cup", "mug"), ("Cup", "cup"), ("Ball", "ball")]
    with pytest.raises(namefinder.TrainingError):
        namefinder.searches({"A": ",".join(f"thing {n}" for n in range(namefinder.MAX_SEARCHES + 1))})

    searched = [("Cup", "mug"), ("Cup", "cup"), ("Ball", "ball")]
    # A 200 by 100 photo, padded by the model to a 200 by 200 square.
    rows = [[10, 10, 90, 90], [12, 12, 92, 88], [11, 9, 90, 91], [120, 20, 260, 140], [0, 0, 0.5, 0.5], [50, 50, 60, 60]]
    boxes = namefinder.to_boxes(rows, [0, 1, 2, 2, 0, 0], [0.6, 0.5, 0.3, 0.4, 0.9, 0.01], searched, 200, 100)
    # The two cups are one find, the ball in the same place goes to the surer cup, the tiny box and
    # the unsure one are dropped, and the box that runs off the photo stops at its edge.
    assert [box["category"] for box in boxes] == ["Cup", "Ball"]
    assert boxes[0] == {"category": "Cup", "chance": 0.6, "x": 0.05, "y": 0.1, "w": 0.4, "h": 0.8}
    assert boxes[1]["x"] == 0.6 and boxes[1]["w"] == 0.4 and boxes[1]["h"] == 1.0 - 0.2


def test_it_asks_for_the_download_first(workspace, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AI_BRICKS_CACHE", str(tmp_path / "empty-cache"))
    dataset = create()
    image_id = add(dataset, balls_photo()[0])
    assert client.get("/api/find-by-name").json()["downloaded"] is False
    reply = client.post(f"/api/datasets/{dataset}/find-by-name/{image_id}", headers=JSON)
    assert reply.status_code == 409 and "Download" in reply.json()["detail"]["message"]
    assert client.post("/api/find-by-name/download").status_code == 415  # needs the JSON header


@pytest.mark.skipif(not namefinder.downloaded(), reason="OWLv2 isn't in the test cache")
def test_it_draws_boxes_around_things_from_their_names(workspace):  # noqa: F811
    assert client.get("/api/find-by-name").json()["downloaded"] is True
    dataset = create(("Ball",))
    data, truth = balls_photo()
    image_id = add(dataset, data)
    reply = client.post(f"/api/datasets/{dataset}/find-by-name/{image_id}", headers=JSON)
    assert reply.status_code == 200, reply.text
    boxes = reply.json()["boxes"]
    sure = [box for box in boxes if box["chance"] >= namefinder.DEFAULT_SURE]
    assert len(sure) == 2, boxes
    for real in truth:
        assert max(namefinder._overlap(real, box) for box in sure) > 0.6
    assert all(box["category"] == "Ball" for box in boxes)
    assert boxes == sorted(boxes, key=lambda box: -box["chance"])
    # The suggestions can be saved as they are, as the labeller does.
    saved = client.put(f"/api/datasets/{dataset}/boxes/{image_id}", json={"boxes": [{key: box[key] for key in "xywh"} | {"category": box["category"]} for box in sure]})
    assert saved.status_code == 200 and len(saved.json()["boxes"]) == 2
    assert client.post(f"/api/datasets/{dataset}/find-by-name/nope.png", headers=JSON).status_code == 404
