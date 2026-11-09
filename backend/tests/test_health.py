"""Data health levels, the label check and reviewing what it finds."""
import hashlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_training import BINARY, BLUE, RED, build, mark_tested, picture, workspace  # noqa: F401

from aibricks import health, training
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}
CACHE = Path(__file__).resolve().parents[1] / ".test-cache"


@pytest.fixture(autouse=True)
def shared_cache(monkeypatch):
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))
    health._job = None


def pattern(kind, seed):
    """Stripes or dots in random colours: varied pictures, as real photos are."""
    import io
    import random

    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    image = Image.new("RGB", (96, 96), tuple(rng.randint(0, 255) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    ink = tuple(rng.randint(0, 255) for _ in range(3))
    if kind == "Red":
        for y in range(rng.randint(0, 6), 96, 12):
            draw.rectangle([0, y, 96, y + 5], fill=ink)
    else:
        for _ in range(14):
            x, y, r = rng.randint(0, 96), rng.randint(0, 96), rng.randint(4, 9)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=ink)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def image_dataset(red=24, blue=24, wrong=0):
    """Red holds stripes and Blue holds dots. Planted mistakes are stripes filed under Blue."""
    dataset = client.post("/api/datasets", json={"name": "Patterns", "kind": "image", "categories": ["Red", "Blue"]}).json()["id"]
    planted = []
    for seed in range(red):
        data = pattern("Red", seed)
        category = "Blue" if seed < wrong else "Red"
        assert client.post("/api/dataset/images", params={"dataset": dataset, "category": category}, content=data, headers=BINARY).status_code == 200
        if seed < wrong:
            planted.append(next(path.name for path in training._category_folder("Blue", dataset).iterdir() if path.read_bytes() == data))
    for seed in range(blue):
        client.post("/api/dataset/images", params={"dataset": dataset, "category": "Blue"}, content=pattern("Blue", seed + 1000), headers=BINARY)
    return dataset, planted


def overview(dataset):
    reply = client.get(f"/api/datasets/{dataset}/health")
    assert reply.status_code == 200, reply.text
    return reply.json()


def check(dataset):
    reply = client.post(f"/api/datasets/{dataset}/health/check", headers=JSON)
    assert reply.status_code == 200, reply.text
    deadline = time.time() + 300
    while time.time() < deadline:
        job = overview(dataset)["job"]
        if job["status"] != "running":
            assert job["status"] == "finished", job["error"]
            return overview(dataset)
        time.sleep(0.2)
    raise AssertionError("the check did not finish")


def review(dataset, **body):
    reply = client.post(f"/api/datasets/{dataset}/health/review", json=body)
    assert reply.status_code == 200, reply.text
    return reply.json()


def test_levels_before_any_check():
    dataset = client.post("/api/datasets", json={"name": "Empty", "kind": "image", "categories": ["Red", "Blue"]}).json()["id"]
    assert overview(dataset)["level"] == 0 and overview(dataset)["levelName"] == "Not ready"
    small, _ = image_dataset(6, 6)
    started = overview(small)
    assert started["level"] == 1 and "Short: Red, Blue" in started["nextSteps"][0]
    solid, _ = image_dataset()
    ready = overview(solid)
    assert ready["level"] == 2 and ready["nextSteps"] == ["Run the label check."] and ready["check"] is None
    uneven, _ = image_dataset(24, 250)
    assert overview(uneven)["level"] == 1 and "Even out" in overview(uneven)["nextSteps"][0]


def test_the_same_photo_in_two_categories_is_a_conflict():
    dataset, _ = image_dataset()
    data = pattern("Red", 3)
    client.post("/api/dataset/images", params={"dataset": dataset, "category": "Blue"}, content=data, headers=BINARY)
    state = overview(dataset)
    assert state["level"] == 1 and len(state["conflicts"]) == 1 and state["conflicts"][0]["labels"] == ["Blue", "Red"]
    item = state["conflicts"][0]["ids"][0]
    after = review(dataset, action="drop", id=item, category="Blue")
    assert after["conflicts"] == [] and after["level"] == 2
    assert client.post(f"/api/datasets/{dataset}/health/review", json={"action": "drop", "id": item, "category": "Red"}).status_code == 422


def test_label_check_finds_planted_mistakes_and_review_reaches_the_top_level(workspace):  # noqa: F811
    dataset, planted = image_dataset(red=30, wrong=6)
    state = check(dataset)
    flagged = {issue["id"]: issue for issue in state["check"]["issues"]}
    # Real data showed 74-82% of planted mistakes caught, so this asks for most, not all.
    caught = [item for item in planted if item in flagged]
    assert len(caught) >= 3, f"caught {len(caught)} of {len(planted)}"
    assert all(flagged[item]["suggested"] == "Red" and flagged[item]["given"] == "Blue" for item in caught)
    assert state["check"]["openIssues"] == len(flagged) and state["check"]["unchecked"] == 0
    assert state["check"]["passedShare"] == pytest.approx((54 - len(flagged)) / 54)
    # The check keeps no copies of the pictures.
    assert not (workspace / "collections" / dataset / "health" / "work").exists()

    review(dataset, action="move", id=caught[0], category="Red")
    review(dataset, action="remove", id=caught[1])
    after = review(dataset, action="keep", id=caught[2], category="Blue")
    states = {issue["id"]: issue["state"] for issue in after["check"]["issues"]}
    assert states[caught[0]] == "fixed" and states[caught[1]] == "removed" and states[caught[2]] == "kept"
    assert (training._category_folder("Red", dataset) / caught[0]).exists()
    assert (training.removed_folder(dataset) / caught[1]).exists()
    for item, issue_state in states.items():
        if issue_state == "open":
            review(dataset, action="keep", id=item, category=flagged[item]["given"])
    final = overview(dataset)
    # Fixing flagged labels doesn't make the check out of date.
    assert final["check"]["unchecked"] == 0 and final["check"]["openIssues"] == 0
    assert final["level"] == 4 and final["levelName"] == "Reviewed" and final["nextSteps"] == []

    # New examples haven't been checked, so the level drops until the check runs again.
    client.post("/api/dataset/images", params={"dataset": dataset, "category": "Red"}, content=pattern("Red", 500), headers=BINARY)
    stale = overview(dataset)
    assert stale["level"] == 2 and stale["check"]["unchecked"] == 1 and "Run it again" in stale["nextSteps"][0]


def test_text_check_and_exact_conflicts():
    dataset = client.post("/api/datasets", json={"name": "Moods", "kind": "text", "categories": ["Good", "Bad"]}).json()["id"]
    good = [f"what a wonderful happy {word} day" for word in ("sunny", "bright", "warm", "lovely", "great", "fine", "kind", "sweet", "calm", "nice")] * 1
    bad = [f"what an awful miserable {word} day" for word in ("rainy", "dark", "cold", "gloomy", "grim", "bleak", "harsh", "sour", "wild", "nasty")]
    texts = good + bad + ["What a wonderful happy SUNNY  day"]
    client.post("/api/label/text", params={"dataset": dataset, "kind": "lines"}, content="\n".join(texts).encode(), headers=BINARY)
    ids = {row["text"]: row["id"] for row in client.get("/api/label/text", params={"dataset": dataset}).json()["items"]}
    for text, item in ids.items():
        label = "Bad" if text in bad or text == good[1] else "Good"
        if text == "What a wonderful happy SUNNY  day":
            label = "Bad"
        client.post(f"/api/label/text/{item}/assign", params={"dataset": dataset}, json={"category": label})
    state = overview(dataset)
    assert len(state["conflicts"]) == 1 and state["conflicts"][0]["labels"] == ["Bad", "Good"]
    checked = check(dataset)
    flagged = {issue["id"]: issue for issue in checked["check"]["issues"]}
    planted = hashlib.sha256(good[1].encode()).hexdigest()[:32]
    assert planted in flagged and flagged[planted]["text"] == good[1] and flagged[planted]["suggested"] == "Good"
    assert any(pair["state"] == "open" for pair in checked["check"]["duplicates"])
    pair = next(pair for pair in checked["check"]["duplicates"] if pair["state"] == "open")
    after = review(dataset, action="keep-pair", pair=[pair["first"], pair["second"]])
    assert next(row for row in after["check"]["duplicates"] if row["first"] == pair["first"])["state"] == "kept"


def test_spreadsheets_get_table_statistics():
    dataset = client.post("/api/datasets", json={"name": "T", "kind": "table"}).json()["id"]
    client.post(f"/api/datasets/{dataset}/table", content=b"a,b,c\n1,,x\n1,,x\n2,3,x\n", headers=BINARY)
    state = overview(dataset)
    assert state["kind"] == "table" and state["table"]["repeatedRows"] == 1 and state["table"]["constantColumns"] == ["c"]
    assert state["table"]["blankShare"] == pytest.approx(2 / 9)
    assert client.post(f"/api/datasets/{dataset}/health/check", headers=JSON).status_code == 422


def test_requests_are_validated():
    dataset, _ = image_dataset()
    assert client.post(f"/api/datasets/{dataset}/health/check").status_code == 415
    assert client.get("/api/datasets/nope/health").status_code == 404
    assert client.post(f"/api/datasets/{dataset}/health/review", json={"action": "keep", "id": "missing.png", "category": "Red"}).status_code == 404
    assert client.post(f"/api/datasets/{dataset}/health/review", json={"action": "delete-everything"}).status_code == 422
    item = next(training._category_folder("Red", dataset).iterdir()).name
    assert client.post(f"/api/datasets/{dataset}/health/review", json={"action": "keep", "id": item, "category": "Blue"}).status_code == 422
    assert client.post(f"/api/datasets/{dataset}/health/review", json={"action": "move", "id": item, "category": "Green"}).status_code == 422
    small = client.post("/api/datasets", json={"name": "Small", "kind": "image", "categories": ["Red", "Blue"]}).json()["id"]
    assert client.post(f"/api/datasets/{small}/health/check", headers=JSON).status_code == 422


def test_check_waits_for_training():
    dataset, _ = image_dataset()
    graph = build()
    mark_tested(graph)
    assert client.post("/api/train", params={"dataset": dataset}, json=graph).status_code == 200
    assert client.post(f"/api/datasets/{dataset}/health/check", headers=JSON).status_code == 409
    from test_training import wait_for_end
    wait_for_end()
