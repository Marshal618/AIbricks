"""The data coach: learning curves that say how much more data would help, and what to label next."""
import random
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from test_health import image_dataset, pattern
from test_training import BINARY, workspace  # noqa: F401

from aibricks import coach
from aibricks.main import app

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aibricks"))
import coach_check  # noqa: E402

client = TestClient(app)
JSON = {"Content-Type": "application/json"}
CACHE = Path(__file__).resolve().parents[1] / ".test-cache"


@pytest.fixture(autouse=True)
def shared_cache(monkeypatch):
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))
    coach._job = None


def run(dataset):
    reply = client.post(f"/api/datasets/{dataset}/coach", headers=JSON)
    assert reply.status_code == 200, reply.text
    deadline = time.time() + 300
    while time.time() < deadline:
        state = client.get(f"/api/datasets/{dataset}/coach").json()
        if state["job"]["status"] != "running":
            assert state["job"]["status"] == "finished", state["job"]["error"]
            return state
        time.sleep(0.2)
    raise AssertionError("the coach did not finish")


def test_a_power_law_is_recovered_and_carried_forward():
    points = [(n, 0.05 + 0.8 * n ** -0.5) for n in (10, 20, 40, 80, 160)]
    curve = coach_check.fit_curve(points)
    for n in (320, 640):
        assert abs(coach_check.error_at(curve, n) - (0.05 + 0.8 * n ** -0.5)) < 0.01
    assert coach_check.fit_curve(points[:2]) is None


def test_picks_are_unsure_and_unlike_each_other():
    torch.manual_seed(0)
    # Two categories along one direction. Unlabelled: ten copies of a borderline example, one other
    # borderline example, and examples deep inside each category.
    labelled = F.normalize(torch.tensor([[1.0, 0.1 * k, 0] for k in range(5)] + [[-1.0, 0.1 * k, 0] for k in range(5)]), dim=1)
    targets = torch.tensor([0] * 5 + [1] * 5)
    pool = F.normalize(torch.tensor([[0.02, 0.0, 1.0]] * 10 + [[0.0, 1.0, 0.3], [1.0, 0.0, 0.0], [-1.0, 0.05, 0.0]]), dim=1)
    ranked = coach_check.next_to_label(labelled, targets, 2, pool, ["A", "B"])
    first_two = [row["index"] for row in ranked[:2]]
    assert sum(index < 10 for index in first_two) == 1 and 10 in first_two  # one copy, then something different
    position = {row["index"]: at for at, row in enumerate(ranked)}
    assert position[10] < min(position[11], position[12])  # clear-cut ones come after the borderline one
    picked = [row["index"] for row in ranked if row["picked"]]
    assert sum(index < 10 for index in picked) == 1  # copies of a pick teach nothing new, so only one is picked
    assert sorted(position) == list(range(len(pool)))  # each example once
    assert all(row["guess"] in ("A", "B") for row in ranked) and len(ranked) == len(pool)


def test_the_coach_draws_a_curve_and_orders_the_labelling_queue(workspace):  # noqa: F811
    dataset, _ = image_dataset(red=20, blue=20)
    for seed in range(12):
        kind = "Red" if seed % 2 else "Blue"
        assert client.post("/api/label/images", params={"dataset": dataset}, content=pattern(kind, 5000 + seed), headers=BINARY).status_code == 200
    before = client.get(f"/api/datasets/{dataset}/coach").json()
    assert before["result"] is None and before["labelledNow"] == 40
    assert client.get("/api/label/images", params={"dataset": dataset}).json()["coachOrder"] is False

    state = run(dataset)
    result = state["result"]
    assert result["labelled"] == 40 and not result["stale"]
    curve = result["curve"]
    assert 0 <= curve["now"]["accuracy"] <= 1 and len(curve["more"]) == 3
    assert [row["examples"] for row in curve["more"]] == [60, 80, 160]
    accuracies = [curve["now"]["accuracy"]] + [row["accuracy"] for row in curve["more"]]
    assert accuracies == sorted(accuracies)  # more data never looks worse on the fitted curve
    assert all(row["low"] <= row["accuracy"] <= row["high"] for row in curve["more"])
    assert len(curve["measured"]) == len(coach_check.SHARES)
    assert {row["category"] for row in result["categories"]} == {"Red", "Blue"}
    assert result["verdict"] in ("helps", "some", "levelling") and result["rough"] is True
    assert len(result["next"]) == 12 and result["unlabelledRanked"] == 12

    queue = client.get("/api/label/images", params={"dataset": dataset}).json()
    assert queue["coachOrder"] is True
    assert [item["id"] for item in queue["items"]] == [row["id"] for row in result["next"]]

    # Labelling one takes it off the list.
    first = result["next"][0]["id"]
    assert client.post(f"/api/label/images/{first}/assign", params={"dataset": dataset}, json={"category": "Red"}).status_code == 200
    after = client.get(f"/api/datasets/{dataset}/coach").json()["result"]
    assert first not in [row["id"] for row in after["next"]] and after["unlabelledRanked"] == 11


def test_text_datasets_are_coached_too(workspace):  # noqa: F811
    dataset = client.post("/api/datasets", json={"name": "Moods", "kind": "text", "categories": ["Good", "Bad"]}).json()["id"]
    rng = random.Random(0)
    happy, sad = ["wonderful", "happy", "lovely", "bright", "kind", "warm"], ["awful", "miserable", "gloomy", "grim", "harsh", "cold"]
    things = ["day", "film", "meal", "trip", "song", "book", "game", "party"]
    good = sorted({f"what a {rng.choice(happy)} {rng.choice(things)}" for _ in range(60)})[:14]
    bad = sorted({f"what an {rng.choice(sad)} {rng.choice(things)}" for _ in range(60)})[:14]
    waiting = ["a truly lovely song", "a really grim party", "an ordinary day"]
    client.post("/api/label/text", params={"dataset": dataset, "kind": "lines"}, content="\n".join(good + bad + waiting).encode(), headers=BINARY)
    for row in client.get("/api/label/text", params={"dataset": dataset}).json()["items"]:
        if row["text"] not in waiting:
            client.post(f"/api/label/text/{row['id']}/assign", params={"dataset": dataset}, json={"category": "Good" if row["text"] in good else "Bad"})
    result = run(dataset)["result"]
    assert result["labelled"] == 28 and result["curve"]["now"]["accuracy"] > 0.7
    assert len(result["next"]) == 3
    queue = client.get("/api/label/text", params={"dataset": dataset}).json()
    assert queue["coachOrder"] is True and [item["id"] for item in queue["items"]] == [row["id"] for row in result["next"]]
    assert "unsuggested" not in queue or queue["unsuggested"] == 0


def test_the_coach_needs_enough_labels_and_a_sorting_dataset(workspace):  # noqa: F811
    dataset, _ = image_dataset(red=4, blue=20)
    reply = client.post(f"/api/datasets/{dataset}/coach", headers=JSON)
    assert reply.status_code == 422 and "5 examples" in reply.json()["detail"]["message"]
    finding = client.post("/api/datasets", json={"name": "Boxes", "kind": "image", "goal": "find", "categories": ["A"]}).json()["id"]
    assert client.post(f"/api/datasets/{finding}/coach", headers=JSON).status_code == 422
    assert client.get(f"/api/datasets/{finding}/coach").status_code == 422
    assert client.post(f"/api/datasets/{dataset}/coach").status_code == 415


def test_advice_reads_the_curve():
    def result(now, doubled, gains):
        return {"labelled": 200, "curve": {"now": {"accuracy": now}, "more": [{}, {"accuracy": doubled}, {}]},
                "categories": [{"category": name, "count": 50, "gainIfDoubled": gain} for name, gain in gains.items()]}
    helps = coach.advice(result(0.70, 0.80, {"Fox": 0.24, "Cat": 0.02, "Owl": 0.04}))
    assert helps["verdict"] == "helps" and helps["focus"] == ["Fox"] and helps["rough"] is False
    # Scaled to add up to the overall gain of 10 points.
    gains = {row["category"]: row["gainIfDoubled"] for row in helps["categories"]}
    assert gains == {"Fox": 0.24, "Cat": 0.02, "Owl": 0.04}
    assert abs(sum(gains.values()) / 3 - 0.10) < 0.001
    halved = coach.advice(result(0.70, 0.75, {"Fox": 0.24, "Cat": 0.02, "Owl": 0.04}))
    assert {row["category"]: row["gainIfDoubled"] for row in halved["categories"]} == {"Fox": 0.12, "Cat": 0.01, "Owl": 0.02}
    assert coach.advice(result(0.95, 0.955, {}))["verdict"] == "levelling"
    assert coach.advice(result(0.90, 0.92, {}))["verdict"] == "some"
    assert coach.advice({"labelled": 3, "curve": None, "categories": []})["verdict"] == "unknown"
