"""Sure sets: cut-offs from held-back chances, and sets that hold the right answer as often as promised."""
import random

import pytest

from aibricks import sure


def test_the_cutoff_follows_the_held_back_chances():
    truths = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # chance given to the right answer
    # With n examples, a level needs the ceil((n + 1) × level)-th smallest miss, so 10 is enough for
    # 9 in 10 (the 10th) but not 19 in 20 (the 11th).
    assert sure.cutoff(truths, 0.9) == pytest.approx(0.2)
    assert sure.cutoff(truths, 0.8) == pytest.approx(0.3)  # the 9th smallest miss is 0.7
    assert sure.cutoff(truths, 0.95) is None
    assert sure.cutoff([1.0] * 40, 0.9) == 1.0


def test_members_are_likeliest_first_and_never_empty():
    chances = {"Cat": 0.55, "Lynx": 0.35, "Dog": 0.1}
    assert sure.members(chances, 0.3) == ["Cat", "Lynx"]
    assert sure.members(chances, 0.9) == ["Cat"]
    assert sure.members(chances, 0.0) == ["Cat", "Lynx", "Dog"]


def test_sets_hold_the_right_answer_as_often_as_promised():
    rng = random.Random(0)
    names = ["A", "B", "C", "D"]

    def example():
        actual = rng.choice(names)
        raw = {name: rng.random() + (1.2 if name == actual else 0) * rng.random() for name in names}
        total = sum(raw.values())
        return {"actual": actual, "chances": {name: value / total for name, value in raw.items()}}

    held = [example() for _ in range(200)]
    found = sure.summary(held)
    assert found["examples"] == 200 and set(found["levels"]) == {"0.8", "0.9", "0.95"}
    fresh = [example() for _ in range(4000)]
    for level, row in found["levels"].items():
        right = sum(item["actual"] in sure.members(item["chances"], row["cutoff"]) for item in fresh) / len(fresh)
        assert right >= float(level) - 0.03, (level, right)
        assert abs(row["right"] - right) < 0.05  # the leave-one-out check predicts it
    # Asking to be right more often needs bigger sets.
    assert found["levels"]["0.8"]["size"] <= found["levels"]["0.9"]["size"] <= found["levels"]["0.95"]["size"]


def test_rows_without_chances_give_nothing():
    assert sure.summary([{"actual": "A", "guess": "A", "chance": 0.9}]) is None
    assert sure.summary([]) is None
