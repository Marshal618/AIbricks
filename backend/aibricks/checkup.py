"""Reads a finished training run's loss chart and says, in plain words, what to try next.

These are the standard checks an experienced person makes by eye (bias and variance, a learning
rate that's too high, training that stopped too soon), written as simple rules. Each finding
offers fixes a person can apply in the builder.
"""
from __future__ import annotations

import math

OVERFIT_RISE = 0.15  # check loss this far above its best by the end
OVERFIT_GAP = 0.25  # check loss this far above training loss at its best
STALL_SHARE = 0.75  # training loss still above this share of where it started
STILL_LEARNING = 0.03  # check loss still falling this much in the last epoch
NEAR_GUESSING = 0.1  # accuracy within this of always guessing the biggest category


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def check(epochs: list[dict], baseline: float) -> list[dict]:
    """Findings for one run, most important first. Each has a kind, a title, what it means and fixes."""
    if len(epochs) < 2:
        return []
    train = [entry.get("trainLoss") for entry in epochs]
    held = [entry.get("checkLoss") for entry in epochs]
    accuracy = [entry.get("accuracy") for entry in epochs if _finite(entry.get("accuracy"))]
    if not all(_finite(value) for value in train + held):
        return [{
            "kind": "unstable", "title": "Training became unstable",
            "detail": "The loss stopped being a number partway through. The steps it took were too big.",
            "fixes": ["slower"],
        }]
    findings = []
    best = min(range(len(held)), key=held.__getitem__)
    if max(train[1:]) > train[0] * 1.5 or any(later > earlier * 2 for earlier, later in zip(train, train[1:])):
        findings.append({
            "kind": "unstable", "title": "The training loss jumped around",
            "detail": "Instead of falling steadily, it got much worse at times. That usually means the learning rate is too high.",
            "fixes": ["slower"],
        })
    rose = held[-1] > held[best] * (1 + OVERFIT_RISE) and best < len(held) - 1
    gap = held[best] > train[best] * (1 + OVERFIT_GAP) and train[-1] < train[best]
    if rose or (gap and best < len(held) - 1):
        findings.append({
            "kind": "memorising", "title": "It started memorising the training photos",
            "detail": f"It did best on the held-back photos at epoch {best + 1}, then got worse on them while still improving on "
                      "its training photos. It's learning those exact pictures instead of what the categories look like. "
                      "The saved model is the one from its best epoch.",
            "fixes": ["augment", "more_photos", "pretrained"],
        })
    top = max(accuracy) if accuracy else None
    stalled = train[-1] > train[0] * STALL_SHARE
    if top is not None and top <= baseline + NEAR_GUESSING:
        findings.append({
            "kind": "too_simple", "title": "It barely beat guessing",
            "detail": f"Its best score, {round(top * 100)}%, is close to the {round(baseline * 100)}% it would get by always "
                      "guessing the biggest category. It hasn't found what tells the categories apart.",
            "fixes": ["pretrained", "longer", "more_photos"],
        })
    elif stalled:
        findings.append({
            "kind": "too_simple", "title": "It hardly learned",
            "detail": "The training loss barely went down, so the model may be too small, or need more time to learn.",
            "fixes": ["longer", "pretrained"],
        })
    if best == len(held) - 1 and held[-2] - held[-1] > held[-2] * STILL_LEARNING and not any(f["kind"] == "unstable" for f in findings):
        findings.append({
            "kind": "still_learning", "title": "It was still improving when training stopped",
            "detail": "The held-back loss was still falling in the last epoch, so more training will probably help.",
            "fixes": ["longer"],
        })
    if not findings:
        findings.append({
            "kind": "healthy", "title": "Training looks healthy",
            "detail": "The losses fell and levelled off without signs of memorising. To do better, add more varied photos, "
                      "or let “Find a better model” try other options.",
            "fixes": ["experiment"],
        })
    return findings
