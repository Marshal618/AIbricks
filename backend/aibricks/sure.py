"""Sure sets: instead of one guess, every answer it can't rule out, right a chosen share of the time.

This is split conformal prediction (Vovk et al., 2005; as explained by Angelopoulos and Bates, 2021).
The held-back examples, which the model never trained on, show how much chance it tends to give the
right answer. A cut-off is set so that, for 9 in 10 of them, the right answer's chance was at least
that high. On a new example, every answer at or above the cut-off goes in the set. If new examples
are like the held-back ones, the set holds the right answer at least 9 times in 10, whatever the
model, however good or bad its chances are. A weak model just gives bigger sets.

The guarantee needs the held-back examples to be untouched by training. Training here keeps the
version of the model that scored best on them, which bends that a little, so the check below
measures it honestly: each held-back example is judged with a cut-off worked out from the others.
"""
from __future__ import annotations

import math

LEVELS = (0.8, 0.9, 0.95)
DEFAULT_LEVEL = 0.9


def cutoff(truths: list[float], level: float) -> float | None:
    """The lowest chance an answer needs to join the set, or None with too few examples for the level."""
    n = len(truths)
    rank = math.ceil((n + 1) * level)
    if rank > n:
        return None
    misses = sorted(1 - truth for truth in truths)  # how far each right answer fell short of certain
    return max(0.0, 1 - misses[rank - 1])


def members(chances: dict[str, float], limit: float) -> list[str]:
    """The answers at or above the cut-off, likeliest first. Never empty: the top answer always goes in."""
    ranked = sorted(chances, key=lambda name: -chances[name])
    chosen = [name for name in ranked if chances[name] >= limit - 1e-9]
    return chosen or ranked[:1]


def summary(rows: list[dict]) -> dict | None:
    """Cut-offs for each level from held-back rows of {"actual", "chances"}, and how they did.

    "right" and "size" come from judging each row with a cut-off from all the other rows.
    """
    rows = [row for row in rows if isinstance(row.get("chances"), dict) and row.get("actual") is not None]
    if not rows:
        return None
    truths = [float(row["chances"].get(str(row["actual"]), 0.0)) for row in rows]
    levels = {}
    for level in LEVELS:
        limit = cutoff(truths, level)
        if limit is None:
            continue
        right, sizes = 0, []
        for index, row in enumerate(rows):
            others = cutoff(truths[:index] + truths[index + 1:], level)
            chosen = members(row["chances"], 0.0 if others is None else others)
            right += str(row["actual"]) in chosen
            sizes.append(len(chosen))
        levels[str(level)] = {
            "cutoff": round(limit, 6), "right": round(right / len(rows), 4),
            "size": round(sum(sizes) / len(sizes), 3), "single": round(sum(size == 1 for size in sizes) / len(sizes), 4),
        }
    return {"examples": len(rows), "levels": levels} if levels else None
