"""Compares two models fairly: on the same held-back examples, with a test of whether the gap is luck.

Every run holds back examples picked by a fingerprint of their content, so runs on one dataset are
checked on mostly the same examples. Only the examples that both runs held back are compared, so
neither model has seen any of them in training, whichever versions of the data they used.

For models that sort into categories this is McNemar's test, which Dietterich (1998) recommends for
comparing two classifiers on one test set. It looks only at the examples where the models disagree.
For models that predict numbers it is the matching sign test: which model was closer on each row.
"""
from __future__ import annotations

import math

from . import training
from .training import TrainingError

MIN_SHARED = 10
SIGNIFICANT = 0.05


def chance_of_luck(wins: int, losses: int) -> float:
    """Two-sided exact binomial test: how often a gap this big appears between equally good models."""
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** total)


def _model(job_id: str, dataset: str) -> tuple[training.Job, dict]:
    job = training.stored(job_id)
    if job.dataset != dataset:
        raise TrainingError(404, "That model wasn't trained on this dataset.")
    if job.status != "finished":
        raise TrainingError(409, f'"{job.projectName}" didn\'t finish training, so it has no results to compare.')
    if job.task == "writer":
        raise TrainingError(422, "Writers can't be compared here yet. Each one's score shows how well it guesses held-back text.")
    results = training.held_out(job)
    if results is None:
        raise TrainingError(409, f'"{job.projectName}" was trained before comparisons were added. Train it again to compare it.')
    return job, results


def _name(job: training.Job) -> dict:
    return {"id": job.id, "name": job.projectName, "version": job.datasetVersion, "startedAt": job.startedAt}


def compare(dataset: str, first_id: str, second_id: str) -> dict:
    if first_id == second_id:
        raise TrainingError(422, "Choose two different models.")
    first, first_results = _model(first_id, dataset)
    second, second_results = _model(second_id, dataset)
    if first_results["task"] != second_results["task"]:
        raise TrainingError(422, "One of these models predicts a number and the other a category, so they can't be compared.")
    one, two = first_results["rows"], second_results["rows"]
    # An example whose label changed between the runs was checked against different answers, so it's left out.
    shared = [key for key in one.keys() & two.keys() if one[key][0] == two[key][0]]
    reply = {"first": _name(first), "second": _name(second), "task": first_results["task"], "shared": len(shared)}
    if len(shared) < MIN_SHARED:
        return reply | {
            "verdict": "too-few",
            "summary": f"These models share only {len(shared)} held-back examples, too few to compare. Train both on more of the same data.",
        }
    if first_results["task"] == "category":
        first_right = {key for key in shared if one[key][1] == one[key][0]}
        second_right = {key for key in shared if two[key][1] == two[key][0]}
        wins, losses = len(first_right - second_right), len(second_right - first_right)
        reply |= {
            "firstScore": len(first_right) / len(shared), "secondScore": len(second_right) / len(shared),
            "bothRight": len(first_right & second_right), "neitherRight": len(shared) - len(first_right | second_right),
            "onlyFirst": wins, "onlySecond": losses,
        }
        lead = "got {} examples right that the other missed, and missed {} that it got right"
    else:
        first_error = {key: abs(one[key][1] - one[key][0]) for key in shared}
        second_error = {key: abs(two[key][1] - two[key][0]) for key in shared}
        wins = sum(first_error[key] < second_error[key] for key in shared)
        losses = sum(second_error[key] < first_error[key] for key in shared)
        reply |= {
            "firstScore": sum(first_error.values()) / len(shared), "secondScore": sum(second_error.values()) / len(shared),
            "onlyFirst": wins, "onlySecond": losses, "ties": len(shared) - wins - losses,
        }
        lead = "was closer on {} rows, and further off on {}"
    luck = chance_of_luck(wins, losses)
    reply["chanceOfLuck"] = luck
    names = ("Model A", "Model B")  # Two runs often share a project name.
    if wins == losses:
        reply["verdict"] = "same"
        reply["summary"] = f"No difference between them. Model A {lead.format(wins, losses)}."
    elif luck < SIGNIFICANT:
        winner, ahead, behind = (names[0], wins, losses) if wins > losses else (names[1], losses, wins)
        reply["verdict"] = "first" if wins > losses else "second"
        reply["summary"] = (
            f"{winner} is better. It {lead.format(ahead, behind)}. "
            f"A gap this big between equally good models happens {_percent(luck)} of the time, so it's unlikely to be luck."
        )
    else:
        reply["verdict"] = "unclear"
        reply["summary"] = (
            f"No clear winner. The split was {max(wins, losses)} to {min(wins, losses)}, and a gap this big between equally good "
            f"models happens {_percent(luck)} of the time. More held-back examples would settle it."
        )
    return reply


def _percent(value: float) -> str:
    if value < 0.001:
        return "less than 0.1%"
    return f"{100 * value:.1f}%" if value < 0.1 else f"{100 * value:.0f}%"
