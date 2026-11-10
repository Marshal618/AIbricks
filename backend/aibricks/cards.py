"""Model cards: a short page saying what a model is for, what it learned from and where it falls short.

The sections follow "Model Cards for Model Reporting" (Mitchell et al., 2019): details, intended use,
training data, evaluation, results and caveats. Everything but the intended use is filled in from the
run itself. Caveats are worked out from the results, because people rarely write them by hand.
"""
from __future__ import annotations

import json
import math
import time

from pydantic import BaseModel, Field

from . import datasets, training, versions
from .training import TrainingError

NOTES_FILE = "card.json"
FEW_CHECKED = 30
WEAK_CATEGORY = 0.7
IMBALANCED = 3
METHODS = {
    "image": "A neural network built from blocks in AI Bricks and trained with PyTorch.",
    "text": "A fine-tuned sentence model (sentence-transformers/all-MiniLM-L6-v2), trained on pairs of examples as in the "
            "SetFit method, with a logistic regression that picks the category.",
    "table": "Gradient-boosted decision trees (scikit-learn's HistGradientBoosting).",
    "anomaly": "Learns from normal photos only, following PatchCore (Roth et al., 2022): patch features from a pretrained "
               "ResNet-18 are kept as a memory, and a photo is unusual when its least normal patch is far from all of them.",
    "sound": "A frozen pretrained Audio Spectrogram Transformer (MIT/ast-finetuned-audioset-10-10-0.4593) reading 1-second "
             "spectrograms, with a linear classifier trained on each clip and two time-shifted, louder or quieter copies.",
}
THINGS = {"anomaly": "photos", "image": "photos", "text": "texts", "table": "rows", "sound": "1-second clips"}
THING = {"anomaly": "photo", "image": "photo", "text": "text", "table": "row", "sound": "clip"}


class CardNotes(BaseModel):
    intendedUse: str = Field(default="", max_length=2000)
    notFor: str = Field(default="", max_length=2000)


def _notes(job: training.Job) -> CardNotes:
    try:
        return CardNotes(**json.loads((training._folder(job) / NOTES_FILE).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return CardNotes()


def save_notes(job_id: str, notes: CardNotes) -> dict:
    with training._lock:
        job = training.stored(job_id)
        (training._folder(job) / NOTES_FILE).write_text(notes.model_dump_json(), encoding="utf-8")
    return card(job_id)


def likely_range(correct: int, total: int) -> tuple[float, float]:
    """The 95% Wilson interval: where the true accuracy probably lies, given how few were checked."""
    if total == 0:
        return 0.0, 1.0
    z, share = 1.96, correct / total
    middle = (share + z * z / (2 * total)) / (1 + z * z / total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, middle - spread), min(1.0, middle + spread)


def _percent(value: float) -> str:
    return f"{100 * value:.0f}%"


def _config(job: training.Job) -> dict:
    """A spreadsheet model's target and columns, kept with the run."""
    try:
        return json.loads((training._folder(job) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _job_text(job: training.Job, metrics: dict, things: str) -> str:
    if metrics.get("task") == "number":
        return f"predicts the number in “{_config(job).get('target') or 'the chosen column'}”"
    target = _config(job).get("target")
    return f"sorts {things}{' by “' + target + '”' if target else ''} into {len(job.categories)} categories: {', '.join(job.categories)}"


def card(job_id: str) -> dict:
    job = training.stored(job_id)
    if job.status != "finished":
        raise TrainingError(409, "A model card is made once training has finished.")
    try:
        meta = datasets._read(job.dataset)
    except TrainingError:
        meta = None
    notes = _notes(job)
    things = THINGS.get(job.task, "examples")
    metrics = job.metrics or {}
    predicts_number = metrics.get("task") == "number"
    sections, caveats = [], []

    trained = time.strftime("%d %B %Y", time.localtime(job.startedAt))
    seconds = (job.finishedAt or job.startedAt) - job.startedAt
    minutes = round(seconds / 60)
    took = "under a minute" if seconds < 60 else f"about {minutes} minute{'s' if minutes != 1 else ''}"
    details = [
        f"Name: {job.projectName}",
        f"Job: {_job_text(job, metrics, things)}",
        f"Method: {METHODS.get(job.task, 'Trained in AI Bricks.')}",
        f"Trained: {trained}, in {took}{' on ' + job.device if job.device else ''}",
        f"Model ID: {job.id}",
    ]
    if job.modelRemoved:
        details.append("The model file has since been removed to save space. This card describes it as it was.")
    sections.append({"title": "Model details", "lines": details})

    sections.append({"title": "Intended use", "lines": [
        f"Use it for: {notes.intendedUse.strip() or 'Not written yet. Say what this model is meant for.'}",
        f"Don't use it for: {notes.notFor.strip() or 'Not written yet. Say where it should not be trusted.'}",
    ], "editable": True})

    data = [f"Dataset: {meta['name'] if meta else 'a dataset that has since been deleted'}"
            + (f", version {job.datasetVersion}" if job.datasetVersion else "")]
    if meta and meta.get("description"):
        data.append(f"About the data: {meta['description']}")
    data.append(f"Licence: {meta.get('licence') or 'not given'}" if meta else "Licence: unknown")
    if job.task == "table":
        data.append(f"Rows: {job.trainCount:,} to learn from and {job.checkCount:,} held back")
        features = [item["name"] for item in _config(job).get("features", [])]
        if features:
            data.append(f"Columns it learned from: {', '.join(features)}")
    else:
        data.append(f"{things.capitalize()}: {job.trainCount:,} to learn from and {job.checkCount:,} held back")
        data.append("Per category: " + ", ".join(f"{name} {amount:,}" for name, amount in job.categories.items()))
        amounts = [amount for amount in job.categories.values() if amount]
        if amounts and max(amounts) > IMBALANCED * min(amounts):
            caveats.append(f"The categories are uneven: the largest has {max(amounts) // min(amounts)} times as many {things} as the "
                           "smallest. The model may lean towards the larger ones.")
    if versions.credits(job):
        data.append("Some of this data came from open datasets. Their credits are in DATA_SOURCES.md, in the download.")
    sections.append({"title": "Training data", "lines": data})

    sections.append({"title": "How it was checked", "lines": [
        f"{job.checkCount:,} {things} were held back from training and used only to check the model.",
        f"They are picked by a fingerprint of each {THING.get(job.task, 'example')}, so other models trained on this dataset were checked on mostly "
        "the same ones. That makes comparing models on the Models tab fair.",
    ]})

    results = []
    review = job.review or {}
    if predicts_number:
        results.append(f"Average error: {metrics['mae']:,.4g}, against {metrics['baselineMae']:,.4g} from always guessing the average")
        results.append(f"R²: {metrics['r2']:.2f} (1 is perfect, 0 is no better than guessing the average)")
        if metrics["r2"] < 0.3:
            caveats.append("It explains little of how this number varies, so treat its predictions as rough.")
    elif review.get("total"):
        correct, total = review["correct"], review["total"]
        low, high = likely_range(correct, total)
        results.append(f"Accuracy: {_percent(correct / total)} ({correct:,} of {total:,} right). The true figure is probably "
                       f"between {_percent(low)} and {_percent(high)}.")
        results.append(f"Always guessing the most common category would score {_percent(job.baseline)}.")
        if total < FEW_CHECKED:
            caveats.append(f"It was checked on only {total} {things}, so its real accuracy could be anywhere from "
                           f"{_percent(low)} to {_percent(high)}. Add more data for a firmer figure.")
        if correct / total < job.baseline + 0.05:
            caveats.append("It's barely better than always guessing the most common category.")
        for name, counts in review.get("categories", {}).items():
            if counts["total"]:
                share = counts["correct"] / counts["total"]
                results.append(f"{name}: {counts['correct']} of {counts['total']} right ({_percent(share)})")
                if counts["total"] >= 3 and share < WEAK_CATEGORY:
                    caveats.append(f"It's weak on “{name}”, getting {_percent(share)} right. Add more {things} of it, or check its labels.")
    for item in (metrics.get("importance") or [])[:5]:
        results.append(f"Leans on “{item['name']}” for {_percent(item['share'])} of its score")
    sections.append({"title": "Results", "lines": results or ["No results were recorded for this run."]})

    caveats += metrics.get("warnings") or []
    caveats.append(f"It was only checked on {things} from this dataset. {things.capitalize()} from elsewhere, or that look different, "
                   "may score lower.")
    sections.append({"title": "Limitations", "lines": caveats})

    return {"jobId": job.id, "notes": notes.model_dump(), "sections": sections, "markdown": _markdown(job, sections)}


def _markdown(job: training.Job, sections: list[dict]) -> str:
    parts = [f"# Model card: {job.projectName}", ""]
    for section in sections:
        parts += [f"## {section['title']}", ""] + [f"- {line}" for line in section["lines"]] + [""]
    parts.append("_Made by AI Bricks. The sections follow “Model Cards for Model Reporting” (Mitchell et al., 2019)._")
    return "\n".join(parts) + "\n"


def markdown(job: training.Job) -> str | None:
    try:
        return card(job.id)["markdown"]
    except TrainingError:
        return None
