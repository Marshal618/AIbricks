"""Finds open labelled datasets on Hugging Face and copies a sample into the local stores.

Everything fetched here is untrusted. Images pass the same checks as the user's own uploads,
text is only ever stored as text, and image links are followed only to Hugging Face's own hosts.
No account or key is used, so only public, ungated datasets can be imported.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx

from . import labelling, training
from .graph import NAME_RE
from .training import TrainingError

HUB = "https://huggingface.co/api/datasets"
ROWS = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100
MAX_PAGES = 120
MAX_PER_CATEGORY = 500
FETCH_WORKERS = 8
TIMEOUT_SECONDS = 30
RETRIES = 3
RETRY_WAIT_SECONDS = 2
MAX_RESPONSE_BYTES = 40 * 1024 * 1024
HEADERS = {"User-Agent": "AI-Bricks"}
TASKS = {"image": "image-classification", "text": "text-classification"}
TEXT_COLUMNS = ("text", "sentence", "content", "review", "message", "sms", "comment", "document", "title")

_DATASET_ID = re.compile(r"^[A-Za-z0-9][\w.\-]{0,95}(/[A-Za-z0-9][\w.\-]{0,95})?$")
_lock = threading.Lock()
_job: dict | None = None
_cancel = threading.Event()


def _get(url: str, params: dict | None = None) -> httpx.Response:
    """Fetches a page, waiting and retrying when the service is busy."""
    for attempt in range(RETRIES):
        try:
            response = httpx.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS, follow_redirects=False)
        except httpx.HTTPError as problem:
            if attempt == RETRIES - 1:
                raise TrainingError(502, f"Hugging Face couldn't be reached: {type(problem).__name__}.") from None
        else:
            if response.status_code not in (429, 500, 502, 503, 504):
                return response
        time.sleep(RETRY_WAIT_SECONDS * (attempt + 1))
    raise TrainingError(502, "Hugging Face is busy right now. Try again in a minute.")


def _get_json(url: str, params: dict | None = None):
    response = _get(url, params)
    if response.status_code != 200:
        raise TrainingError(502, "Hugging Face can't provide this dataset's rows, so pick another one.")
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise TrainingError(502, "Hugging Face sent more data than expected.")
    try:
        return response.json()
    except ValueError:
        raise TrainingError(502, "Hugging Face sent a reply that couldn't be read.") from None


def _licence(tags: list) -> str:
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return "unknown"


def search(query: str, kind: str) -> list[dict]:
    params = {"filter": f"task_categories:{TASKS[kind]}", "sort": "downloads", "limit": 20, "full": "true"}
    if query.strip():
        params["search"] = query.strip()
    rows = _get_json(HUB, params)
    return [
        {"id": row["id"], "downloads": row.get("downloads", 0), "licence": _licence(row.get("tags"))}
        for row in rows
        if isinstance(row, dict) and row.get("id") and not row.get("gated") and not row.get("private")
    ]


def _check_id(dataset: str) -> str:
    if not _DATASET_ID.match(dataset):
        raise TrainingError(422, "That dataset name isn't valid.")
    return dataset


def _describe(dataset: str, kind: str) -> dict:
    """Works out which split and columns hold the examples and their categories."""
    _check_id(dataset)
    hub = _get_json(f"{HUB}/{dataset}")
    if hub.get("gated") or hub.get("private"):
        raise TrainingError(422, "This dataset needs an account and permission, so it can't be imported here.")
    splits = _get_json(f"{ROWS}/splits", {"dataset": dataset}).get("splits") or []
    if not splits:
        raise TrainingError(422, "Hugging Face has no readable rows for this dataset, so pick another one.")
    chosen = next((split for split in splits if split.get("split") == "train"), splits[0])
    first = _get_json(f"{ROWS}/rows", {
        "dataset": dataset, "config": chosen["config"], "split": chosen["split"], "offset": 0, "length": 8,
    })
    features = first.get("features") or []
    label = next((f for f in features if f.get("type", {}).get("_type") == "ClassLabel"), None)
    if label is None:
        raise TrainingError(422, "This dataset has no category column that the app can read, so pick another one.")
    if kind == "image":
        content = next((f for f in features if f.get("type", {}).get("_type") == "Image"), None)
        if content is None:
            raise TrainingError(422, "This dataset has no image column that the app can read.")
    else:
        strings = [f for f in features if f.get("type", {}).get("_type") == "Value" and f.get("type", {}).get("dtype") == "string"]
        content = next((f for name in TEXT_COLUMNS for f in strings if f.get("name", "").lower() == name), strings[0] if strings else None)
        if content is None:
            raise TrainingError(422, "This dataset has no text column that the app can read.")
    return {
        "dataset": dataset, "kind": kind, "licence": _licence(hub.get("tags")), "config": chosen["config"], "split": chosen["split"],
        "labelColumn": label["name"], "contentColumn": content["name"], "labels": label["type"].get("names") or [],
        "totalRows": first.get("num_rows_total") or len(first.get("rows") or []), "firstRows": first.get("rows") or [],
    }


def _names_usable(labels: list[str]) -> bool:
    """Whether these labels can become the model's categories exactly as written."""
    folded = {str(label).casefold() for label in labels}
    return (
        2 <= len(labels) <= 100 and len(folded) == len(labels)
        and all(NAME_RE.match(str(label)) and "," not in str(label) and "/" not in str(label) and not str(label).startswith(".") for label in labels)
    )


def _safe_image_link(cell) -> str | None:
    link = cell.get("src") if isinstance(cell, dict) else None
    if not isinstance(link, str):
        return None
    parts = urlsplit(link)
    host = parts.hostname or ""
    return link if parts.scheme == "https" and (host == "huggingface.co" or host.endswith(".huggingface.co")) else None


def inspect(dataset: str, kind: str) -> dict:
    info = _describe(dataset, kind)
    preview = []
    for row in info.pop("firstRows"):
        cell = row.get("row", {}).get(info["contentColumn"])
        if kind == "image":
            link = _safe_image_link(cell)
            if link:
                preview.append({"image": link})
        elif isinstance(cell, str):
            preview.append({"text": cell[:300]})
    return info | {"preview": preview[:6], "namesUsable": _names_usable(info["labels"]), "url": f"https://huggingface.co/datasets/{dataset}"}


def _fetch_image(link: str) -> bytes | None:
    try:
        with httpx.stream("GET", link, headers=HEADERS, timeout=TIMEOUT_SECONDS, follow_redirects=False) as response:
            if response.status_code != 200:
                return None
            data = bytearray()
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > training.MAX_IMAGE_BYTES:
                    return None
            return bytes(data)
    except httpx.HTTPError:
        return None


def _store_image(link: str, category: str, target: str) -> str:
    data = _fetch_image(link)
    if data is None:
        return "failed"
    try:
        return "duplicate" if training.add_image(category, data, target)["duplicate"] else "added"
    except TrainingError as problem:
        return "full" if problem.status == 413 and "full" in problem.message else "failed"


def _sources_file(target: str):
    return training.root(target) / "imports.json"


def sources(target: str = training.DEFAULT_DATASET) -> list[dict]:
    try:
        return json.loads(_sources_file(target).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def sources_markdown(target: str = training.DEFAULT_DATASET) -> str | None:
    """Credits for imported data, included with a trained model so the licences travel with it."""
    rows = sources(target)
    if not rows:
        return None
    lines = ["# Data sources", "", "Open datasets imported into AI Bricks, which may have helped train this model.", ""]
    lines += [f"- [{row['dataset']}]({row['url']}): licence {row['licence']}, {row['added']} items" for row in rows]
    return "\n".join(lines) + "\n"


def _record_source(info: dict, added: int, target: str) -> None:
    rows = [row for row in sources(target) if row["dataset"] != info["dataset"]]
    previous = next((row["added"] for row in sources(target) if row["dataset"] == info["dataset"]), 0)
    rows.append({
        "dataset": info["dataset"], "licence": info["licence"], "added": previous + added,
        "url": f"https://huggingface.co/datasets/{info['dataset']}",
    })
    path = _sources_file(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(rows), encoding="utf-8")
    os.replace(temporary, path)


def _run(job: dict, info: dict, mapping: dict[str, str], per_category: int, target: str) -> None:
    """Reads pages spread across the dataset, because many datasets store each category together."""
    categories = sorted(set(mapping.values()))
    have = dict.fromkeys(categories, 0)
    job["counts"] = dict.fromkeys(categories, 0)
    labels, kind = info["labels"], info["kind"]
    offsets = list(range(0, max(info["totalRows"], 1), PAGE_SIZE))
    random.Random(0).shuffle(offsets)
    try:
        for offset in offsets[:MAX_PAGES]:
            if _cancel.is_set() or all(amount >= per_category for amount in have.values()):
                break
            page = _get_json(f"{ROWS}/rows", {
                "dataset": info["dataset"], "config": info["config"], "split": info["split"], "offset": offset, "length": PAGE_SIZE,
            })
            candidates = []
            for row in page.get("rows") or []:
                values = row.get("row", {})
                index = values.get(info["labelColumn"])
                if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(labels):
                    continue
                category = mapping.get(str(labels[index]))
                if category is None or info["contentColumn"] in (row.get("truncated_cells") or []):
                    continue
                cell = values.get(info["contentColumn"])
                content = _safe_image_link(cell) if kind == "image" else (cell.strip() if isinstance(cell, str) else None)
                if not content or (kind == "text" and len(content) > labelling.MAX_TEXT_LENGTH):
                    job["skipped"] += 1
                    continue
                candidates.append((content, category))
            # Downloads can fail, so keep working through the page until each category is full.
            while candidates and not _cancel.is_set():
                planned = dict.fromkeys(categories, 0)
                wanted, later = [], []
                for pair in candidates:
                    if have[pair[1]] + planned[pair[1]] < per_category:
                        planned[pair[1]] += 1
                        wanted.append(pair)
                    else:
                        later.append(pair)
                if not wanted:
                    break
                candidates = later
                if kind == "image":
                    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
                        outcomes = list(pool.map(lambda pair: _store_image(*pair, target), wanted))
                    if "full" in outcomes:
                        raise TrainingError(413, f"The image store is full at {training.MAX_IMAGES:,} images.")
                else:
                    outcomes = labelling.import_texts(wanted, target)
                for (_, category), outcome in zip(wanted, outcomes):
                    if outcome == "failed":
                        job["skipped"] += 1
                        continue
                    have[category] += 1
                    job["counts"][category] += 1
                    job["added"] += outcome == "added"
                    job["duplicates"] += outcome == "duplicate"
            job["pagesRead"] += 1
        job["status"] = "cancelled" if _cancel.is_set() else "finished"
    except TrainingError as problem:
        job["status"], job["error"] = "failed", problem.message
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as problem:
        job["status"], job["error"] = "failed", f"The import stopped: {type(problem).__name__}."
    if job["added"]:
        _record_source(info, job["added"], target)


def start_import(dataset: str, kind: str, mapping: dict[str, str], per_category: int, target: str = training.DEFAULT_DATASET) -> dict:
    global _job
    if not 1 <= per_category <= MAX_PER_CATEGORY:
        raise TrainingError(422, f"Choose between 1 and {MAX_PER_CATEGORY} per category.")
    info = _describe(dataset, kind)
    mapping = {label: category for label, category in mapping.items() if label in {str(name) for name in info["labels"]} and category}
    if not mapping:
        raise TrainingError(422, "Match at least one of the dataset's categories to your model first.")
    with _lock:
        if _job and _job["status"] == "running":
            raise TrainingError(409, "A dataset is already being imported.")
        _cancel.clear()
        _job = {
            "dataset": dataset, "target": target, "kind": kind, "licence": info["licence"], "status": "running", "perCategory": per_category,
            "counts": dict.fromkeys(sorted(set(mapping.values())), 0), "added": 0, "duplicates": 0, "skipped": 0, "pagesRead": 0, "error": None,
        }
        job = _job
    threading.Thread(target=_run, args=(job, info, mapping, per_category, target), daemon=True).start()
    return job


def status(target: str = training.DEFAULT_DATASET) -> dict:
    return {"job": _job if _job and _job["target"] == target else None, "sources": sources(target)}


def cancel() -> None:
    _cancel.set()
