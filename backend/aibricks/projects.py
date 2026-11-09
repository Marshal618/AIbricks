"""Saves named builds as JSON files on this computer, so they outlive the browser's storage."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from .training import workspace

_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class ProjectIn(BaseModel):
    projectName: str = Field(max_length=200)
    training: str = Field(max_length=40)
    trainingOverride: dict[str, Any] | None = None
    nodes: list[dict[str, Any]] = Field(max_length=1000)
    edges: list[dict[str, Any]] = Field(max_length=5000)
    kitId: str | None = Field(default=None, max_length=80)
    datasetId: str | None = Field(default=None, max_length=40)


def _path(project_id: str):
    if not _ID.match(project_id):
        raise KeyError(project_id)
    return workspace() / "projects" / f"{project_id}.json"


def save(project_id: str, project: ProjectIn) -> dict:
    path = _path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = time.time()
    temporary = path.with_name(f"{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(project.model_dump() | {"updatedAt": updated}), encoding="utf-8")
    os.replace(temporary, path)
    return {"id": project_id, "updatedAt": updated}


def load(project_id: str) -> dict:
    try:
        return json.loads(_path(project_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise KeyError(project_id) from None


def delete(project_id: str) -> None:
    _path(project_id).unlink(missing_ok=True)


def listing() -> list[dict]:
    rows = []
    for path in (workspace() / "projects").glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "id": path.stem,
                "projectName": data["projectName"],
                "components": len(data["nodes"]),
                "updatedAt": data["updatedAt"],
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue  # A damaged file shouldn't hide the others.
    return sorted(rows, key=lambda row: row["updatedAt"], reverse=True)
