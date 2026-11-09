"""The home screen's overview: runs from every dataset, and what this computer offers."""
from fastapi.testclient import TestClient
from test_compare_and_cards import train_table
from test_tables import create, houses, to_csv, upload
from test_training import workspace  # noqa: F401

from aibricks.main import app

client = TestClient(app)


def test_runs_from_every_dataset_are_listed_newest_first(workspace):  # noqa: F811
    assert client.get("/api/models").json() == {"models": []}
    first, second = create("Houses"), create("Flats")
    for dataset in (first, second):
        upload(dataset, to_csv(*houses(120)))
    older = train_table(first, ["size", "area"])
    newer = train_table(second, ["size"])
    rows = client.get("/api/models").json()["models"]
    assert [row["id"] for row in rows] == [newer["id"], older["id"]]
    assert [row["datasetName"] for row in rows] == ["Flats", "Houses"]
    assert "epochs" not in rows[0] and rows[0]["task"] == "table" and rows[0]["modelBytes"] > 0
    assert [row["id"] for row in client.get("/api/models", params={"limit": 1}).json()["models"]] == [newer["id"]]
    assert client.get("/api/models", params={"limit": 0}).status_code == 422


def test_the_system_check_reports_the_device_and_storage(workspace):  # noqa: F811
    info = client.get("/api/system").json()
    assert info["device"] in {"cuda", "mps", "cpu"} and info["torch"] and info["cpus"] >= 1
    storage = info["storage"]
    assert storage["path"] == str(workspace) and storage["freeBytes"] > 0
    assert set(storage["parts"]) == {"data", "models", "projects", "downloads"}
    assert storage["bytes"] == sum(storage["parts"].values())
