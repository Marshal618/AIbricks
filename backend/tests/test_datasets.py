"""Datasets as saved items, kept apart from each other, and training a model from one."""
import csv
import io
import time
import zipfile

from fastapi.testclient import TestClient
from test_opendata import fake_hub  # noqa: F401
from test_training import BINARY, BLUE, RED, build, mark_tested, picture, wait_for_end, workspace  # noqa: F401

from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


def create(name="Colours", kind="image", categories=("Red", "Blue")):
    reply = client.post("/api/datasets", json={"name": name, "kind": kind, "categories": list(categories)})
    assert reply.status_code == 200, reply.text
    return reply.json()


def counts(dataset, *names):
    return client.get("/api/dataset", params={"dataset": dataset, "category": list(names)}).json()["counts"]


def add(dataset, category, data):
    return client.post("/api/dataset/images", params={"dataset": dataset, "category": category}, content=data, headers=BINARY)


def test_create_list_rename_delete():
    assert client.get("/api/datasets").json() == {"datasets": []}
    made = create()
    assert made["kind"] == "image" and made["categories"] == ["Red", "Blue"] and made["labelled"] == 0
    rows = client.get("/api/datasets").json()["datasets"]
    assert [row["id"] for row in rows] == [made["id"]]
    renamed = client.put(f"/api/datasets/{made['id']}/name", json={"name": "  Better   name "}).json()
    assert renamed["name"] == "Better name"
    assert client.delete(f"/api/datasets/{made['id']}").status_code == 200
    assert client.get(f"/api/datasets/{made['id']}").status_code == 404
    assert client.get("/api/datasets").json() == {"datasets": []}


def test_invalid_datasets_are_refused():
    assert client.post("/api/datasets", json={"name": "", "kind": "image"}).status_code == 422
    assert client.post("/api/datasets", json={"name": "   ", "kind": "image"}).status_code == 422
    assert client.post("/api/datasets", json={"name": "x", "kind": "video"}).status_code == 422
    for bad in (["a/b"], ["a,b"], [".hidden"], ["<script>"], ["Same", "same"]):
        assert client.post("/api/datasets", json={"name": "x", "kind": "image", "categories": bad}).status_code == 422, bad


def test_unknown_or_malicious_dataset_ids_are_refused(workspace):  # noqa: F811
    for bad in ["../../etc", "nope", "..", "0" * 12, "a" * 40]:
        assert client.get("/api/dataset", params={"dataset": bad, "category": ["x"]}).status_code == 404, bad
        assert add(bad, "x", picture(RED)).status_code == 404, bad
        assert client.get("/api/label/images", params={"dataset": bad}).status_code == 404, bad
    assert not list(workspace.parent.glob("etc*"))


def test_datasets_are_kept_apart():
    first, second = create("One"), create("Two")
    add(first["id"], "Red", picture(RED, 1))
    assert counts(first["id"], "Red") == {"Red": 1} and counts(second["id"], "Red") == {"Red": 0}
    # The same image can belong to both.
    assert add(second["id"], "Red", picture(RED, 1)).json() == {"duplicate": False, "count": 1}
    client.post("/api/label/images", params={"dataset": first["id"]}, content=picture(BLUE, 2), headers=BINARY)
    assert client.get("/api/label/images", params={"dataset": first["id"]}).json()["total"] == 1
    assert client.get("/api/label/images", params={"dataset": second["id"]}).json()["total"] == 0
    assert client.get(f"/api/datasets/{first['id']}").json()["unsorted"] == 1
    client.delete(f"/api/datasets/{first['id']}")
    assert counts(second["id"], "Red") == {"Red": 1}


def test_categories_can_be_added_renamed_and_removed_without_losing_work():
    made = create(categories=[])
    dataset = made["id"]
    assert client.post(f"/api/datasets/{dataset}/categories", json={"names": ["Red", "Blue"]}).json()["categories"] == ["Red", "Blue"]
    assert client.post(f"/api/datasets/{dataset}/categories", json={"names": ["red", "Green"]}).json()["categories"] == ["Red", "Blue", "Green"]
    assert client.post(f"/api/datasets/{dataset}/categories", json={"names": ["a/b"]}).status_code == 422
    add(dataset, "Red", picture(RED, 1))
    add(dataset, "Red", picture(RED, 2))
    renamed = client.put(f"/api/datasets/{dataset}/categories", json={"old": "Red", "new": "Crimson"}).json()
    assert renamed["categories"] == ["Crimson", "Blue", "Green"] and renamed["counts"]["Crimson"] == 2
    assert client.put(f"/api/datasets/{dataset}/categories", json={"old": "Crimson", "new": "blue"}).status_code == 422
    assert client.put(f"/api/datasets/{dataset}/categories", json={"old": "Nope", "new": "X"}).status_code == 404
    removed = client.delete(f"/api/datasets/{dataset}/categories", params={"name": "Crimson"}).json()
    assert removed["categories"] == ["Blue", "Green"] and removed["unsorted"] == 2


def test_text_datasets_keep_labels_through_category_changes():
    dataset = create("Tickets", "text", ["Billing", "Bug"])["id"]
    params = {"dataset": dataset}
    client.post("/api/label/text", params=params, content=b"My invoice\nIt crashes\n", headers=BINARY)
    items = client.get("/api/label/text", params={**params, "category": ["Billing", "Bug"]}).json()["items"]
    client.post(f"/api/label/text/{items[0]['id']}/assign", params=params, json={"category": "Billing"})
    client.post(f"/api/label/text/{items[1]['id']}/assign", params=params, json={"category": "Bug"})
    client.put(f"/api/datasets/{dataset}/categories", json={"old": "Billing", "new": "Money"})
    summary = client.get(f"/api/datasets/{dataset}").json()
    assert summary["counts"] == {"Money": 1, "Bug": 1} and summary["kind"] == "text"
    rows = list(csv.DictReader(io.StringIO(client.get("/api/label/text/export", params={**params, "category": ["Money", "Bug"]}).text)))
    assert rows == [{"text": "My invoice", "label": "Money"}, {"text": "It crashes", "label": "Bug"}]
    removed = client.delete(f"/api/datasets/{dataset}/categories", params={"name": "Bug"}).json()
    assert removed["unsorted"] == 1 and removed["counts"] == {"Money": 1}
    other = create("Other", "text", ["Billing"])["id"]
    assert client.get("/api/label/text", params={"dataset": other, "category": ["Billing"]}).json()["total"] == 0


def test_train_from_a_dataset_review_and_fix_within_it(workspace):  # noqa: F811
    dataset = create()["id"]
    other = create("Other")["id"]
    for seed in range(20):
        add(dataset, "Red", picture(RED, seed))
        add(dataset, "Blue", picture(BLUE, seed))
    for seed in range(100, 110):
        add(dataset, "Red", picture(BLUE, seed))
    add(other, "Red", picture(RED, 500))
    graph = build()
    mark_tested(graph)
    # An empty dataset can't train, even though another one has images.
    assert client.post("/api/train", params={"dataset": other}, json=graph).status_code == 422
    assert client.post("/api/train", params={"dataset": dataset}, json=graph).status_code == 200
    assert client.delete(f"/api/datasets/{dataset}").status_code == 409
    job = wait_for_end()
    assert job["status"] == "finished" and job["dataset"] == dataset and job["trainCount"] == 40
    assert job["categories"] == {"Red": 30, "Blue": 20}

    # Mistakes read from, and fix within, the dataset that trained the model.
    assert job["review"]["mistakeCount"] > 0
    mistake = job["review"]["mistakes"][0]
    assert client.get(f"/api/train/{job['id']}/mistakes/0").status_code == 200
    target = "Blue" if mistake["actual"] == "Red" else "Red"
    before = counts(dataset, "Red", "Blue")
    client.post(f"/api/train/{job['id']}/mistakes/0/fix", json={"action": "move", "category": target})
    assert counts(dataset, mistake["actual"])[mistake["actual"]] == before[mistake["actual"]] - 1
    assert counts(other, "Red") == {"Red": 1}

    # Suggestions need a model trained on this dataset, not another one.
    assert client.post("/api/label/suggest", params={"dataset": other}, headers=JSON).status_code == 409
    client.post("/api/label/images", params={"dataset": other}, content=picture(BLUE, 700), headers=BINARY)
    assert client.get("/api/label/images", params={"dataset": other}).json()["modelCategories"] is None
    client.post("/api/label/images", params={"dataset": dataset}, content=picture(BLUE, 701), headers=BINARY)
    assert client.get("/api/label/images", params={"dataset": dataset}).json()["modelCategories"] == ["Blue", "Red"]

    # Deleting the dataset leaves the trained model usable.
    assert client.delete(f"/api/datasets/{dataset}").status_code == 200
    assert client.post(f"/api/train/{job['id']}/predict", content=picture(RED, 800), headers=BINARY).status_code == 200
    assert client.get(f"/api/train/{job['id']}/mistakes/0").status_code == 404


def test_imports_go_to_one_dataset_and_credit_only_it(fake_hub):  # noqa: F811
    target, other = create("Target")["id"], create("Other")["id"]
    reply = client.post("/api/open-data/import", params={"dataset": target}, json={
        "dataset": "open/cats", "kind": "image", "mapping": {"cat": "Red", "dog": "Blue"}, "perCategory": 12,
    })
    assert reply.status_code == 200
    deadline = time.time() + 30
    while time.time() < deadline:
        job = client.get("/api/open-data/import", params={"dataset": target}).json()["job"]
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job["status"] == "finished"
    assert counts(target, "Red", "Blue") == {"Red": 12, "Blue": 12} and counts(other, "Red") == {"Red": 0}
    assert client.get("/api/open-data/import", params={"dataset": other}).json() == {"job": None, "sources": []}
    assert client.get("/api/open-data/import", params={"dataset": target}).json()["sources"][0]["added"] == 24

    graph = build()
    mark_tested(graph)
    client.post("/api/train", params={"dataset": target}, json=graph)
    job = wait_for_end()
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/train/{job['id']}/download").content))
    assert "open/cats" in archive.read("colours/DATA_SOURCES.md").decode()


def test_projects_remember_their_dataset():
    project = {"projectName": "p", "training": "quick", "trainingOverride": None, "kitId": None, "datasetId": "abc", "nodes": [], "edges": []}
    client.put("/api/projects/one", json=project)
    assert client.get("/api/projects/one").json()["datasetId"] == "abc"
    del project["datasetId"]
    client.put("/api/projects/two", json=project)
    assert client.get("/api/projects/two").json()["datasetId"] is None
