"""Dataset versions, the dataset page, and models linked to the data that trained them."""
import io
import time
import zipfile

from fastapi.testclient import TestClient
from test_opendata import fake_hub  # noqa: F401
from test_training import BINARY, BLUE, RED, build, mark_tested, picture, wait_for_end, workspace  # noqa: F401

from aibricks import training
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


def create(kind="image", categories=("Red", "Blue")):
    return client.post("/api/datasets", json={"name": "Colours", "kind": kind, "categories": list(categories)}).json()["id"]


def add(dataset, category, data):
    assert client.post("/api/dataset/images", params={"dataset": dataset, "category": category}, content=data, headers=BINARY).status_code == 200


def fill(dataset, amount=10, start=0):
    for seed in range(start, start + amount):
        add(dataset, "Red", picture(RED, seed))
        add(dataset, "Blue", picture(BLUE, seed))


def save(dataset, message="A version"):
    reply = client.post(f"/api/datasets/{dataset}/versions", json={"message": message})
    assert reply.status_code == 200, reply.text
    return reply.json()


def overview(dataset):
    return client.get(f"/api/datasets/{dataset}/versions").json()


def summary(dataset):
    return client.get(f"/api/datasets/{dataset}").json()


def objects(workspace, dataset):  # noqa: F811
    return workspace / "collections" / dataset / "versions" / "objects"


def test_saving_needs_data_and_changes():
    dataset = create()
    assert overview(dataset) == {"versions": [], "head": None, "unsaved": None, "canSave": False}
    assert client.post(f"/api/datasets/{dataset}/versions", json={}).status_code == 422
    fill(dataset, 2)
    assert overview(dataset)["canSave"] is True
    first = save(dataset, "  First   batch ")
    assert first["number"] == 1 and first["message"] == "First batch" and first["labelled"] == 4
    assert first["counts"] == {"Red": 2, "Blue": 2} and first["changes"] is None and first["automatic"] is False
    assert overview(dataset)["canSave"] is False
    assert client.post(f"/api/datasets/{dataset}/versions", json={}).status_code == 409
    add(dataset, "Red", picture(RED, 50))
    latest = overview(dataset)
    assert latest["canSave"] is True and latest["unsaved"]["added"] == 1
    second = save(dataset)
    assert second["number"] == 2 and second["changes"]["added"] == 1
    assert [row["number"] for row in overview(dataset)["versions"]] == [2, 1]


def test_versions_share_files_with_the_dataset(workspace):  # noqa: F811
    dataset = create()
    add(dataset, "Red", picture(RED))
    save(dataset)
    kept = next(objects(workspace, dataset).iterdir())
    live = training._category_folder("Red", dataset) / kept.name
    assert kept.stat().st_ino == live.stat().st_ino


def test_compare_counts_additions_removals_and_label_changes():
    dataset = create()
    fill(dataset, 3)
    save(dataset)
    queue_item = picture(BLUE, 90)
    client.post("/api/label/images", params={"dataset": dataset}, content=queue_item, headers=BINARY)
    client.delete("/api/dataset/images", params={"dataset": dataset, "category": "Blue"})
    client.post(f"/api/datasets/{dataset}/categories", json={"names": ["Green"]})
    changes = client.get(f"/api/datasets/{dataset}/versions/1/compare").json()
    assert changes["added"] == 1 and changes["removed"] == 3 and changes["relabelled"] == 0
    assert changes["categoriesAdded"] == ["Green"]
    assert changes["counts"] == [{"name": "Blue", "before": 3, "after": 0}]

    client.put(f"/api/datasets/{dataset}/categories", json={"old": "Red", "new": "Crimson"})
    changes = client.get(f"/api/datasets/{dataset}/versions/1/compare").json()
    assert changes["relabelled"] == 3 and changes["categoriesRemoved"] == ["Red"]


def test_restore_brings_data_back_and_saves_unsaved_work_first():
    dataset = create()
    fill(dataset, 3)
    client.post("/api/label/images", params={"dataset": dataset}, content=picture(BLUE, 70), headers=BINARY)
    save(dataset, "Original")
    client.delete("/api/dataset/images", params={"dataset": dataset, "category": "Red"})
    client.post(f"/api/datasets/{dataset}/categories", json={"names": ["Green"]})
    add(dataset, "Green", picture(BLUE, 80))
    client.delete("/api/label/images", params={"dataset": dataset})

    reply = client.post(f"/api/datasets/{dataset}/versions/1/restore", headers=JSON).json()
    assert reply["savedFirst"] == 2
    restored = reply["dataset"]
    assert restored["categories"] == ["Red", "Blue"] and restored["counts"] == {"Red": 3, "Blue": 3} and restored["unsorted"] == 1
    assert client.get("/api/dataset", params={"dataset": dataset, "category": ["Green"]}).json()["counts"] == {"Green": 0}

    # After restoring, the dataset counts as being on that version, with nothing to save.
    latest = overview(dataset)
    assert latest["head"] == 1 and latest["canSave"] is False
    # The work from before the restore can be restored too.
    versions = overview(dataset)["versions"]
    assert versions[0]["automatic"] is True and "v1" in versions[0]["message"]
    back = client.post(f"/api/datasets/{dataset}/versions/2/restore", headers=JSON).json()
    assert back["savedFirst"] is None
    assert back["dataset"]["categories"] == ["Red", "Blue", "Green"]
    assert back["dataset"]["counts"] == {"Red": 0, "Blue": 3, "Green": 1} and back["dataset"]["unsorted"] == 0


def test_restore_leaves_other_folders_alone():
    dataset, other = create(), create()
    add(dataset, "Red", picture(RED))
    add(other, "Red", picture(RED, 5))
    save(dataset)
    # A model's own category can hold images even when the dataset doesn't list that category.
    add(dataset, "From a model", picture(BLUE))
    client.delete("/api/dataset/images", params={"dataset": dataset, "category": "Red"})
    assert client.post(f"/api/datasets/{dataset}/versions/1/restore", headers=JSON).status_code == 200
    amounts = client.get("/api/dataset", params={"dataset": dataset, "category": ["Red", "From a model"]}).json()["counts"]
    assert amounts == {"Red": 1, "From a model": 1}
    assert summary(other)["counts"]["Red"] == 1


def test_deleting_a_version_erases_files_only_it_kept(workspace):  # noqa: F811
    dataset = create()
    add(dataset, "Red", picture(RED, 1))
    add(dataset, "Red", picture(RED, 2))
    save(dataset)
    client.delete("/api/dataset/images", params={"dataset": dataset, "category": "Red"})
    add(dataset, "Red", picture(RED, 2))
    save(dataset)
    assert len(list(objects(workspace, dataset).iterdir())) == 2
    assert client.delete(f"/api/datasets/{dataset}/versions/1").status_code == 200
    assert len(list(objects(workspace, dataset).iterdir())) == 1
    assert client.post(f"/api/datasets/{dataset}/versions/1/restore", headers=JSON).status_code == 404
    assert client.delete(f"/api/datasets/{dataset}/versions/2").status_code == 200
    assert list(objects(workspace, dataset).iterdir()) == []
    # Version numbers aren't reused, so an old number can't come to mean different data.
    add(dataset, "Red", picture(RED, 3))
    assert save(dataset)["number"] == 3


def test_text_versions_restore_labels_and_order(workspace):  # noqa: F811
    dataset = create("text", ("Good", "Bad"))
    client.post("/api/label/text", params={"dataset": dataset, "kind": "lines"}, content=b"zebra one\napple two\nsecret three\n", headers=BINARY)
    def queue():
        return client.get("/api/label/text", params={"dataset": dataset, "category": ["Good", "Bad"]}).json()
    items = {row["text"]: row["id"] for row in queue()["items"]}
    client.post(f"/api/label/text/{items['zebra one']}/assign", params={"dataset": dataset}, json={"category": "Good"})
    first = save(dataset)
    assert first["labelled"] == 1 and first["unsorted"] == 2

    client.post(f"/api/label/text/{items['apple two']}/assign", params={"dataset": dataset}, json={"category": "Bad"})
    client.post(f"/api/label/text/{items['secret three']}/remove", params={"dataset": dataset}, headers=JSON)
    client.post(f"/api/label/text/{items['zebra one']}/unassign", params={"dataset": dataset}, headers=JSON)
    changes = overview(dataset)["unsaved"]
    assert changes["removed"] == 1 and changes["relabelled"] == 2

    reply = client.post(f"/api/datasets/{dataset}/versions/1/restore", headers=JSON).json()
    assert reply["savedFirst"] == 2 and reply["dataset"]["counts"] == {"Good": 1, "Bad": 0}
    assert [row["text"] for row in queue()["items"]] == ["apple two", "secret three"]
    export = client.get("/api/label/text/export", params={"dataset": dataset, "category": ["Good", "Bad"]}).text
    assert "zebra one,Good" in export

    # Deleting versions erases the text they were keeping, even after removal from the dataset.
    client.delete(f"/api/label/text", params={"dataset": dataset})
    client.delete(f"/api/datasets/{dataset}/versions/1")
    client.delete(f"/api/datasets/{dataset}/versions/2")
    assert b"secret three" not in (workspace / "collections" / dataset / "versions" / "texts.sqlite").read_bytes()


def test_bad_ids_are_refused(workspace):  # noqa: F811
    dataset = create()
    fill(dataset, 1)
    save(dataset)
    for bad in ["nope", "0" * 12]:
        assert client.get(f"/api/datasets/{bad}/versions").status_code == 404
        assert client.post(f"/api/datasets/{bad}/versions", json={}).status_code == 404
        assert client.get(f"/api/datasets/{bad}/models").status_code == 404
    assert client.get(f"/api/datasets/{dataset}/versions/99/compare").status_code == 404
    assert client.get(f"/api/datasets/{dataset}/versions/abc/compare").status_code == 422
    assert client.post(f"/api/datasets/{dataset}/versions/1/restore").status_code == 415
    assert client.delete(f"/api/datasets/{dataset}/versions/99").status_code == 404
    assert client.delete("/api/models/nope").status_code == 404
    assert client.delete("/api/models/..%2F..").status_code == 404
    assert client.get(f"/api/train/{'0' * 12}/download").status_code == 404
    for name in ["..%2Fdataset.json", "x.png", "index.json"]:
        assert client.get(f"/api/datasets/{dataset}/preview/{name}", params={"category": "Red"}).status_code == 404
    item = client.get(f"/api/datasets/{dataset}/preview").json()["categories"][0]["samples"][0]["id"]
    assert client.get(f"/api/datasets/{dataset}/preview/{item}", params={"category": "Unknown"}).status_code == 404


def test_dataset_page_details_and_preview():
    dataset = create()
    assert summary(dataset)["description"] == "" and summary(dataset)["licence"] == ""
    updated = client.put(f"/api/datasets/{dataset}/details", json={"description": " Red and blue squares. ", "licence": " CC0 "}).json()
    assert updated["description"] == "Red and blue squares." and updated["licence"] == "CC0"
    assert summary(dataset)["description"] == "Red and blue squares."
    assert client.put(f"/api/datasets/{dataset}/details", json={"description": "x" * 2001, "licence": ""}).status_code == 422
    fill(dataset, 10)
    preview = client.get(f"/api/datasets/{dataset}/preview").json()
    assert [row["name"] for row in preview["categories"]] == ["Red", "Blue"] and preview["sources"] == []
    assert len(preview["categories"][0]["samples"]) == 8
    item = preview["categories"][0]["samples"][0]["id"]
    image = client.get(f"/api/datasets/{dataset}/preview/{item}", params={"category": "Red"})
    assert image.status_code == 200 and image.headers["x-content-type-options"] == "nosniff"
    assert image.headers["cross-origin-resource-policy"] == "same-origin"

    text = create("text", ("Good",))
    client.post("/api/label/text", params={"dataset": text, "kind": "lines"}, content=b"lovely\n", headers=BINARY)
    item = client.get("/api/label/text", params={"dataset": text}).json()["items"][0]["id"]
    client.post(f"/api/label/text/{item}/assign", params={"dataset": text}, json={"category": "Good"})
    assert client.get(f"/api/datasets/{text}/preview").json()["categories"][0]["samples"][0]["text"] == "lovely"


def train(dataset):
    graph = build()
    mark_tested(graph)
    reply = client.post("/api/train", params={"dataset": dataset}, json=graph)
    assert reply.status_code == 200, reply.text
    return reply.json()["job"]


def test_models_link_to_the_exact_version_and_older_ones_are_kept(workspace):  # noqa: F811
    dataset = create()
    fill(dataset)
    started = train(dataset)
    assert started["datasetVersion"] == 1
    # Labelling during training doesn't change what this run is using.
    add(dataset, "Red", picture(RED, 200))
    first = wait_for_end()
    assert first["status"] == "finished" and first["datasetVersion"] == 1 and first["trainCount"] + first["checkCount"] == 20
    version = overview(dataset)["versions"][0]
    assert version["automatic"] is True and version["labelled"] == 20
    assert overview(dataset)["unsaved"]["added"] == 1
    # Results keep no copy of the pictures once the run ends.
    assert not (workspace / "runs" / first["id"] / "data").exists()
    assert client.post(f"/api/train/{first['id']}/predict", content=picture(RED, 300), headers=BINARY).status_code == 200

    # Training again makes a new version only because the data changed, and keeps the first model.
    second = train(dataset)
    assert second["datasetVersion"] == 2
    second = wait_for_end()
    third = train(dataset)
    assert third["datasetVersion"] == 2
    third = wait_for_end()
    assert len(overview(dataset)["versions"]) == 2
    assert client.get("/api/train/current").json()["job"]["id"] == third["id"]

    models = client.get(f"/api/datasets/{dataset}/models").json()["models"]
    assert [row["id"] for row in models] == [third["id"], second["id"], first["id"]]
    assert [row["datasetVersion"] for row in models] == [2, 2, 1]
    assert models[2]["modelBytes"] > 0 and "epochs" not in models[2] and "mistakes" not in models[2]["review"]
    older = client.get(f"/api/train/{first['id']}/download")
    assert older.status_code == 200
    with zipfile.ZipFile(io.BytesIO(older.content)) as archive:
        assert "colours/model.pt" in archive.namelist()
        assert not [name for name in archive.namelist() if name.endswith(".png")]
    assert client.get(f"/api/train/{first['id']}/mistakes/0").status_code == 404

    # Deleting one model leaves the others and doesn't change which is newest.
    assert client.delete(f"/api/models/{second['id']}").status_code == 200
    assert client.get(f"/api/train/current").json()["job"]["id"] == third["id"]
    assert len(client.get(f"/api/datasets/{dataset}/models").json()["models"]) == 2
    # Deleting the newest makes the previous one current again.
    assert client.delete(f"/api/models/{third['id']}").status_code == 200
    assert client.get("/api/train/current").json()["job"]["id"] == first["id"]


def test_only_the_newest_model_files_are_kept_but_records_stay(workspace, monkeypatch):  # noqa: F811
    monkeypatch.setattr(training, "MODELS_KEPT", 1)
    dataset = create()
    fill(dataset)
    train(dataset)
    first = wait_for_end()
    train(dataset)
    wait_for_end()
    models = client.get(f"/api/datasets/{dataset}/models").json()["models"]
    assert [row["modelRemoved"] for row in models] == [False, True]
    assert models[1]["id"] == first["id"] and models[1]["best"] is not None and models[1]["modelBytes"] == 0
    assert client.get(f"/api/train/{first['id']}/download").status_code == 409


def test_a_model_keeps_the_credits_for_its_own_version(fake_hub):  # noqa: F811
    dataset = create()
    reply = client.post("/api/open-data/import", params={"dataset": dataset}, json={
        "dataset": "open/cats", "kind": "image", "mapping": {"cat": "Red", "dog": "Blue"}, "perCategory": 12,
    })
    assert reply.status_code == 200
    deadline = time.time() + 30
    while time.time() < deadline and client.get("/api/open-data/import", params={"dataset": dataset}).json()["job"]["status"] == "running":
        time.sleep(0.1)
    assert client.post(f"/api/datasets/{dataset}/versions", json={}).status_code == 200
    assert client.get(f"/api/datasets/{dataset}/versions").json()["versions"][0]["sources"][0]["dataset"] == "open/cats"
    train(dataset)
    job = wait_for_end()
    assert client.delete(f"/api/datasets/{dataset}").status_code == 200
    with zipfile.ZipFile(io.BytesIO(client.get(f"/api/train/{job['id']}/download").content)) as archive:
        assert "open/cats" in archive.read("colours/DATA_SOURCES.md").decode()


def test_restore_is_refused_during_training():
    dataset = create()
    fill(dataset)
    train(dataset)
    assert client.post(f"/api/datasets/{dataset}/versions/1/restore", headers=JSON).status_code == 409
    assert client.delete(f"/api/datasets/{dataset}/versions/1").status_code == 409
    wait_for_end()
    assert client.post(f"/api/datasets/{dataset}/versions/1/restore", headers=JSON).status_code == 200
