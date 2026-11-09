"""Spreadsheet datasets, the checks that keep scores honest, and models trained from them."""
import csv
import io
import os
import random
import subprocess
import sys
import time
import zipfile

from fastapi.testclient import TestClient
from test_training import BINARY, workspace  # noqa: F401

from aibricks import tables, training
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


def to_csv(header, rows, delimiter=","):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode()


def houses(amount=300, seed=1):
    """House prices that depend on size and area, with columns that should be left out."""
    rng = random.Random(seed)
    rows = []
    for index in range(amount):
        size, area = rng.randint(40, 300), rng.choice(["City", "Coast", "Rural"])
        price = size * 3000 + {"City": 200000, "Coast": 120000, "Rural": 0}[area] + rng.randint(-30000, 30000)
        band = "High" if price > 600000 else "Low"
        rows.append([index + 1, size, area, f"Owner {rng.randint(0, 10**9)}", f"20{10 + index % 10}-01-15", price * 1.0001, "yes", band, price])
    return ["id", "size", "area", "owner", "listed", "price copy", "always", "band", "price"], rows


def create(name="Houses"):
    return client.post("/api/datasets", json={"name": name, "kind": "table"}).json()["id"]


def upload(dataset, data):
    return client.post(f"/api/datasets/{dataset}/table", content=data, headers=BINARY)


def analysis(dataset, **params):
    return client.get(f"/api/datasets/{dataset}/table/analysis", params=params).json()


def wait(limit=120):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": "table"}).json()["job"]
        if job["status"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("training did not finish")


def train(dataset, target, task, columns):
    reply = client.post(f"/api/datasets/{dataset}/train-table", json={"target": target, "task": task, "columns": columns})
    assert reply.status_code == 200, reply.text
    job = wait()
    assert job["status"] == "finished", job["error"]
    return job


def test_upload_reads_common_csv_shapes_and_refuses_bad_ones():
    dataset = create()
    reply = upload(dataset, b"\xef\xbb\xbfa;b\n1;2\n3;4\n\n5;6;7\n")
    assert reply.status_code == 200
    assert reply.json()["dataset"]["columns"] == ["a", "b"] and reply.json()["dataset"]["labelled"] == 2 and reply.json()["damagedRows"] == 1
    preview = client.get(f"/api/datasets/{dataset}/preview").json()["table"]
    assert preview["columns"] == ["a", "b"] and preview["rows"][0] == ["1", "2"]
    for bad in (b"", b"onlyone\n1\n", b"a,b\n", b"a,a\n1,2\n", b"a,\n1,2\n", b"\xff\xfe\x00bad"):
        assert upload(dataset, bad).status_code == 422, bad
    assert client.post(f"/api/datasets/{dataset}/table", content=b"a,b\n1,2\n", headers=JSON).status_code == 415
    assert client.post("/api/datasets/nope/table", content=b"a,b\n1,2\n", headers=BINARY).status_code == 404
    images = client.post("/api/datasets", json={"name": "Pics", "kind": "image"}).json()["id"]
    assert upload(images, b"a,b\n1,2\n").status_code == 422
    assert client.get(f"/api/datasets/{images}/table/analysis").status_code == 422
    assert client.post(f"/api/datasets/{dataset}/categories", json={"names": ["x"]}).status_code == 422
    assert client.post("/api/datasets", json={"name": "T", "kind": "table", "categories": ["x"]}).status_code == 422


def test_checks_leave_out_columns_that_would_mislead():
    dataset = create()
    assert client.get(f"/api/datasets/{dataset}/table/analysis").status_code == 409
    upload(dataset, to_csv(*houses()))
    result = analysis(dataset)
    assert result["target"] == "price" and result["task"] == "number" and result["rows"] == 300 and result["problems"] == []
    columns = {column["name"]: column for column in result["columns"]}
    assert "price" not in columns
    assert columns["size"]["use"] is True and columns["size"]["kind"] == "number"
    assert columns["area"]["use"] is True and columns["area"]["kind"] == "category"
    for name, phrase in (("id", "row number"), ("owner", "memorise"), ("listed", "Dates"), ("always", "same value"), ("price copy", "gives away")):
        assert columns[name]["use"] is False and phrase in columns[name]["warning"], name

    # A category that fully decides another column is flagged when predicting that column.
    band = analysis(dataset, target="band")
    assert band["task"] == "category" and set(band["classes"]) == {"High", "Low"}
    assert {column["name"]: column for column in band["columns"]}["price"]["use"] is False


def test_target_problems_are_explained():
    dataset = create()
    rows = [[i, f"name{i}"] for i in range(100)]
    upload(dataset, to_csv(["number", "name"], rows))
    assert "too many" in analysis(dataset, target="name", task="category")["problems"][0]
    assert "doesn't hold numbers" in analysis(dataset, target="name", task="number")["problems"][0]
    assert client.get(f"/api/datasets/{dataset}/table/analysis", params={"target": "unknown"}).status_code == 422
    upload(dataset, to_csv(["a", "b"], [[i, "x"] for i in range(30)]))
    assert "at least 2" in analysis(dataset, target="b")["problems"][0]
    upload(dataset, to_csv(["a", "b"], [[i, "x" if i % 2 else "y"] for i in range(10)]))
    assert "at least 20 rows" in analysis(dataset, target="b")["problems"][0]
    # Rare answers and rows without an answer are left out, and the page is told.
    rows = [[i, "x" if i % 2 else "y"] for i in range(40)] + [[50, "rare"], [51, ""]]
    upload(dataset, to_csv(["a", "b"], rows))
    result = analysis(dataset, target="b", task="category")
    assert result["rows"] == 40 and set(result["classes"]) == {"x", "y"} and "fewer than 5" in result["notes"][0]


def test_predicting_a_number_links_a_version_and_runs_on_its_own(workspace):  # noqa: F811
    dataset = create()
    upload(dataset, to_csv(*houses()))
    job = train(dataset, "price", "number", ["size", "area"])
    assert job["task"] == "table" and job["datasetVersion"] == 1 and job["hasModel"] is True
    metrics = job["metrics"]
    assert metrics["task"] == "number" and metrics["r2"] > 0.8 and metrics["mae"] < metrics["baselineMae"] / 2
    assert metrics["importance"][0]["name"] == "size" and abs(sum(row["share"] for row in metrics["importance"]) - 1) < 0.01
    # Results hold no rows of data, and the run keeps no copy of the table.
    run = workspace / "runs" / job["id"]
    assert not (run / "data").exists() and b"Owner " not in (run / "job.json").read_bytes()

    models = client.get(f"/api/datasets/{dataset}/models").json()["models"]
    assert models[0]["metrics"]["r2"] == metrics["r2"] and models[0]["modelBytes"] > 0
    schema = client.get(f"/api/models/{job['id']}/schema").json()
    assert [item["name"] for item in schema["features"]] == ["size", "area"]
    assert set(schema["features"][1]["choices"]) == {"City", "Coast", "Rural"}

    small = client.post(f"/api/models/{job['id']}/try-row", json={"values": {"size": "50", "area": "Rural"}}).json()["value"]
    large = client.post(f"/api/models/{job['id']}/try-row", json={"values": {"size": "290", "area": "City"}}).json()["value"]
    assert large > small * 2
    # Blank and unknown values still get an answer rather than an error.
    assert "value" in client.post(f"/api/models/{job['id']}/try-row", json={"values": {"size": "", "area": "Mars"}}).json()

    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        names = archive.namelist()
        assert "houses/model/model.joblib" in names and "houses/config.json" in names and not [name for name in names if "/data/" in name]
        archive.extractall(workspace / "download")
    project = workspace / "download" / "houses"
    (project / "new.csv").write_text("size,area\n100,City\n250,Coast\n", encoding="utf-8")
    runner = "import runpy,sys; sys.path.insert(0, '.'); sys.argv=['predict.py','new.csv']; runpy.run_path('predict.py', run_name='__main__')"
    done = subprocess.run([sys.executable, "-I", "-c", runner], cwd=project, capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
    assert done.returncode == 0, done.stderr
    assert len(done.stdout.split()) == 2


def test_predicting_a_category_reports_against_guessing():
    dataset = create()
    upload(dataset, to_csv(*houses()))
    job = train(dataset, "band", "category", ["size", "area"])
    metrics = job["metrics"]
    assert metrics["task"] == "category" and metrics["accuracy"] > metrics["baseline"]
    assert job["best"]["accuracy"] == metrics["accuracy"] or abs(job["best"]["accuracy"] - metrics["accuracy"]) < 0.001
    assert job["review"]["total"] == metrics["total"] and set(job["review"]["categories"]) == {"High", "Low"}
    answers = client.post(f"/api/models/{job['id']}/try-row", json={"values": {"size": "290", "area": "City"}}).json()["answers"]
    assert answers[0]["category"] == "High" and abs(sum(row["chance"] for row in answers) - 1) < 0.01


def test_leak_is_warned_about_after_training_if_the_user_keeps_it():
    dataset = create()
    upload(dataset, to_csv(*houses()))
    job = train(dataset, "price", "number", ["size", "area", "price copy"])
    assert job["metrics"]["warnings"] and "price copy" in job["metrics"]["warnings"][0]


def test_repeated_rows_cant_inflate_the_score():
    dataset = create()
    header, rows = houses(100)
    upload(dataset, to_csv(header, rows * 5))
    job = train(dataset, "price", "number", ["size", "area", "owner"])
    assert job["metrics"]["repeatedRows"] == 400 and job["metrics"]["total"] == 20


def test_different_rows_that_match_on_chosen_columns_are_kept():
    dataset = create()
    rows = [[f"Person {index}", index % 2, "a" if index % 2 else "b"] for index in range(100)]
    upload(dataset, to_csv(["name", "group", "answer"], rows))
    job = train(dataset, "answer", "category", ["group"])
    assert job["metrics"]["repeatedRows"] == 0 and job["metrics"]["total"] == 20


def test_replacing_saves_first_and_versions_restore(workspace):  # noqa: F811
    dataset = create()
    first = to_csv(["a", "b"], [[i, i * 2] for i in range(30)])
    upload(dataset, first)
    assert client.get(f"/api/datasets/{dataset}/versions").json()["canSave"] is True
    assert client.post(f"/api/datasets/{dataset}/versions", json={"message": "First"}).json()["counts"] == {"Rows": 30}

    # Nothing unsaved, so replacing needs no extra version. The next replacement does.
    reply = upload(dataset, to_csv(["a", "b"], [[i, i * 2] for i in range(40)])).json()
    assert reply["savedFirst"] is None
    changes = client.get(f"/api/datasets/{dataset}/versions").json()["unsaved"]
    assert changes["added"] == 10 and changes["removed"] == 0
    reply = upload(dataset, to_csv(["x", "y", "z"], [[1, 2, 3]])).json()
    assert reply["savedFirst"] == 2

    restored = client.post(f"/api/datasets/{dataset}/versions/1/restore", headers=JSON).json()
    assert restored["dataset"]["labelled"] == 30 and restored["dataset"]["columns"] == ["a", "b"]
    assert client.get(f"/api/datasets/{dataset}/preview").json()["table"]["columns"] == ["a", "b"]
    assert analysis(dataset)["totalRows"] == 30

    # Deleting every version erases their copies of the table.
    for number in (3, 2, 1):
        assert client.delete(f"/api/datasets/{dataset}/versions/{number}").status_code in {200, 404}
    assert not list((workspace / "collections" / dataset / "versions" / "tables").glob("*"))


def test_training_refuses_bad_requests():
    dataset = create()
    assert client.post(f"/api/datasets/{dataset}/train-table", json={"target": "a", "task": "number", "columns": ["b"]}).status_code == 409
    upload(dataset, to_csv(*houses()))
    for bad in (
        {"target": "price", "task": "number", "columns": []},
        {"target": "price", "task": "number", "columns": ["nope"]},
        {"target": "price", "task": "number", "columns": ["price"]},
        {"target": "nope", "task": "number", "columns": ["size"]},
        {"target": "area", "task": "number", "columns": ["size"]},
    ):
        assert client.post(f"/api/datasets/{dataset}/train-table", json=bad).status_code == 422, bad
    assert client.post(f"/api/datasets/{dataset}/train-table", json={"target": "price", "task": "other", "columns": ["size"]}).status_code == 422
    assert client.post(f"/api/models/{'0' * 12}/try-row", json={"values": {}}).status_code == 404
    assert client.get("/api/models/nope/schema").status_code == 404


def test_odd_column_names_are_data_not_code():
    dataset = create()
    bad = 'x"; import os; os.system("touch hacked") #'
    rows = [[i, "a" if i % 2 else "b", i % 3] for i in range(60)]
    upload(dataset, to_csv([bad, "answer", "{other}"], rows))
    job = train(dataset, "answer", "category", ["{other}"])
    assert job["status"] == "finished"
    assert not list(training.workspace().rglob("hacked"))
    assert isinstance(tables.schema(job["id"]), dict)
