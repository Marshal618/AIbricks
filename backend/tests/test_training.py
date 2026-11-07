"""Training on the user's own images inside the app, then asking the trained model."""
import io
import json
import random
import socket
import subprocess
import sys
import time
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aibricks import training
from aibricks.main import _passed, app
from aibricks.codegen import build_hash, generate
from aibricks.graph import GraphIn, analyse

client = TestClient(app)
BINARY = {"Content-Type": "application/octet-stream"}


def build(classes="Red, Blue", training_dial="quick", override=None):
    bricks = [
        ("image_folder", {"image_size": 32, "channels": 3}),
        ("cnn_encoder", {}),
        ("classify", {"classes": classes}),
    ]
    nodes = [{"id": f"n{i}", "type": t, "params": p} for i, (t, p) in enumerate(bricks)]
    edges = [{"source": f"n{i}", "target": f"n{i+1}", "targetHandle": "in"} for i in range(2)]
    return {"projectName": "Colours", "training": training_dial, "trainingOverride": override, "nodes": nodes, "edges": edges}


def mark_tested(graph):
    payload = GraphIn(**graph)
    _passed.add(build_hash(generate(payload, analyse(payload))))


def picture(colour, seed=0, kind="PNG"):
    rng = random.Random(seed)
    image = Image.new("RGB", (40, 40), tuple(min(255, c + rng.randint(0, 40)) for c in colour))
    buffer = io.BytesIO()
    image.save(buffer, kind)
    return buffer.getvalue()


RED, BLUE = (200, 20, 20), (20, 20, 200)


def upload(category, data):
    return client.post("/api/dataset/images", params={"category": category}, content=data, headers=BINARY)


def fill(amount=20):
    for seed in range(amount):
        assert upload("Red", picture(RED, seed)).status_code == 200
        assert upload("Blue", picture(BLUE, seed)).status_code == 200


def wait_for_end(limit=240):
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current").json()["job"]
        if job["status"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("training did not finish")


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_BRICKS_WORKSPACE", str(tmp_path))
    training._job = None
    training._process = None
    _passed.clear()
    yield tmp_path
    training.stop_any()
    process = training._process
    if process is not None:
        process.wait(30)
    time.sleep(0.05)
    training._job = None
    training._process = None


def test_train_try_and_download(workspace):
    fill()
    graph = build()
    mark_tested(graph)
    assert client.post("/api/train", json=graph).status_code == 200
    job = wait_for_end()
    assert job["status"] == "finished" and job["hasModel"]
    assert job["trainCount"] == 32 and job["checkCount"] == 8
    assert job["baseline"] == 0.5
    assert len(job["epochs"]) == job["totalEpochs"] > 0
    assert job["best"]["accuracy"] > 0.8
    assert job["best"]["checkLoss"] == min(e["checkLoss"] for e in job["epochs"])

    response = client.post(f"/api/train/{job['id']}/predict", content=picture(RED, 99), headers=BINARY)
    answers = response.json()["answers"]
    assert answers[0]["category"] == "Red"
    assert sum(a["chance"] for a in answers) == pytest.approx(1)
    answers = client.post(f"/api/train/{job['id']}/predict", content=picture(BLUE, 99, "JPEG"), headers=BINARY).json()["answers"]
    assert answers[0]["category"] == "Blue"

    # The download runs on its own and carries the model, but never the user's images.
    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(download.content))
    names = archive.namelist()
    assert "colours/model.pt" in names and "colours/predict.py" in names
    assert not any(n.endswith((".jpg", ".png")) or "/data/" in n for n in names)
    unpacked = workspace / "unpacked"
    archive.extractall(unpacked)
    (unpacked / "photo.png").write_bytes(picture(BLUE, 7))
    done = subprocess.run([sys.executable, "predict.py", "../photo.png"], cwd=unpacked / "colours", capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0].endswith("Blue")

    # The review agrees with the score, and its images can be read back.
    review = job["review"]
    assert review["total"] == 8
    assert review["correct"] / review["total"] == pytest.approx(job["best"]["accuracy"])
    assert sum(c["total"] for c in review["categories"].values()) == 8
    assert review["mistakeCount"] == review["total"] - review["correct"]

    # The downloaded web API answers real requests.
    api_folder = unpacked / "colours"
    (api_folder / "data").mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen([sys.executable, "serve.py", "--port", str(port)], cwd=api_folder, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(240):
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.25)
        reply = httpx.post(f"{base}/predict", content=picture(RED, 11), headers=BINARY, timeout=60)
        assert reply.status_code == 200 and reply.json()["best"] == "Red"
        assert httpx.post(f"{base}/predict", content=b"junk", headers=BINARY, timeout=60).status_code == 422
    finally:
        server.terminate()
        server.wait(30)

    # The result survives a server restart.
    training._job = None
    reloaded = client.get("/api/train/current").json()["job"]
    assert reloaded["id"] == job["id"] and reloaded["status"] == "finished" and reloaded["hasModel"]
    assert client.post(f"/api/train/{job['id']}/predict", content=picture(RED, 5), headers=BINARY).status_code == 200


def test_stop_keeps_best_so_far():
    fill()
    graph = build(override={"epochs": 500})
    mark_tested(graph)
    job = client.post("/api/train", json=graph).json()["job"]
    assert job["status"] == "running"
    assert client.post("/api/train", json=graph).status_code == 409
    assert client.delete("/api/dataset/images", params={"category": "Red"}).status_code == 409
    deadline = time.time() + 120
    while time.time() < deadline and not client.get("/api/train/current").json()["job"]["epochs"]:
        time.sleep(0.1)
    assert client.delete(f"/api/train/{job['id']}").status_code == 200
    ended = wait_for_end()
    assert ended["status"] == "stopped" and ended["hasModel"]
    assert client.post(f"/api/train/{job['id']}/predict", content=picture(RED, 1), headers=BINARY).status_code == 200


def test_stopping_before_first_epoch_has_no_model():
    fill(5)
    graph = build(override={"epochs": 500})
    mark_tested(graph)
    job = client.post("/api/train", json=graph).json()["job"]
    client.delete(f"/api/train/{job['id']}")
    ended = wait_for_end()
    assert ended["status"] == "stopped" and not ended["hasModel"]
    assert client.post(f"/api/train/{job['id']}/predict", content=picture(RED), headers=BINARY).status_code == 409
    assert client.get(f"/api/train/{job['id']}/download").status_code == 409


def test_a_file_damaged_after_upload_fails_with_a_message(workspace):
    fill(6)
    folder = training._category_folder("Red")
    for path in folder.iterdir():
        path.write_bytes(b"damaged")
    graph = build()
    mark_tested(graph)
    client.post("/api/train", json=graph)
    ended = wait_for_end()
    assert ended["status"] == "failed" and not ended["hasModel"]
    assert ended["error"]


def test_upload_rejects_what_it_cannot_read():
    assert upload("Red", b"not an image").status_code == 422
    assert upload("Red", b"....ftypheic....").json()["detail"]["message"].startswith("HEIC")
    assert upload("Red", b"<svg xmlns='http://www.w3.org/2000/svg'/>").status_code == 422
    good = picture(RED)
    assert upload("Red", good[: len(good) // 2]).status_code == 422
    assert upload("Red", b"x" * (training.MAX_IMAGE_BYTES + 1)).status_code == 413
    assert client.get("/api/dataset", params={"category": "Red"}).json()["counts"] == {"Red": 0}


def test_duplicates_count_once_and_clear_works():
    assert upload("Red", picture(RED)).json() == {"duplicate": False, "count": 1}
    assert upload("Red", picture(RED)).json() == {"duplicate": True, "count": 1}
    assert client.delete("/api/dataset/images", params={"category": "Red"}).status_code == 200
    assert client.get("/api/dataset", params={"category": "Red"}).json()["counts"] == {"Red": 0}


def test_other_websites_cannot_post_files_or_rebind():
    # A plain form post from another site can't set this content type without being blocked by the browser.
    for kind in ("text/plain", "multipart/form-data", "application/x-www-form-urlencoded"):
        response = client.post("/api/dataset/images", params={"category": "Red"}, content=picture(RED), headers={"Content-Type": kind})
        assert response.status_code == 415
    assert client.get("/api/catalogue", headers={"Host": "evil.example"}).status_code == 400


def test_category_names_cannot_escape_the_workspace(workspace):
    for name in ["../../escape", "/etc/passwd", "a/../../b", ".."]:
        assert upload(name, picture(RED, 3)).status_code == 200
    stored = {p.name for p in (workspace / "datasets").iterdir()}
    assert all(len(n) == 24 for n in stored)
    assert not (workspace.parent / "escape").exists()


def test_training_is_gated():
    fill(6)
    untested = build()
    assert client.post("/api/train", json=untested).status_code == 409

    # Too few images.
    graph = build(classes="Red, Blue, Green")
    mark_tested(graph)
    response = client.post("/api/train", json=graph)
    assert response.status_code == 422 and "Green" in response.json()["detail"]["message"]

    # A category name with a slash can't be a folder.
    graph = build(classes="Red/Dark, Blue")
    mark_tested(graph)
    response = client.post("/api/train", json=graph)
    assert response.status_code == 422 and "Rename" in response.json()["detail"]["message"]

    # Other tasks use the export instead.
    number = build()
    number["nodes"][2] = {"id": "n2", "type": "predict_number", "params": {}}
    assert client.post("/api/analyse", json=number).json()["categories"] is None
    assert client.post("/api/train", json=number).status_code == 422
    assert client.post("/api/analyse", json=build()).json()["categories"] == ["Red", "Blue"]


def test_predict_on_unknown_run():
    assert client.post("/api/train/nope/predict", content=picture(RED), headers=BINARY).status_code == 404
    assert client.get("/api/train/nope/download").status_code == 404
    assert client.delete("/api/train/nope").status_code == 404


def test_only_image_sorting_builds_get_predict_py():
    number = build()
    number["nodes"][2] = {"id": "n2", "type": "predict_number", "params": {}}
    payload = GraphIn(**number)
    assert "predict.py" not in generate(payload, analyse(payload))
    payload = GraphIn(**build())
    files = generate(payload, analyse(payload))
    for name in ("predict.py", "review.py", "serve.py"):
        assert name in files and name in files["README.md"]
    assert "fastapi" in files["requirements.txt"]
    json.dumps(files)


def test_mistakes_show_the_right_image_and_vanish_with_removal(workspace):
    fill(10)
    # Half of the Red images are really blue, so the model has to get some wrong.
    for seed in range(100, 110):
        assert upload("Red", picture(BLUE, seed)).status_code == 200
    graph = build()
    mark_tested(graph)
    client.post("/api/train", json=graph)
    job = wait_for_end()
    review = job["review"]
    assert job["status"] == "finished" and review["mistakeCount"] > 0, job
    chances = [m["chance"] for m in review["mistakes"]]
    assert chances == sorted(chances, reverse=True)
    first = review["mistakes"][0]
    image = client.get(f"/api/train/{job['id']}/mistakes/0")
    assert image.status_code == 200 and image.headers["x-content-type-options"] == "nosniff"
    assert image.headers["cross-origin-resource-policy"] == "same-origin"
    Image.open(io.BytesIO(image.content)).load()
    assert (training._category_folder(first["actual"]) / first["file"]).read_bytes() == image.content
    assert client.get(f"/api/train/{job['id']}/mistakes/9999").status_code == 404
    assert client.get(f"/api/train/{job['id']}/mistakes/-1").status_code == 404
    client.delete("/api/dataset/images", params={"category": first["actual"]})
    assert client.get(f"/api/train/{job['id']}/mistakes/0").status_code == 404
    # Stored results don't keep a copy of the pictures.
    assert not [p for p in workspace.joinpath("runs").rglob("*") if p.suffix in {".jpg", ".png"} and not p.is_symlink() and "images" not in p.parts]


def project(name="Saved model"):
    return {"projectName": name, "training": "quick", "trainingOverride": None, "kitId": None,
            "nodes": [{"id": "a", "type": "brick", "position": {"x": 1, "y": 2}, "data": {"brickId": "image_folder", "params": {}}}], "edges": []}


def test_projects_save_list_open_delete():
    assert client.get("/api/projects").json() == {"projects": []}
    assert client.put("/api/projects/one", json=project("First")).status_code == 200
    time.sleep(0.01)
    assert client.put("/api/projects/two", json=project("Second")).status_code == 200
    rows = client.get("/api/projects").json()["projects"]
    assert [r["projectName"] for r in rows] == ["Second", "First"] and rows[0]["components"] == 1
    assert client.get("/api/projects/one").json()["nodes"][0]["position"] == {"x": 1, "y": 2}
    client.put("/api/projects/one", json=project("First, renamed"))
    assert client.get("/api/projects").json()["projects"][0]["projectName"] == "First, renamed"
    assert client.delete("/api/projects/one").status_code == 200
    assert client.get("/api/projects/one").status_code == 404


def test_project_ids_cannot_escape_and_damage_is_contained(workspace):
    for bad in ["..%2F..%2Fescape", "a.b", "x" * 41]:
        assert client.put(f"/api/projects/{bad}", json=project()).status_code in {404, 422}
        assert client.get(f"/api/projects/{bad}").status_code == 404
    assert not list(workspace.rglob("escape*"))
    client.put("/api/projects/good", json=project())
    (workspace / "projects" / "broken.json").write_text("{not json")
    assert [r["id"] for r in client.get("/api/projects").json()["projects"]] == ["good"]
    assert client.get("/api/projects/broken").status_code == 404
    assert client.put("/api/projects/big", json=project("x" * 201)).status_code == 422
    # Plain form posts from other sites can't save, because JSON and PUT need permission first.
    assert client.post("/api/projects/good", content="{}", headers={"Content-Type": "text/plain"}).status_code == 405
