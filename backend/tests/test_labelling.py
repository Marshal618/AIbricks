"""Sorting unlabelled images and text by hand, with model suggestions and label fixes."""
import csv
import io
import time

from fastapi.testclient import TestClient
from test_training import BINARY, BLUE, RED, build, fill, held_back_blues, mark_tested, picture, upload, wait_for_end, workspace  # noqa: F401

from aibricks import labelling, training
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


def inbox_add(data):
    return client.post("/api/label/images", content=data, headers=BINARY)


def counts(*names):
    return client.get("/api/dataset", params={"category": list(names)}).json()["counts"]


def text_build(classes="Billing, Bug"):
    bricks = [("text_input", {"vocab_size": 200, "max_length": 12}), ("embedding", {"size": 32}), ("lstm", {"hidden": 16}), ("last_step", {}), ("classify", {"classes": classes})]
    nodes = [{"id": f"n{i}", "type": t, "params": p} for i, (t, p) in enumerate(bricks)]
    edges = [{"source": f"n{i}", "target": f"n{i+1}", "targetHandle": "in"} for i in range(len(bricks) - 1)]
    return {"projectName": "Tickets", "training": "quick", "nodes": nodes, "edges": edges}


def test_image_label_undo_and_remove():
    assert inbox_add(picture(RED, 1)).json() == {"duplicate": False}
    assert inbox_add(picture(RED, 1)).json() == {"duplicate": True}
    assert inbox_add(picture(BLUE, 2)).json() == {"duplicate": False}
    assert inbox_add(b"nope").status_code == 422
    queue = client.get("/api/label/images").json()
    assert queue["total"] == 2 and len(queue["items"]) == 2
    first, second = (item["id"] for item in queue["items"])
    file = client.get(f"/api/label/images/{first}/file")
    assert file.status_code == 200 and file.headers["x-content-type-options"] == "nosniff"

    assert client.post(f"/api/label/images/{first}/assign", json={"category": "Red"}).status_code == 200
    assert counts("Red") == {"Red": 1}
    assert client.get("/api/label/images").json()["total"] == 1
    # A labelled image can't sneak back in as new.
    assert client.post("/api/label/images", content=(training._category_folder("Red") / first).read_bytes(), headers=BINARY).json()["duplicate"]

    assert client.post(f"/api/label/images/{first}/unassign", json={"category": "Red"}).status_code == 200
    assert counts("Red") == {"Red": 0} and client.get("/api/label/images").json()["total"] == 2

    assert client.post(f"/api/label/images/{second}/remove", headers=JSON).status_code == 200
    assert client.get("/api/label/images").json()["total"] == 1
    assert client.post(f"/api/label/images/{second}/unassign", json={"category": None}).status_code == 200
    assert client.get("/api/label/images").json()["total"] == 2

    assert client.delete("/api/label/images").status_code == 200
    assert client.get("/api/label/images").json()["total"] == 0


def test_labelled_images_train_like_any_others():
    for seed in range(12):
        for colour, name in ((RED, "Red"), (BLUE, "Blue")):
            inbox_add(picture(colour, seed))
    items = client.get("/api/label/images").json()["items"]
    for item in items:
        data = client.get(f"/api/label/images/{item['id']}/file").content
        from PIL import Image
        red, _, blue = Image.open(io.BytesIO(data)).convert("RGB").getpixel((1, 1))
        client.post(f"/api/label/images/{item['id']}/assign", json={"category": "Red" if red > blue else "Blue"})
    assert counts("Red", "Blue") == {"Red": 12, "Blue": 12}
    graph = build()
    mark_tested(graph)
    assert client.post("/api/train", json=graph).status_code == 200
    job = wait_for_end()
    assert job["status"] == "finished" and job["best"]["accuracy"] > 0.7


def test_suggestions_come_from_the_model_and_hardest_first():
    assert client.post("/api/label/suggest", headers=JSON).status_code == 409
    fill(20)
    graph = build()
    mark_tested(graph)
    client.post("/api/train", json=graph)
    job = wait_for_end()
    for seed in range(200, 204):
        inbox_add(picture(RED, seed))
        inbox_add(picture(BLUE, seed))
    assert client.get("/api/label/images").json()["unsuggested"] == 8
    assert client.get("/api/label/images").json()["modelCategories"] == ["Blue", "Red"]
    assert client.post("/api/label/suggest", headers=JSON).status_code == 200
    deadline = time.time() + 120
    while client.get("/api/label/images").json()["suggesting"] and time.time() < deadline:
        time.sleep(0.1)
    queue = client.get("/api/label/images").json()
    assert queue["suggestError"] is None and queue["unsuggested"] == 0
    chances = [item["suggestion"]["chance"] for item in queue["items"]]
    assert chances == sorted(chances)
    right = 0
    from PIL import Image
    for item in queue["items"]:
        data = client.get(f"/api/label/images/{item['id']}/file").content
        red, _, blue = Image.open(io.BytesIO(data)).convert("RGB").getpixel((1, 1))
        right += item["suggestion"]["category"] == ("Red" if red > blue else "Blue")
    assert right >= 7  # A model trained on 20 photos a side can miss one; it's checking the suggestions are real.

    # Suggestions belong to one model, so training again drops them.
    client.post("/api/train", json=graph)
    wait_for_end()
    assert client.get("/api/label/images").json()["unsuggested"] == 8


def test_undo_is_blocked_while_training_because_it_would_break_the_run():
    fill(6)
    inbox_add(picture(RED, 300))
    item = client.get("/api/label/images").json()["items"][0]["id"]
    client.post(f"/api/label/images/{item}/assign", json={"category": "Red"})
    graph = build(override={"epochs": 500})
    mark_tested(graph)
    job = client.post("/api/train", json=graph).json()["job"]
    assert client.post(f"/api/label/images/{item}/unassign", json={"category": "Red"}).status_code == 409
    # Labelling into a category is still safe.
    inbox_add(picture(BLUE, 301))
    new = client.get("/api/label/images").json()["items"][0]["id"]
    assert client.post(f"/api/label/images/{new}/assign", json={"category": "Blue"}).status_code == 200
    client.delete(f"/api/train/{job['id']}")
    wait_for_end()


def test_bad_ids_and_cross_site_requests_are_refused(workspace):  # noqa: F811
    for bad in ["..%2F..%2Fsecret.jpg", "abc", "0" * 32 + ".exe"]:
        assert client.get(f"/api/label/images/{bad}/file").status_code == 404
        assert client.post(f"/api/label/images/{bad}/assign", json={"category": "Red"}).status_code == 404
    # A plain form post from another site can't set JSON or binary content types.
    assert client.post("/api/label/images", content=picture(RED), headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.post("/api/label/text", content=b"hello", headers={"Content-Type": "text/plain"}).status_code == 415
    for path in ("/api/label/suggest", "/api/label/images/x/remove", "/api/label/text/x/remove", "/api/label/text/x/unassign"):
        assert client.post(path).status_code == 415, path
    assert client.post("/api/label/images/x/assign", content='{"category":"a"}', headers={"Content-Type": "text/plain"}).status_code == 422


def test_fixing_a_mistake_moves_removes_and_restores():
    fill(20)
    for data in held_back_blues(20):
        upload("Red", data)
    graph = build()
    mark_tested(graph)
    client.post("/api/train", json=graph)
    job = wait_for_end()
    mistake = job["review"]["mistakes"][0]
    actual, other = mistake["actual"], ({"Red", "Blue"} - {mistake["actual"]}).pop()
    start = counts("Red", "Blue")
    url = f"/api/train/{job['id']}/mistakes/0"

    moved = client.post(f"{url}/fix", json={"action": "move", "category": other})
    assert moved.json()["job"]["review"]["mistakes"][0]["movedTo"] == other
    assert counts(actual)[actual] == start[actual] - 1 and counts(other)[other] == start[other] + 1
    assert client.get(url).status_code == 200  # the picture still shows after moving

    assert client.post(f"{url}/fix", json={"action": "remove"}).json()["job"]["review"]["mistakes"][0]["removed"]
    assert counts(other)[other] == start[other] and client.get(url).status_code == 200

    restored = client.post(f"{url}/fix", json={"action": "undo"}).json()["job"]["review"]["mistakes"][0]
    assert not restored["removed"] and restored["movedTo"] is None
    assert counts("Red", "Blue") == start

    assert client.post(f"{url}/fix", json={"action": "move", "category": "Green"}).status_code == 422
    assert client.post(f"{url}/fix", json={"action": "explode"}).status_code == 422
    assert client.post(f"/api/train/{job['id']}/mistakes/9999/fix", json={"action": "remove"}).status_code == 404
    # Fixes survive a restart.
    client.post(f"{url}/fix", json={"action": "move", "category": other})
    training._job = None
    assert client.get("/api/train/current").json()["job"]["review"]["mistakes"][0]["movedTo"] == other


def test_undo_keeps_a_copy_that_was_already_in_the_target_category():
    fill(10)
    # The same photos sit in both categories, so moving one lands on an existing copy.
    for seed in range(10):
        assert upload("Red", picture(BLUE, seed)).status_code == 200
    graph = build()
    mark_tested(graph)
    client.post("/api/train", json=graph)
    job = wait_for_end()
    shared = [(i, m) for i, m in enumerate(job["review"]["mistakes"]) if (training._category_folder("Red") / m["file"]).exists() and (training._category_folder("Blue") / m["file"]).exists()]
    assert shared, job["review"]
    index, mistake = shared[0]
    other = "Blue" if mistake["actual"] == "Red" else "Red"
    start = counts("Red", "Blue")
    url = f"/api/train/{job['id']}/mistakes/{index}/fix"
    client.post(url, json={"action": "move", "category": other})
    assert counts(mistake["actual"])[mistake["actual"]] == start[mistake["actual"]] - 1
    assert counts(other)[other] == start[other]
    client.post(url, json={"action": "undo"})
    assert counts("Red", "Blue") == start
    # The same holds when a merged image is removed and brought back.
    client.post(url, json={"action": "move", "category": other})
    client.post(url, json={"action": "remove"})
    assert counts(other)[other] == start[other]
    client.post(url, json={"action": "undo"})
    assert counts("Red", "Blue") == start


def test_labelling_elsewhere_takes_an_image_out_of_the_inbox():
    inbox_add(picture(RED, 400))
    assert client.get("/api/label/images").json()["total"] == 1
    upload("Red", picture(RED, 400))
    assert client.get("/api/label/images").json()["total"] == 0


def add_text(content, kind="lines"):
    return client.post("/api/label/text", params={"kind": kind}, content=content.encode(), headers=BINARY)


def test_text_label_undo_remove_and_export():
    reply = add_text("My invoice is wrong\n\nThe app crashes\nMy invoice is wrong\n" + "x" * 5001 + '\nHe said "hi", then left\n=SUM(A1)\n').json()
    assert reply == {"added": 4, "duplicates": 1, "tooLong": 1, "column": None}
    queue = client.get("/api/label/text", params={"category": ["Billing", "Bug"]}).json()
    assert [i["text"] for i in queue["items"]][:2] == ["My invoice is wrong", "The app crashes"] and queue["total"] == 4
    ids = [item["id"] for item in queue["items"]]
    client.post(f"/api/label/text/{ids[0]}/assign", json={"category": "Billing"})
    client.post(f"/api/label/text/{ids[1]}/assign", json={"category": "Bug"})
    client.post(f"/api/label/text/{ids[2]}/assign", json={"category": "Bug"})
    client.post(f"/api/label/text/{ids[3]}/remove", headers=JSON)
    queue = client.get("/api/label/text", params={"category": ["Billing", "Bug"]}).json()
    assert queue["total"] == 0 and queue["counts"] == {"Billing": 1, "Bug": 2}

    client.post(f"/api/label/text/{ids[3]}/unassign", headers=JSON)
    client.post(f"/api/label/text/{ids[1]}/unassign", headers=JSON)
    assert client.get("/api/label/text", params={"category": ["Billing"]}).json()["total"] == 2
    client.post(f"/api/label/text/{ids[1]}/assign", json={"category": "Bug"})

    export = client.get("/api/label/text/export", params={"category": ["Billing", "Bug"]})
    assert "texts.csv" in export.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(export.text)))
    assert rows == [
        {"text": "My invoice is wrong", "label": "Billing"},
        {"text": "The app crashes", "label": "Bug"},
        {"text": 'He said "hi", then left', "label": "Bug"},
    ]
    # Labels for a category the model no longer has stay out of the file, and are counted.
    only = client.get("/api/label/text", params={"category": ["Billing"]}).json()
    assert only["otherLabels"] == 2
    assert len(list(csv.DictReader(io.StringIO(client.get("/api/label/text/export", params={"category": ["Billing"]}).text)))) == 1
    assert client.delete("/api/label/text").status_code == 200
    assert client.get("/api/label/text", params={"category": ["Billing"]}).json()["total"] == 0
    assert client.post("/api/label/text/nope/assign", json={"category": "Bug"}).status_code == 404


def test_text_csv_reading():
    assert add_text('id,Text,other\n1,"First, with comma",a\n2,"Line one\nline two",b\n3\n', "csv").json() == {
        "added": 2, "duplicates": 0, "tooLong": 0, "column": "Text"}
    assert add_text("message,x\nFallback column,1\n", "csv").json()["column"] == "message"
    assert client.post("/api/label/text", params={"kind": "csv"}, content="".encode(), headers=BINARY).status_code == 422
    assert client.post("/api/label/text", content=b"\xff\xfe\x00bad", headers=BINARY).status_code == 422


def test_exported_text_labels_train_in_the_generated_project(tmp_path):
    import subprocess
    import sys
    from aibricks.codegen import generate
    from aibricks.graph import GraphIn, analyse

    billing = [f"my invoice number {i} is wrong and the charge is high" for i in range(20)]
    bug = [f"the app crashes on screen {i} with an error message" for i in range(20)]
    add_text("\n".join(billing + bug))
    queue = client.get("/api/label/text", params={"category": ["Billing", "Bug"]}).json()
    for item in queue["items"]:
        client.post(f"/api/label/text/{item['id']}/assign", json={"category": "Billing" if "invoice" in item["text"] else "Bug"})
    payload = GraphIn(**text_build())
    analysis = analyse(payload)
    assert analysis.ok, analysis.issues
    files = generate(payload, analysis)
    for name, contents in files.items():
        (tmp_path / name).write_text(contents)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "texts.csv").write_text(client.get("/api/label/text/export", params={"category": ["Billing", "Bug"]}).text, newline="")
    done = subprocess.run([sys.executable, "train.py"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert "check accuracy" in done.stdout


def test_analysis_says_what_can_be_labelled():
    assert client.post("/api/analyse", json=build()).json()["labelKind"] == "image"
    reply = client.post("/api/analyse", json=text_build()).json()
    assert reply["labelKind"] == "text" and reply["labelCategories"] == ["Billing", "Bug"]
    assert reply["categories"] is None
    number = build()
    number["nodes"][2] = {"id": "n2", "type": "predict_number", "params": {}}
    assert client.post("/api/analyse", json=number).json()["labelKind"] is None
