"""Every socket and head combination: generate, pass the smoke test, then train on real files."""
import csv
import io
import math
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aibricks.catalogue import KITS
from aibricks.codegen import generate
from aibricks.graph import GraphIn, analyse
from aibricks.main import app

client = TestClient(app)


def chain(*bricks, name="Test build", training="quick"):
    nodes = [{"id": f"n{i}", "type": t, "params": p} for i, (t, p) in enumerate(bricks)]
    edges = [{"source": f"n{i}", "target": f"n{i+1}", "targetHandle": "in"} for i in range(len(bricks) - 1)]
    return {"projectName": name, "training": training, "nodes": nodes, "edges": edges}


IMG = ("image_folder", {"image_size": 32, "channels": 3})
SER = ("time_series", {"columns": "a, b", "sequence_length": 12})
ENC = ("cnn_encoder", {})
LE = ("lstm_encoder", {"hidden": 16})

BUILDS = {
    "image_classify": chain(IMG, ENC, ("classify", {"classes": "Good part, Bad-part"})),
    "image_number": chain(IMG, ENC, ("predict_number", {})),
    "image_anomaly": chain(IMG, ENC, ("cnn_decoder", {"filters": 8}), ("detect_anomalies", {})),
    "series_classify": chain(SER, LE, ("classify", {"classes": "up, down"})),
    "series_number": chain(SER, LE, ("predict_number", {})),
    "series_forecast": chain(SER, LE, ("forecast", {"horizon": 4})),
    "series_seq2seq": chain(SER, LE, ("lstm_decoder", {"hidden": 16, "steps": 4}), ("forecast", {"horizon": 4})),
    "series_anomaly": chain(SER, LE, ("lstm_decoder", {"hidden": 16}), ("detect_anomalies", {})),
    "gru_and_rnn": chain(SER, ("gru", {"hidden": 16, "bidirectional": True}), ("rnn", {"hidden": 16}), ("last_step", {}), ("dropout", {"percent": 10}), ("dense", {"size": 16}), ("predict_number", {})),
    "pretrained": chain(IMG, ("backbone", {"freeze": True}), ("residual", {}), ("global_pool", {}), ("classify", {"classes": "OK, Scratch, Dent"})),
    "upsample_flatten": chain(IMG, ("conv", {"filters": 4, "kernel_size": 5}), ("pool", {"mode": "average"}), ("upsample", {"filters": 4}), ("detect_anomalies", {})),
    "big_complete": chain(IMG, ("image_recogniser", {"classes": "a, b, c"})),
}


def run_events(build):
    response = client.post("/api/test-run", json=build)
    assert response.status_code == 200, response.text
    import json
    return [json.loads(line) for line in response.text.splitlines()]


@pytest.mark.parametrize("name", BUILDS)
def test_smoke_test_passes(name):
    events = run_events(BUILDS[name])
    result = events[-1]
    assert result == {"event": "result", "passed": True}, events[-3:]
    checks = {e["check"]: e["passed"] for e in events if e["event"] == "check"}
    assert checks == {"connections": True, "shapes": True, "loss": True}
    assert sum(e["event"] == "loss" for e in events) == 60


def _write_images(root: Path, count=6, size=20):
    root.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (size, size), (i * 30, 40, 200 - i * 20)).save(root / f"img_{i}.png")


def _write_series(path: Path, rows=120, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["a", "b"] + ([extra] if extra else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for i in range(rows):
            row = {"a": math.sin(i / 4), "b": random.random()}
            if extra == "label":
                row[extra] = "up" if i % 2 else "down"
            elif extra == "target":
                row[extra] = i % 7
            w.writerow(row)


DATA_SETUP = {
    "image_classify": lambda d: (_write_images(d / "data/images/Good part"), _write_images(d / "data/images/Bad-part")),
    "image_number": lambda d: (_write_images(d / "data/images"), (d / "data/labels.csv").write_text("filename,value\n" + "".join(f"img_{i}.png,{i * 10}\n" for i in range(6)))),
    "image_anomaly": lambda d: _write_images(d / "data/images"),
    "series_classify": lambda d: _write_series(d / "data/series.csv", extra="label"),
    "series_number": lambda d: _write_series(d / "data/series.csv", extra="target"),
    "series_forecast": lambda d: _write_series(d / "data/series.csv"),
    "series_seq2seq": lambda d: _write_series(d / "data/series.csv"),
    "series_anomaly": lambda d: _write_series(d / "data/series.csv"),
}


@pytest.mark.parametrize("name", DATA_SETUP)
def test_generated_project_trains_on_real_files(name, tmp_path):
    graph = GraphIn(**BUILDS[name])
    analysis = analyse(graph)
    assert analysis.ok, analysis.issues
    for filename, contents in generate(graph, analysis).items():
        (tmp_path / filename).write_text(contents)
    DATA_SETUP[name](tmp_path)
    done = subprocess.run([sys.executable, "train.py"], cwd=tmp_path, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "model.pt").exists()
    import torch
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)
    assert "model" in checkpoint
    if name.endswith("anomaly"):
        assert checkpoint["anomaly_threshold"] > 0


def test_generated_files_compile_and_mark_the_data_slot():
    for build in BUILDS.values():
        graph = GraphIn(**build)
        files = generate(graph, analyse(graph))
        for name, contents in files.items():
            if name.endswith(".py"):
                compile(contents, name, "exec")
        assert "PUT YOUR DATA HERE" in files["data.py"]


def test_export_requires_a_passed_test():
    build = chain(IMG, ENC, ("classify", {"classes": "never, tested"}))
    assert client.post("/api/export", json=build).status_code == 409
    assert run_events(build)[-1]["passed"]
    response = client.post("/api/export", json=build)
    assert response.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert "test_build/model.py" in names and "test_build/README.md" in names
    changed = chain(IMG, ENC, ("classify", {"classes": "never, tested, changed"}))
    assert client.post("/api/export", json=changed).status_code == 409


@pytest.mark.parametrize("bricks, fragment", [
    ((IMG, ("lstm", {})), "takes Time series"),
    ((SER, ("conv", {})), "takes Image"),
    ((IMG, ("classify", {"classes": "a, b"})), "takes Summary"),
    ((IMG, ENC, ("forecast", {})), "needs a Time series input"),
    ((SER, LE, ("cnn_decoder", {}), ("detect_anomalies", {})), "needs an Image folder input"),
    ((("image_folder", {"channels": 1, "image_size": 64}), ("backbone", {}), ("global_pool", {}), ("predict_number", {})), "3 colour channels"),
    ((("image_folder", {"image_size": 16}), ("pool", {}), ("pool", {}), ("pool", {}), ("pool", {}), ("pool", {}), ("global_pool", {}), ("predict_number", {})), "can't be pooled"),
    ((IMG, ("conv", {"filters": 2000}),), "between 1 and 1024"),
])
def test_invalid_builds_explain_themselves(bricks, fragment):
    analysis = analyse(GraphIn(**chain(*bricks)))
    assert not analysis.ok
    assert any(fragment in i.message for i in analysis.issues), analysis.issues


def test_loop_and_dangling_builds_are_rejected():
    build = chain(IMG, ("conv", {}), ("conv", {}), ("global_pool", {}), ("predict_number", {}))
    build["edges"].append({"source": "n2", "target": "n1", "targetHandle": "in"})
    assert not analyse(GraphIn(**build)).ok
    assert any("is not connected" in i.message for i in analyse(GraphIn(**chain(IMG, ("conv", {})))).issues)


def test_names_cannot_inject_code():
    for evil in ['a"); import os; os.system("x"); ("', "a\nimport os", "'''", "{os.system('x')}", "a\\"]:
        analysis = analyse(GraphIn(**chain(IMG, ENC, ("classify", {"classes": f"{evil}, b"}))))
        assert not analysis.ok, evil
        analysis = analyse(GraphIn(**chain(("time_series", {"columns": evil}), LE, ("predict_number", {}))))
        assert not analysis.ok, evil
    assert not analyse(GraphIn(**chain(IMG, ENC, ("predict_number", {}), name='x"""; import os'))).ok


def test_parallel_branches_with_combine():
    build = {
        "projectName": "Two views", "training": "quick",
        "nodes": [
            {"id": "s", "type": "time_series", "params": {"columns": "a", "sequence_length": 12}},
            {"id": "l", "type": "lstm", "params": {"hidden": 8}}, {"id": "g", "type": "gru", "params": {"hidden": 12}},
            {"id": "ll", "type": "last_step"}, {"id": "gl", "type": "last_step"}, {"id": "c", "type": "combine"},
            {"id": "h", "type": "predict_number"},
        ],
        "edges": [
            {"source": "s", "target": "l"}, {"source": "s", "target": "g"},
            {"source": "l", "target": "ll"}, {"source": "g", "target": "gl"},
            {"source": "ll", "target": "c", "targetHandle": "a"}, {"source": "gl", "target": "c", "targetHandle": "b"},
            {"source": "c", "target": "h"},
        ],
    }
    assert analyse(GraphIn(**build)).shapes["c"] == "summary of 20 numbers"
    assert run_events(build)[-1]["passed"]


def test_kits_start_half_built_and_complete_as_instructed():
    finish = {
        "defect_detector": ["global_pool", "classify"],
        "energy_forecaster": ["last_step", "forecast"],
        "sensor_watchdog": ["detect_anomalies"],
        "ticket_sorter": ["mean_pool", "classify"],
        "text_generator": ["generate"],
        "writer_tuner": ["generate"],
        "translator": ["generate"],
        "vit_classifier": ["classify"],
    }
    for kit in KITS:
        nodes = [{"id": n["id"], "type": n["type"], "params": n["params"]} for n in kit["nodes"]]
        edges = list(kit["edges"])
        assert not analyse(GraphIn(nodes=nodes, edges=edges)).ok
        previous = nodes[-1]["id"]
        for i, brick in enumerate(finish[kit["id"]]):
            nodes.append({"id": f"x{i}", "type": brick, "params": {"classes": "OK, Scratch, Dent"} if brick == "classify" else {}})
            edges.append({"source": previous, "target": f"x{i}", "targetHandle": "in"})
            previous = f"x{i}"
        build = {"projectName": kit["projectName"], "training": kit["training"], "nodes": nodes, "edges": edges}
        assert run_events(build)[-1]["passed"], kit["id"]


def test_big_brick_issues_use_the_name_on_the_canvas():
    messages = [i.message for i in analyse(GraphIn(**chain(ENC, ("predict_number", {})))).issues]
    assert "CNN encoder has no input connected." in messages


def test_shape_descriptions_read_naturally():
    shapes = analyse(GraphIn(**chain(("time_series", {"columns": "a", "sequence_length": 2}), LE, ("predict_number", {})))).shapes
    assert shapes["n0"] == "2 steps, 1 value each"


# ---------------------------------------------------------------------- transformers and NLP

TXT = ("text_input", {"vocab_size": 200, "max_length": 12})
EMB = ("embedding", {"size": 32})
POS = ("positional", {})


def translator(memory_size=32):
    nodes = [
        {"id": "s", "type": "text_input", "params": {"vocab_size": 200, "max_length": 10}},
        {"id": "se", "type": "embedding", "params": {"size": memory_size}},
        {"id": "sp", "type": "positional"},
        {"id": "e", "type": "transformer_encoder", "params": {"heads": 2, "layers": 1}},
        {"id": "t", "type": "target_text", "params": {"vocab_size": 150, "max_length": 8}},
        {"id": "te", "type": "embedding", "params": {"size": 32}},
        {"id": "tp", "type": "positional"},
        {"id": "d", "type": "transformer_decoder", "params": {"heads": 2, "layers": 1}},
        {"id": "g", "type": "generate"},
    ]
    edges = [
        {"source": "s", "target": "se"}, {"source": "se", "target": "sp"}, {"source": "sp", "target": "e"},
        {"source": "t", "target": "te"}, {"source": "te", "target": "tp"}, {"source": "tp", "target": "d"},
        {"source": "e", "target": "d", "targetHandle": "memory"}, {"source": "d", "target": "g"},
    ]
    return {"projectName": "Translator", "training": "quick", "nodes": nodes, "edges": edges}


NLP_BUILDS = {
    "text_classify": chain(TXT, EMB, POS, ("transformer_encoder", {"heads": 2, "layers": 1}), ("mean_pool", {}), ("classify", {"classes": "Billing, Bug, Question"})),
    "text_number": chain(TXT, EMB, ("attention", {"heads": 2}), ("mean_pool", {}), ("predict_number", {})),
    "text_lstm": chain(TXT, EMB, ("lstm", {"hidden": 16}), ("last_step", {}), ("classify", {"classes": "a, b"})),
    "gpt_style": chain(TXT, EMB, POS, ("transformer_decoder", {"heads": 2, "layers": 1}), ("generate", {})),
    "translator": translator(),
    "translator_adapter": translator(memory_size=64),
    "composite": chain(TXT, ("text_classifier", {"classes": "a, b"})),
    "composite_encoder": chain(TXT, ("transformer_text_encoder", {}), ("predict_number", {})),
    "distilbert": chain(TXT, ("pretrained_language", {}), ("mean_pool", {}), ("classify", {"classes": "a, b"})),
    "writer": chain(("text_input", {"max_length": 32}), ("pretrained_writer", {}), ("generate", {})),
    "granite_writer": chain(TXT, ("pretrained_writer", {"model": "Granite 4.0 350M", "adjust_layers": 1}), ("generate", {})),
}


@pytest.mark.parametrize("name", NLP_BUILDS)
def test_nlp_smoke_test_passes(name):
    events = run_events(NLP_BUILDS[name])
    assert events[-1] == {"event": "result", "passed": True}, events[-3:]


def test_vision_transformer_smoke_test_passes():
    build = chain(("image_folder", {"image_size": 224, "channels": 3}), ("pretrained_vision", {}), ("classify", {"classes": "a, b"}))
    assert run_events(build)[-1] == {"event": "result", "passed": True}


_WORDS = "the quick brown fox jumps over a lazy dog and then runs far away".split()


def _texts_csv(d, answer):
    rows = ["text," + answer] + [f"{' '.join(random.choices(_WORDS, k=6))},{random.choice(['Billing', 'Bug', 'Question']) if answer == 'label' else i}" for i in range(30)]
    (d / "data").mkdir()
    (d / "data/texts.csv").write_text("\n".join(rows))


NLP_DATA = {
    "text_classify": lambda d: _texts_csv(d, "label"),
    "text_number": lambda d: _texts_csv(d, "target"),
    "gpt_style": lambda d: ((d / "data").mkdir(), (d / "data/corpus.txt").write_text(" ".join(random.choices(_WORDS, k=800)))),
    "translator": lambda d: ((d / "data").mkdir(), (d / "data/pairs.csv").write_text("source,target\n" + "\n".join(f"{w} one,{w} un" for w in _WORDS * 2))),
}


@pytest.mark.parametrize("name", NLP_DATA)
def test_nlp_project_trains_and_generates(name, tmp_path):
    graph = GraphIn(**NLP_BUILDS[name])
    analysis = analyse(graph)
    assert analysis.ok, analysis.issues
    for filename, contents in generate(graph, analysis).items():
        (tmp_path / filename).write_text(contents)
    NLP_DATA[name](tmp_path)
    done = subprocess.run([sys.executable, "train.py"], cwd=tmp_path, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr
    if (tmp_path / "generate.py").exists():
        out = subprocess.run([sys.executable, "generate.py", "the quick"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip()


@pytest.mark.parametrize("build, fragment", [
    (chain(TXT, ("embedding", {"size": 30}), ("transformer_encoder", {"heads": 4, "layers": 1}), ("mean_pool", {}), ("predict_number", {})), "divide evenly"),
    (chain(TXT, ("lstm", {}),), "takes Time series"),
    (chain(TXT, EMB, ("generate", {})), "needs a Transformer decoder"),
    (chain(TXT, ("pretrained_language", {}), ("generate", {})), "not generating text"),
    (chain(("image_folder", {"image_size": 64}), ("pretrained_vision", {}), ("predict_number", {})), "224×224"),
    (chain(SER, ("lstm", {"hidden": 16}), ("generate", {})), "needs a Text input"),
    (chain(TXT, ("pretrained_writer", {}), ("mean_pool", {}), ("classify", {"classes": "a, b"})), "straight to Generate text"),
    (chain(TXT, EMB, ("pretrained_writer", {}), ("generate", {})), "takes Text"),
    (chain(TXT, ("pretrained_writer", {"model": "GPT-9"}), ("generate", {})), "Model must be one of"),
])
def test_invalid_nlp_builds_explain_themselves(build, fragment):
    analysis = analyse(GraphIn(**build))
    assert any(fragment in i.message for i in analysis.issues), analysis.issues


def test_a_pretrained_writer_is_adjusted_on_your_text_and_chats(tmp_path):
    """Downloads SmolLM2 135M once into the test cache, adjusts its last 2 layers, then writes and chats."""
    graph = GraphIn(**NLP_BUILDS["writer"])
    analysis = analyse(graph)
    assert analysis.ok, analysis.issues
    assert analysis.shapes["n0"] == "32 tokens, using SmolLM2 135M's own tokenizer"
    files = generate(graph, analysis)
    assert "LEARNING_RATE = 0.0001" in files["train.py"] and "transformers>=4.56" in files["requirements.txt"]
    assert "--chat" in files["generate.py"] and "270 MB" in files["README.md"]
    for filename, contents in files.items():
        (tmp_path / filename).write_text(contents)
    (tmp_path / "data").mkdir()
    random.seed(2)
    (tmp_path / "data/corpus.txt").write_text("\n".join(" ".join(random.choices(_WORDS, k=12)) + "." for _ in range(80)))
    env = {**os.environ, "HF_HOME": str(Path(__file__).parent.parent / ".test-cache" / "huggingface")}
    done = subprocess.run([sys.executable, "train.py"], cwd=tmp_path, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, done.stderr
    import torch

    saved = torch.load(tmp_path / "model.pt", weights_only=True)["model"]
    assert saved and all(".layers.28." in key or ".layers.29." in key or key.endswith("norm.weight") for key in saved), list(saved)[:5]
    env["HF_HUB_OFFLINE"] = "1"  # after the first download it works without the internet
    for arguments in (["the quick"], ["--chat", "Say hello."]):
        out = subprocess.run([sys.executable, "generate.py", *arguments], cwd=tmp_path, capture_output=True, text=True, timeout=180, env=env)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip()


def test_translator_inputs_must_not_be_swapped():
    build = translator()
    for edge in build["edges"]:
        if edge["target"] == "d":
            edge["targetHandle"] = "in" if edge.get("targetHandle") == "memory" else "memory"
    assert any("Tokens input" in i.message for i in analyse(GraphIn(**build)).issues)


def test_padding_mask_reaches_the_bricks_that_use_it():
    graph = GraphIn(**NLP_BUILDS["text_classify"])
    model_py = generate(graph, analyse(graph))["model.py"]
    assert "padding_mask = x == 0" in model_py
    assert "self.encoder(x, padding_mask=padding_mask)" in model_py
    assert "self.mean_pool(x, padding_mask=padding_mask)" in model_py


# ------------------------------------------------------------------ code to blocks

from aibricks.code_parser import COMPONENTS, STARTER, parse_model  # noqa: E402
from aibricks.brick_sources import SOURCES  # noqa: E402


@pytest.mark.parametrize("name", list(BUILDS) + list(NLP_BUILDS))
def test_code_round_trips_to_identical_code(name):
    build = {**BUILDS, **NLP_BUILDS}[name]
    graph = GraphIn(**build)
    analysis = analyse(graph)
    assert analysis.ok, analysis.issues
    original = generate(graph, analysis)["model.py"]
    # Parsing without any existing blocks proves the code alone is enough.
    parsed = parse_model(original)
    assert not parsed.errors, parsed.errors
    rebuilt = GraphIn(projectName=graph.projectName, training=graph.training, nodes=parsed.nodes, edges=parsed.edges)
    second = analyse(rebuilt)
    assert second.ok, second.issues
    assert generate(rebuilt, second)["model.py"] == original


def test_every_generated_component_is_understood():
    generated = {cls for cls, _ in SOURCES.values()}
    assert generated == set(COMPONENTS)


def _request(code, nodes=()):
    return client.post("/api/parse-model", json={"code": code, "graph": {"nodes": list(nodes), "edges": []}}).json()


def test_starter_code_converts_and_passes_a_test_run():
    reply = _request(STARTER)
    assert reply["ok"], reply
    build = {"projectName": "Starter", "training": "quick", "nodes": reply["nodes"], "edges": reply["edges"]}
    assert run_events(build)[-1]["passed"]


def test_editing_code_changes_the_blocks():
    code = STARTER.replace("ConvBlock(3, 32, kernel_size=3)", "ConvBlock(3, 16, kernel_size=5)").replace("['OK', 'Defect']", "['A', 'B', 'C']")
    nodes = {n["id"]: n for n in _request(code)["nodes"]}
    assert nodes["conv"]["params"] == {"filters": 16, "kernel_size": 5}
    assert nodes["classify_head"]["params"]["classes"] == "A, B, C"
    assert nodes["input"]["params"] == {"channels": 3, "image_size": 64}


def test_wrong_sizes_are_corrected_and_explained():
    code = STARTER.replace("ClassifyHead(32,", "ClassifyHead(999,")
    reply = _request(code)
    assert reply["ok"]
    assert any("self.classify_head was changed" in note["message"] for note in reply["notes"])


def test_existing_input_details_are_kept():
    series = chain(("time_series", {"columns": "temp, speed", "sequence_length": 12}), LE, ("predict_number", {}))
    graph = GraphIn(**series)
    code = generate(graph, analyse(graph))["model.py"]
    reply = _request(code, series["nodes"])
    socket = next(n for n in reply["nodes"] if n["id"] == "input")
    assert socket["params"]["columns"] == "temp, speed"


@pytest.mark.parametrize("edit, line_text, fragment", [
    ("x = self.conv(x)", "x = self.conv(x) + 1", "lines like: x = self.name(x)"),
    ("self.conv = ConvBlock(3, 32, kernel_size=3)", "self.conv = nn.Conv2d(3, 32, 3)", "lines like: self.name"),
    ("self.conv = ConvBlock(3, 32, kernel_size=3)", "self.conv = MadeUp(3)", "isn't an AI Bricks component"),
    ("self.conv = ConvBlock(3, 32, kernel_size=3)", "self.conv = ConvBlock(3, 2 ** 40)", "plain value"),
    ("self.conv = ConvBlock(3, 32, kernel_size=3)", "self.conv = ConvBlock(3, 32, bogus=1)", "has no argument"),
    ("x = self.pool(x)\n", "\n", "never used"),
    ("        return x\n", "", "return line"),
    ("def forward(self, x):", "def forward(self, x, y):", "forward must be"),
    ("def forward(self, x):", "def forward(self, x:", "syntax error"),
])
def test_code_that_cannot_become_blocks_is_explained_with_a_line(edit, line_text, fragment):
    assert edit in STARTER
    reply = _request(STARTER.replace(edit, line_text, 1))
    assert not reply["ok"]
    assert any(fragment in e["message"] for e in reply["errors"]), reply["errors"]
    assert any(e["line"] for e in reply["errors"])


def test_code_is_never_run(tmp_path):
    marker = tmp_path / "ran"
    attacks = [
        f"open({str(marker)!r}, 'w')\n" + STARTER,
        f"import os\nos.system('touch {marker}')\n" + STARTER,
        STARTER.replace("ConvBlock(3, 32, kernel_size=3)", f"ConvBlock(3, open({str(marker)!r}, 'w'))"),
        STARTER.replace("def forward(self, x):", f"def forward(self, x, y=open({str(marker)!r}, 'w')):"),
        STARTER.replace("class MyModel(nn.Module):", f"@open({str(marker)!r}, 'w')\nclass MyModel(nn.Module):"),
    ]
    for code in attacks:
        reply = _request(code)
        assert not marker.exists()
        if reply["ok"]:
            # Anything accepted is rebuilt from validated settings, so the text can't survive.
            assert str(marker) not in str(reply)


def test_custom_training_values_reach_the_script_and_are_checked():
    build = chain(IMG, ENC, ("predict_number", {})) | {"trainingOverride": {"epochs": 7, "batch_size": 4, "learning_rate": 0.01}}
    graph = GraphIn(**build)
    train = generate(graph, analyse(graph))["train.py"]
    assert "EPOCHS = 7" in train and "BATCH_SIZE = 4" in train and "LEARNING_RATE = 0.01" in train
    bad = GraphIn(**(chain(IMG, ENC, ("predict_number", {})) | {"trainingOverride": {"epochs": "print('x')"}}))
    assert any("Epochs must be" in i.message for i in analyse(bad).issues)
