"""Turns a trained photo or pose model into a single web page that runs on its own, in any browser.

The model is exported to ONNX, an open format for trained models, and embedded in the page. The
page runs it with ONNX Runtime Web, so photos are checked on the device and never uploaded. It
also checks itself when it opens: it runs a fixed test pattern and compares the answer with the
one PyTorch gave, so a model that converted wrongly says so instead of guessing quietly.

A pose model's page also carries MediaPipe's point-finding model and runs it with MediaPipe's own
web library. It checks that its sums on the points match the ones training used, the same way.
"""
from __future__ import annotations

import base64
import html
import json
import subprocess
import sys

from . import training
from .training import TrainingError

APP_FILE = "app.onnx"
APP_INFO = "app.json"
MAX_APP_MODEL_BYTES = 120 * 1024 * 1024  # Base64 adds a third, and a phone has to hold all of it.
RUNTIME_VERSION = "1.30.0"
RUNTIME_URL = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{RUNTIME_VERSION}/dist/"
RUNTIME_INTEGRITY = "sha384-N94xSNjPDbfBJj4+QINst0nbHpcCm8kc1qNlIP/wA0muqjR3aVyIuDkMeccPrLvQ"
VISION_VERSION = "1.1.1"
VISION_URL = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{VISION_VERSION}/"
VISION_INTEGRITY = "sha384-33ConKAFiqnDTqoc8FFsTiKhp23eN52R5hP3ZqPBuRvIe9XyWblqpaEYnZZNVIvI"

# Runs in the model's own folder, so it reads the same settings that training used.
_EXPORT = """
import json, math, sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import torch
from torchvision import transforms
from predict import load_trained_model
from data import CHANNELS, IMAGE_SIZE, _image_transform
from model import CLASS_NAMES
model = load_trained_model()
mean = std = None
for step in _image_transform().transforms:
    if isinstance(step, transforms.Normalize):
        mean, std = [float(v) for v in step.mean], [float(v) for v in step.std]
shape = (1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
count = CHANNELS * IMAGE_SIZE * IMAGE_SIZE
pattern = torch.tensor([math.sin(i * 0.37) for i in range(count)], dtype=torch.float32).reshape(shape)
with torch.no_grad():
    expected = model(pattern)[0].tolist()
torch.onnx.export(model, (torch.zeros(shape),), 'app.onnx.partial', dynamo=False, input_names=['image'], output_names=['logits'], opset_version=17)
import os
os.replace('app.onnx.partial', 'app.onnx')
print(json.dumps({'size': IMAGE_SIZE, 'channels': CHANNELS, 'mean': mean, 'std': std, 'labels': list(CLASS_NAMES), 'check': expected}))
"""


# A pose model's network, and a check that the page measures points exactly as training did.
_EXPORT_POSE = """
import json, math, sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import torch
import model as m
settings = json.loads(open('model/settings.json', encoding='utf-8').read())
network = m.build(settings['categories'])
network.load_state_dict(torch.load('model/weights.pt', map_location='cpu', weights_only=True))
network.eval()
pattern = torch.tensor([math.sin(i * 0.37) for i in range(m.FEATURES)], dtype=torch.float32).reshape(1, m.FEATURES)
with torch.no_grad():
    expected = network(pattern)[0].tolist()
torch.onnx.export(network, (torch.zeros(1, m.FEATURES),), 'app.onnx.partial', dynamo=False, input_names=['features'], output_names=['logits'], opset_version=17)
rng = np.random.default_rng(7)
width, height = 640, 480
sx, sy = width / max(width, height), height / max(width, height)
def points(count):
    return [{'x': float(x), 'y': float(y), 'z': float(z), 'visibility': float(v)} for x, y, z, v in rng.uniform([0.1, 0.1, -0.2, 0], [0.9, 0.9, 0.2, 1], (count, 4))]
def xyz(rows):
    return np.array([[p['x'] * sx, p['y'] * sy, p['z'] * sx] for p in rows], dtype=np.float32)
if m.POINTS == 'body':
    body = points(m.BODY_POINTS)
    sample = {'landmarks': [body]}
    features = m.body_features(xyz(body), np.array([p['visibility'] > m.SEEN for p in body], dtype=np.float32))
else:
    left, right = points(m.HAND_POINTS), points(m.HAND_POINTS)
    sample = {'landmarks': [right, left], 'handedness': [[{'categoryName': 'Right'}], [{'categoryName': 'Left'}]]}
    features = m.hand_features({'Left': xyz(left), 'Right': xyz(right)})
import os
os.replace('app.onnx.partial', 'app.onnx')
print(json.dumps({'kind': 'pose', 'points': m.POINTS, 'features': m.FEATURES, 'labels': settings['categories'], 'check': expected,
                  'sample': {'width': width, 'height': height, 'result': sample, 'features': [float(v) for v in features]}}))
"""


# A fast detector, at the 320 by 320 size it reads. The surest boxes for a fixed pattern check the page.
_EXPORT_DETECT = """
import json, math, sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import torch
import model as m
settings = json.loads(open('model/settings.json', encoding='utf-8').read())
detector = m.build(settings['categories'], pretrained=False)
detector.load_state_dict(torch.load('model/model.pt', map_location='cpu', weights_only=True))
detector.eval()
class Flat(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, image):
        out = self.inner([image])[0]
        return out['boxes'], out['scores'], out['labels']
flat = Flat(detector).eval()
pattern = torch.tensor([math.sin(i * 0.37) * 0.5 + 0.5 for i in range(3 * 320 * 320)], dtype=torch.float32).reshape(3, 320, 320)
with torch.no_grad():
    boxes, scores, _ = flat(pattern)
torch.onnx.export(flat, (pattern,), 'app.onnx.partial', dynamo=False, opset_version=17, input_names=['image'],
                  output_names=['boxes', 'scores', 'labels'], dynamic_axes={'boxes': {0: 'n'}, 'scores': {0: 'n'}, 'labels': {0: 'n'}})
import os
os.replace('app.onnx.partial', 'app.onnx')
top = min(20, len(scores))
print(json.dumps({'kind': 'detect', 'labels': settings['categories'], 'size': 320,
                  'check': {'scores': scores[:top].tolist(), 'boxes': boxes[:top].reshape(-1).tolist()}}))
"""


# An outlining model, with the photo's normalising built in so the page passes plain 0 to 1 pixels.
# Every 997th number it gives for a fixed pattern checks the page.
_EXPORT_OUTLINE = """
import json, math, sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import torch
import model as m
settings = json.loads(open('model/settings.json', encoding='utf-8').read())
network = m.build(settings['categories'], pretrained=False)
network.load_state_dict(torch.load('model/model.pt', map_location='cpu', weights_only=True))
network.eval()
class Flat(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.register_buffer('mean', torch.tensor(m.MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(m.STD).reshape(1, 3, 1, 1))
    def forward(self, image):
        return self.inner((image - self.mean) / self.std)['out']
flat = Flat(network).eval()
side = m.SIZE
pattern = torch.tensor([math.sin(i * 0.37) * 0.5 + 0.5 for i in range(3 * side * side)], dtype=torch.float32).reshape(1, 3, side, side)
with torch.no_grad():
    expected = flat(pattern).reshape(-1)[::997].tolist()
torch.onnx.export(flat, (pattern,), 'app.onnx.partial', dynamo=False, opset_version=17, input_names=['image'], output_names=['logits'])
import os
os.replace('app.onnx.partial', 'app.onnx')
print(json.dumps({'kind': 'outline', 'labels': settings['categories'], 'size': side, 'colours': m.COLOURS, 'check': {'step': 997, 'values': expected}}))
"""


def _export(job: training.Job) -> dict:
    """Exports the model once. A trained model never changes, so later pages reuse the file."""
    folder = training._folder(job)
    info_path, model_path = folder / APP_INFO, folder / APP_FILE
    if info_path.is_file() and model_path.is_file():
        return json.loads(info_path.read_text(encoding="utf-8"))
    try:
        with training._predictions:
            done = subprocess.run(
                [sys.executable, "-I", "-c", {"pose": _EXPORT_POSE, "detect": _EXPORT_DETECT, "outline": _EXPORT_OUTLINE}.get(job.task, _EXPORT)], cwd=folder, env=training._environment(folder) | {"OMP_NUM_THREADS": "2"},
                stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace", timeout=training.PREDICT_LIMIT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        raise TrainingError(504, "Converting the model took too long.") from None
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise TrainingError(500, f"This model couldn't be converted for the web: {lines[-1][:400] if lines else 'unknown error'}")
    info = json.loads(done.stdout.strip().splitlines()[-1])
    info_path.write_text(json.dumps(info), encoding="utf-8")
    return info


def _job(job_id: str, where: str = "a web page", made: str = "an app") -> training.Job:
    job = training.stored(job_id)
    if job.task not in ("image", "pose", "detect", "outline"):
        raise TrainingError(422, f"{made[0].upper()}{made[1:]} can only be made from a photo, pose, outlining or fast find-things model for now.")
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This run didn't save a model.")
    if job.task == "detect" and _detect_settings(job).get("detector", "accurate") != "fast":
        raise TrainingError(422, f"The accurate detector is too big and slow for {where}. Train again with the fast detector to make {made}.")
    return job


def _detect_settings(job: training.Job) -> dict:
    return json.loads((training._folder(job) / "model" / "settings.json").read_text(encoding="utf-8"))


def page(job_id: str) -> tuple[str, str]:
    """The finished page and a file name for it."""
    job = _job(job_id)
    info = _export(job)
    model = (training._folder(job) / APP_FILE).read_bytes()
    if len(model) > MAX_APP_MODEL_BYTES:
        raise TrainingError(413, f"This model is {len(model) // (1024 * 1024)} MB, too big to put in a web page. Smaller models, like the ones built from blocks, work best.")
    # No "<" at all, so nothing in a name can end the script block or change how it's read.
    pose, detect, outlining = job.task == "pose", job.task == "detect", job.task == "outline"
    labels = ", ".join(info["labels"])
    rule = {}
    if outlining:
        about, nothing = f"Outlines {labels}, and shows how much of the view each one covers.", "Nothing outlined."
        rule = {"rule": {"words": ["When", "covers at least", "% of the view,"], "min": 1, "start": 20}}
    elif detect:
        about, nothing = f"Finds and counts: {labels}.", "Nothing found."
    elif pose:
        hands = info.get("points") == "hands"
        about, nothing = f"Sorts {'hand signs' if hands else 'poses'} into: {labels}.", "No hands in view." if hands else "No one in view."
    else:
        about, nothing = f"Sorts photos into: {labels}.", ""
    extra_settings = {"sure": _detect_settings(job).get("sure", 0.5)} if detect else {}  # the slider may have moved since export
    settings = json.dumps({**info, **extra_settings, **rule, "name": job.projectName, "trainedAt": job.startedAt, "about": about, "nothing": nothing}).replace("<", "\\u003c")
    finder = training._folder(job) / "model" / "points.task"
    extra = ""
    if pose:
        extra = (f'<script id="finder" type="application/octet-stream">{base64.b64encode(finder.read_bytes()).decode()}</script>\n'
                 f'<script src="{VISION_URL}vision_bundle.js" integrity="{VISION_INTEGRITY}" crossorigin="anonymous"></script>')
    text = (_PAGE
            .replace("__MODEL_CODE__", (_POSE_CODE if pose else _DETECT_CODE if detect else _OUTLINE_CODE if outlining else _IMAGE_CODE).replace("__VISION__", VISION_URL))
            .replace("__EXTRA__", extra)
            .replace("__TITLE__", html.escape(job.projectName))
            .replace("__RUNTIME__", RUNTIME_URL)
            .replace("__INTEGRITY__", RUNTIME_INTEGRITY)
            .replace("__SETTINGS__", settings)
            .replace("__MODEL__", base64.b64encode(model).decode()))
    return text, f"{job.slug or 'model'}-app.html"


_IMAGE_CODE = r"""// ---------------------------------------------------------------- the model

function prepare(source, width, height) {
  // The same steps as training: squash to a square, scale to 0-1, then normalise if it used a pretrained model.
  const side = S.size, canvas = document.createElement("canvas");
  canvas.width = side; canvas.height = side;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, width, height, 0, 0, side, side);
  const pixels = context.getImageData(0, 0, side, side).data;
  const plane = side * side, data = new Float32Array(S.channels * plane);
  for (let i = 0; i < plane; i++) {
    const r = pixels[i * 4] / 255, g = pixels[i * 4 + 1] / 255, b = pixels[i * 4 + 2] / 255;
    const values = S.channels === 1 ? [0.299 * r + 0.587 * g + 0.114 * b] : [r, g, b];
    values.forEach((value, c) => { data[c * plane + i] = S.mean ? (value - S.mean[c]) / S.std[c] : value; });
  }
  return new ort.Tensor("float32", data, [1, S.channels, side, side]);
}

async function run(tensor) {
  const output = await session.run({ image: tensor });
  return Array.from(output.logits.data);
}

async function answer(source, width, height) {
  return ranked(await run(prepare(source, width, height)));
}

async function load() {
  if (typeof ort === "undefined") { say("The model runner couldn't load. Connect to the internet once to open this page.", "bad"); return; }
  ort.env.wasm.wasmPaths = "__RUNTIME__";
  ort.env.wasm.numThreads = 1;
  const bytes = await (await fetch(`data:application/octet-stream;base64,${document.getElementById("model").textContent.trim()}`)).arrayBuffer();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  // A fixed pattern, run by PyTorch when the page was made, must give the same answer here.
  const count = S.channels * S.size * S.size, pattern = new Float32Array(count);
  for (let i = 0; i < count; i++) pattern[i] = Math.sin(i * 0.37);
  const got = await run(new ort.Tensor("float32", pattern, [1, S.channels, S.size, S.size]));
  const gap = Math.max(...got.map((v, i) => Math.abs(v - S.check[i]) / Math.max(1, Math.abs(S.check[i]))));
  window.selfCheckGap = gap;
  if (gap < 1e-3) say("✓ Model loaded and checked. It runs on this device.", "good");
  else say("This model gave different answers here than when it was made, so its guesses may be wrong.", "bad");
  $("camera").disabled = false;
  $("choose").disabled = false;
}

"""

_POSE_CODE = r"""// ---------------------------------------------------------------- the model

// MediaPipe finds the points, then the network names the pose. Points are measured exactly as in training.
const SEEN = 0.5, HAND_POINTS = 21;
const HAND_LINES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const BODY_LINES = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]];
let finder = null;

function scaled(points, width, height) {
  const side = Math.max(width, height), sx = width / side, sy = height / side;
  return points.map((p) => [p.x * sx, p.y * sy, p.z * sx]);
}

function bodyFeatures(points, width, height) {
  const xyz = scaled(points, width, height), seen = points.map((p) => ((p.visibility || 0) > SEEN ? 1 : 0));
  const centre = [0, 1, 2].map((k) => (xyz[11][k] + xyz[12][k]) / 2);
  const moved = xyz.map((p) => p.map((v, k) => v - centre[k]));
  let total = 0, count = 0;
  moved.forEach((p, i) => { if (seen[i]) { total += p[0] * p[0] + p[1] * p[1]; count++; } });
  const scale = Math.max(count ? Math.sqrt(total / count) : 1, 1e-3);
  return Float32Array.from([...moved.flatMap((p, i) => p.map((v) => (v / scale) * seen[i])), ...seen]);
}

function handFeatures(hands) {
  const parts = [];
  for (const side of ["Left", "Right"]) {
    const xyz = hands[side];
    if (!xyz) { parts.push(...new Array(HAND_POINTS * 3 + 1).fill(0)); continue; }
    const moved = xyz.map((p) => p.map((v, k) => v - xyz[0][k]));
    const scale = Math.max(Math.max(...moved.map((p) => Math.sqrt(p[0] * p[0] + p[1] * p[1]))), 1e-3);
    parts.push(...moved.flatMap((p) => p.map((v) => v / scale)), 1);
  }
  return Float32Array.from(parts);
}

function features(result, width, height) {
  const found = result.landmarks || [];
  if (!found.length) return null;
  if (S.points === "body") return bodyFeatures(found[0], width, height);
  const sides = result.handedness || result.handednesses || [], hands = {};
  found.forEach((points, i) => {
    let side = (sides[i] && sides[i][0] && sides[i][0].categoryName) || "Right";
    if (hands[side]) side = side === "Right" ? "Left" : "Right";
    hands[side] = scaled(points, width, height);
  });
  return handFeatures(hands);
}

function draw(result, element, width, height, mirrored) {
  const canvas = $("overlay"), ratio = window.devicePixelRatio || 1, area = shownArea(element, width, height);
  canvas.width = Math.round(area.frame.width * ratio);
  canvas.height = Math.round(area.frame.height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const at = (p) => [area.left + (mirrored ? 1 - p.x : p.x) * area.width, area.top + p.y * area.height];
  const visible = (p) => S.points === "hands" || (p.visibility || 0) >= SEEN;
  for (const points of result.landmarks || []) {
    context.strokeStyle = "#fff"; context.lineWidth = 2.5; context.lineCap = "round";
    context.shadowColor = "rgba(0,0,0,.8)"; context.shadowBlur = 2;
    for (const [a, b] of S.points === "hands" ? HAND_LINES : BODY_LINES) {
      if (!points[a] || !points[b] || !visible(points[a]) || !visible(points[b])) continue;
      context.beginPath(); context.moveTo(...at(points[a])); context.lineTo(...at(points[b])); context.stroke();
    }
    context.fillStyle = "#2f5fb8";
    for (const p of points) { if (!visible(p)) continue; context.beginPath(); context.arc(...at(p), 3.5, 0, 2 * Math.PI); context.fill(); }
  }
}

async function run(values) {
  const output = await session.run({ features: new ort.Tensor("float32", values, [1, S.features]) });
  return Array.from(output.logits.data);
}

async function answer(source, width, height, mirrored) {
  const result = finder.detect(source);
  draw(result, source, width, height, mirrored);
  const values = features(result, width, height);
  return values ? ranked(await run(values)) : null;
}

async function load() {
  if (typeof ort === "undefined" || typeof Vision === "undefined") { say("The model runner couldn't load. Connect to the internet once to open this page.", "bad"); return; }
  ort.env.wasm.wasmPaths = "__RUNTIME__";
  ort.env.wasm.numThreads = 1;
  const decode = async (id) => new Uint8Array(await (await fetch(`data:application/octet-stream;base64,${document.getElementById(id).textContent.trim()}`)).arrayBuffer());
  session = await ort.InferenceSession.create(await decode("model"), { executionProviders: ["wasm"] });
  // A fixed pattern, run by PyTorch when the page was made, must give the same answer here.
  const pattern = Float32Array.from({ length: S.features }, (_, i) => Math.sin(i * 0.37));
  const got = await run(pattern);
  const gap = Math.max(...got.map((v, i) => Math.abs(v - S.check[i]) / Math.max(1, Math.abs(S.check[i]))));
  // So must measuring a set of made-up points.
  const measured = features(S.sample.result, S.sample.width, S.sample.height);
  const pointGap = Math.max(...S.sample.features.map((v, i) => Math.abs(v - measured[i])));
  window.selfCheckGap = Math.max(gap, pointGap);
  say("Loading the model that finds the points…");
  const files = await Vision.FilesetResolver.forVisionTasks("__VISION__wasm");
  const options = { baseOptions: { modelAssetBuffer: await decode("finder"), delegate: "CPU" }, runningMode: "IMAGE" };
  finder = S.points === "hands"
    ? await Vision.HandLandmarker.createFromOptions(files, { ...options, numHands: 2 })
    : await Vision.PoseLandmarker.createFromOptions(files, { ...options, numPoses: 1 });
  if (window.selfCheckGap < 1e-3) say("✓ Model loaded and checked. It runs on this device.", "good");
  else say("This model gave different answers here than when it was made, so its guesses may be wrong.", "bad");
  $("camera").disabled = false;
  $("choose").disabled = false;
}

"""

_DETECT_CODE = r"""// ---------------------------------------------------------------- the model

// The detector reads a 320 by 320 picture and returns boxes in that picture's pixels, surest first.
const SIDE = 320, COLOURS = ["#2f5fb8", "#d1495b", "#2e7d5b", "#c77d00", "#7b4fb8", "#0f8b8d", "#b34a7d", "#5b6472"];

function prepare(source, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = SIDE; canvas.height = SIDE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, width, height, 0, 0, SIDE, SIDE);
  const pixels = context.getImageData(0, 0, SIDE, SIDE).data, plane = SIDE * SIDE, data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) data[c * plane + i] = pixels[i * 4 + c] / 255;
  return data;
}

async function run(data) {
  const output = await session.run({ image: new ort.Tensor("float32", data, [3, SIDE, SIDE]) });
  return { boxes: Array.from(output.boxes.data), scores: Array.from(output.scores.data), labels: Array.from(output.labels.data, Number) };
}

function draw(found, element, width, height, mirrored) {
  const canvas = $("overlay"), ratio = window.devicePixelRatio || 1, area = shownArea(element, width, height);
  canvas.width = Math.round(area.frame.width * ratio);
  canvas.height = Math.round(area.frame.height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.font = "600 12px system-ui, sans-serif";
  for (const box of found) {
    const colour = COLOURS[S.labels.indexOf(box.name) % COLOURS.length];
    const left = area.left + (mirrored ? 1 - box.x - box.w : box.x) * area.width, top = area.top + box.y * area.height;
    context.strokeStyle = colour; context.lineWidth = 2.5;
    context.strokeRect(left, top, box.w * area.width, box.h * area.height);
    const tag = `${box.name} ${Math.round(box.chance * 100)}%`, tagWidth = context.measureText(tag).width + 10;
    context.fillStyle = colour; context.fillRect(left, Math.max(0, top - 18), tagWidth, 18);
    context.fillStyle = "#fff"; context.fillText(tag, left + 5, Math.max(0, top - 18) + 13);
  }
}

async function answer(source, width, height, mirrored) {
  const out = await run(prepare(source, width, height)), found = [];
  for (let i = 0; i < out.scores.length; i++) {
    if (out.scores[i] < S.sure) continue;
    const [x1, y1, x2, y2] = out.boxes.slice(i * 4, i * 4 + 4).map((v) => Math.min(Math.max(v / SIDE, 0), 1));
    found.push({ name: S.labels[out.labels[i] - 1], chance: out.scores[i], x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
  }
  draw(found, source, width, height, mirrored);
  window.lastBoxes = found;
  if (!found.length) return null;
  // One row per thing it found: how many, and how sure it is of the surest one.
  return S.labels
    .map((name) => { const mine = found.filter((box) => box.name === name); return { name, count: mine.length, chance: Math.max(0, ...mine.map((box) => box.chance)) }; })
    .filter((row) => row.count)
    .sort((a, b) => b.count - a.count || b.chance - a.chance);
}

async function load() {
  if (typeof ort === "undefined") { say("The model runner couldn't load. Connect to the internet once to open this page.", "bad"); return; }
  ort.env.wasm.wasmPaths = "__RUNTIME__";
  ort.env.wasm.numThreads = 1;
  const bytes = await (await fetch(`data:application/octet-stream;base64,${document.getElementById("model").textContent.trim()}`)).arrayBuffer();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  // A fixed pattern, run by PyTorch when the page was made, must give the same surest boxes here.
  const pattern = Float32Array.from({ length: 3 * SIDE * SIDE }, (_, i) => Math.sin(i * 0.37) * 0.5 + 0.5);
  const got = await run(pattern);
  let gap = got.scores.length >= S.check.scores.length ? 0 : 1;
  S.check.scores.forEach((v, i) => { gap = Math.max(gap, Math.abs(v - got.scores[i])); });
  S.check.boxes.forEach((v, i) => { gap = Math.max(gap, Math.abs(v - got.boxes[i]) / SIDE); });
  window.selfCheckGap = gap;
  if (gap < 1e-3) say("✓ Model loaded and checked. It runs on this device.", "good");
  else say("This model gave different answers here than when it was made, so its guesses may be wrong.", "bad");
  $("camera").disabled = false;
  $("choose").disabled = false;
}

"""

_OUTLINE_CODE = r"""// ---------------------------------------------------------------- the model

// The model reads a SIDE by SIDE picture and scores each thing at every spot; the best score wins.
const SIDE = S.size;
let painted = null;

function prepare(source, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = SIDE; canvas.height = SIDE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, width, height, 0, 0, SIDE, SIDE);
  const pixels = context.getImageData(0, 0, SIDE, SIDE).data, plane = SIDE * SIDE, data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) data[c * plane + i] = pixels[i * 4 + c] / 255;
  return data;
}

async function run(data) {
  const output = await session.run({ image: new ort.Tensor("float32", data, [1, 3, SIDE, SIDE]) });
  return output.logits.data;
}

function labelsOf(logits) {
  const plane = SIDE * SIDE, classes = S.labels.length + 1, labels = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    let best = 0, top = logits[i];
    for (let c = 1; c < classes; c++) { const v = logits[c * plane + i]; if (v > top) { top = v; best = c; } }
    labels[i] = best;
  }
  return labels;
}

function draw(labels, element, width, height, mirrored) {
  if (!painted) { painted = document.createElement("canvas"); painted.width = SIDE; painted.height = SIDE; }
  const small = painted.getContext("2d"), picture = small.createImageData(SIDE, SIDE);
  const colours = S.colours.map((hex) => [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16)));
  labels.forEach((label, i) => {
    if (!label) return;
    const [r, g, b] = colours[(label - 1) % colours.length];
    picture.data.set([r, g, b, 150], i * 4);
  });
  small.putImageData(picture, 0, 0);
  const canvas = $("overlay"), ratio = window.devicePixelRatio || 1, area = shownArea(element, width, height);
  canvas.width = Math.round(area.frame.width * ratio);
  canvas.height = Math.round(area.frame.height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  if (mirrored) { context.translate(area.left * 2 + area.width, 0); context.scale(-1, 1); }
  context.drawImage(painted, area.left, area.top, area.width, area.height);
}

async function answer(source, width, height, mirrored) {
  const labels = labelsOf(await run(prepare(source, width, height)));
  draw(labels, source, width, height, mirrored);
  const counts = new Array(S.labels.length + 1).fill(0);
  labels.forEach((label) => { counts[label]++; });
  window.lastAreas = Object.fromEntries(S.labels.map((name, i) => [name, counts[i + 1] / labels.length]));
  // One row per thing it outlined: how much of the view it covers.
  const rows = S.labels.map((name, i) => ({ name, chance: counts[i + 1] / labels.length })).filter((row) => row.chance >= 0.005);
  return rows.length ? rows.sort((a, b) => b.chance - a.chance) : null;
}

async function load() {
  if (typeof ort === "undefined") { say("The model runner couldn't load. Connect to the internet once to open this page.", "bad"); return; }
  ort.env.wasm.wasmPaths = "__RUNTIME__";
  ort.env.wasm.numThreads = 1;
  const bytes = await (await fetch(`data:application/octet-stream;base64,${document.getElementById("model").textContent.trim()}`)).arrayBuffer();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  // A fixed pattern, run by PyTorch when the page was made, must give the same scores here.
  const got = await run(Float32Array.from({ length: 3 * SIDE * SIDE }, (_, i) => Math.sin(i * 0.37) * 0.5 + 0.5));
  let gap = 0;
  S.check.values.forEach((v, i) => { gap = Math.max(gap, Math.abs(v - got[i * S.check.step]) / Math.max(1, Math.abs(v))); });
  window.selfCheckGap = gap;
  if (gap < 1e-3) say("✓ Model loaded and checked. It runs on this device.", "good");
  else say("This model gave different answers here than when it was made, so its guesses may be wrong.", "bad");
  $("camera").disabled = false;
  $("choose").disabled = false;
}

"""

_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #f5f6f8; --surface: #fff; --ink: #181c24; --soft: #5b6472; --line: #e1e4ea; --accent: #2f5fb8; --accent-tint: #edf2fb; --good: #2e7d5b; --bad: #b3362b; --radius: 8px; color-scheme: light; }
  @media (prefers-color-scheme: dark) { :root { --bg: #111318; --surface: #1a1d24; --ink: #e8eaef; --soft: #a3aab7; --line: #2c313b; --accent: #7aa2f0; --accent-tint: #1f2a40; --good: #5cc296; --bad: #f08a7e; color-scheme: dark; } }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 900px; margin: 0 auto; padding: 20px 16px 40px; display: flex; flex-direction: column; gap: 14px; }
  h1 { font-size: 22px; margin: 0; letter-spacing: -0.01em; }
  h2 { font-size: 15px; margin: 0 0 8px; }
  p { margin: 0; }
  .muted { color: var(--soft); }
  .small { font-size: 13px; }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px; min-width: 0; }
  .layout { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 14px; align-items: start; }
  .frame { position: relative; aspect-ratio: 4 / 3; background: #000; border-radius: var(--radius); overflow: hidden; }
  .frame video, .frame img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .frame video { transform: scaleX(-1); }
  .frame canvas { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  .answers li.nothing:first-child { display: block; font-weight: 400; }
  .frame .note { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; padding: 16px; text-align: center; color: #fff; }
  .buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  button, select, input { font: inherit; color: inherit; }
  button { padding: 6px 12px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: 0.5; cursor: default; }
  select, input[type=text], input[type=url], input[type=number] { padding: 5px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); min-width: 0; }
  .answers { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
  .answers li { display: grid; grid-template-columns: minmax(0, 8em) minmax(0, 1fr) 3.2em; align-items: center; gap: 10px; }
  .answers li:first-child { font-weight: 600; }
  .name { overflow-wrap: anywhere; }
  .track { height: 8px; border-radius: 4px; background: var(--accent-tint); overflow: hidden; }
  .track span { display: block; height: 100%; background: var(--accent); }
  .value { text-align: right; font-variant-numeric: tabular-nums; }
  .rules { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
  .rule { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding: 8px; border: 1px solid var(--line); border-radius: var(--radius); }
  .rule.firing { border-color: var(--good); box-shadow: inset 0 0 0 1px var(--good); }
  .rule input[type=text], .rule input[type=url] { flex: 1 1 12em; }
  .rule input[type=number] { width: 4.5em; }
  .count { font-variant-numeric: tabular-nums; font-weight: 600; }
  .banner { position: fixed; left: 50%; top: 20px; transform: translateX(-50%); max-width: calc(100% - 32px); padding: 14px 22px; border-radius: var(--radius); background: var(--ink); color: var(--bg); font-size: 22px; font-weight: 600; text-align: center; box-shadow: 0 8px 30px rgba(0,0,0,.25); }
  .status { display: inline-flex; align-items: center; gap: 6px; }
  .good { color: var(--good); }
  .bad { color: var(--bad); }
  footer { text-align: center; }
  @media (max-width: 680px) { .layout { grid-template-columns: minmax(0, 1fr); } }
</style>
</head>
<body>
<main>
  <header>
    <h1>__TITLE__</h1>
    <p class="muted small" id="about"></p>
    <p class="small status" id="status" role="status">Loading the model…</p>
  </header>

  <div class="layout">
    <section class="card">
      <div class="frame">
        <video id="video" muted playsinline aria-label="Camera"></video>
        <img id="photo" alt="The photo you chose" hidden>
        <canvas id="overlay" aria-hidden="true"></canvas>
        <span class="note" id="note">Start the camera, or choose a photo.</span>
      </div>
      <div class="buttons">
        <button class="primary" id="camera" disabled>Start camera</button>
        <button id="choose" disabled>Choose a photo</button>
        <input type="file" id="file" accept="image/*" hidden>
      </div>
    </section>
    <section class="card">
      <h2>What it sees</h2>
      <ul class="answers" id="answers" aria-live="polite"></ul>
      <p class="muted small" id="rate" style="margin-top:8px"></p>
    </section>
  </div>

  <section class="card">
    <h2>When it sees something</h2>
    <p class="muted small">Make it do something when it's sure for half a second. These rules are saved in this browser.</p>
    <ul class="rules" id="rules" style="margin-top:10px"></ul>
    <div class="buttons"><button id="add">Add a rule</button></div>
  </section>

  <footer class="muted small">
    Made with AI Bricks. Photos and video are checked on this device and never uploaded, unless a rule sends a result to a web address you typed.
  </footer>
</main>
<div class="banner" id="banner" hidden role="alert"></div>

<script id="settings" type="application/json">__SETTINGS__</script>
<script id="model" type="application/octet-stream">__MODEL__</script>
<script src="__RUNTIME__ort.min.js" integrity="__INTEGRITY__" crossorigin="anonymous"></script>
__EXTRA__
<script>
"use strict";
const S = JSON.parse(document.getElementById("settings").textContent);
const $ = (id) => document.getElementById(id);
const FRAMES_TO_FIRE = 3;
// How a rule reads: most models are some percent sure; an outlining model covers some share of the view.
const RULE = S.rule || { words: ["When it sees", "at least", "% sure,"], min: 50, start: 80 };
const COOLDOWN_MS = 2000;
let session = null, stream = null, running = false;

$("about").textContent = `${S.about} Trained ${new Date(S.trainedAt * 1000).toLocaleDateString()}.`;

function say(text, tone) {
  const box = $("status");
  box.textContent = text;
  box.className = `small status ${tone || ""}`;
}

function ranked(logits) {
  const top = Math.max(...logits), exps = logits.map((v) => Math.exp(v - top)), total = exps.reduce((a, b) => a + b, 0);
  return S.labels.map((name, i) => ({ name, chance: exps[i] / total })).sort((a, b) => b.chance - a.chance);
}

function show(answers) {
  if (!answers) {
    const row = document.createElement("li");
    row.className = "nothing muted";
    row.textContent = S.nothing;
    $("answers").replaceChildren(row);
    check(null);
    return;
  }
  $("answers").replaceChildren(...answers.map((a) => {
    const row = document.createElement("li");
    row.innerHTML = `<span class="name"></span><span class="track"><span></span></span><span class="value"></span>`;
    row.children[0].textContent = a.count === undefined ? a.name : `${a.name} × ${a.count}`;
    row.children[1].firstChild.style.width = `${a.chance * 100}%`;
    row.children[2].textContent = `${Math.round(a.chance * 100)}%`;
    return row;
  }));
  check(answers);
}

// Where a picture shown with object-fit: contain really sits inside its frame.
function shownArea(element, width, height) {
  const box = element.getBoundingClientRect(), scale = Math.min(box.width / width, box.height / height);
  return { left: (box.width - width * scale) / 2, top: (box.height - height * scale) / 2, width: width * scale, height: height * scale, frame: box };
}

__MODEL_CODE__
// ---------------------------------------------------------------- camera and photos

async function startCamera() {
  if (stream) { stopCamera(); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (error) {
    $("note").textContent = error && error.name === "NotAllowedError" ? "The camera wasn't allowed. Allow it in the address bar." : "The camera couldn't start. On a phone, this page must be opened from an https:// address.";
    return;
  }
  $("photo").hidden = true;
  $("video").hidden = false;
  $("video").srcObject = stream;
  await $("video").play();
  $("note").hidden = true;
  $("camera").textContent = "Stop camera";
  running = true;
  loop();
}

function stopCamera() {
  running = false;
  stream && stream.getTracks().forEach((track) => track.stop());
  stream = null;
  $("camera").textContent = "Start camera";
  const overlay = $("overlay");
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  $("note").hidden = false;
  $("note").textContent = "Camera off.";
  $("rate").textContent = "";
}

async function loop() {
  let frames = 0, since = performance.now();
  while (running) {
    const video = $("video");
    if (video.videoWidth && !document.hidden) {
      show(await answer(video, video.videoWidth, video.videoHeight, true));
      frames++;
      const now = performance.now();
      if (now - since > 1000) { $("rate").textContent = `About ${Math.round(frames / ((now - since) / 1000))} guesses a second.`; frames = 0; since = now; }
    }
    await new Promise((done) => setTimeout(done, 60));
  }
}

$("camera").onclick = startCamera;
$("choose").onclick = () => $("file").click();
$("file").onchange = async () => {
  const file = $("file").files[0];
  if (!file) return;
  stopCamera();
  const image = $("photo");
  image.src = URL.createObjectURL(file);
  await image.decode();
  image.hidden = false;
  $("video").hidden = true;
  $("note").hidden = true;
  show(await answer(image, image.naturalWidth, image.naturalHeight, false));
};

// ---------------------------------------------------------------- rules

const ACTIONS = { beep: "play a beep", message: "show a message", count: "count it", send: "send to a web address" };
let rules = [];
try { rules = JSON.parse(localStorage.getItem(`ai-bricks-rules-${S.name}`) || "[]"); } catch { rules = []; }
const save = () => { try { localStorage.setItem(`ai-bricks-rules-${S.name}`, JSON.stringify(rules.map(({ category, threshold, action, text }) => ({ category, threshold, action, text })))); } catch { /* private window */ } };

function beep() {
  const audio = new AudioContext(), tone = audio.createOscillator(), gain = audio.createGain();
  tone.frequency.value = 880; gain.gain.setValueAtTime(0.25, audio.currentTime); gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.4);
  tone.connect(gain).connect(audio.destination); tone.start(); tone.stop(audio.currentTime + 0.4);
  tone.onended = () => audio.close();
}

function banner(text) {
  const box = $("banner");
  box.textContent = text;
  box.hidden = false;
  clearTimeout(banner.timer);
  banner.timer = setTimeout(() => { box.hidden = true; }, 2500);
}

function fire(rule, answer) {
  rule.count = (rule.count || 0) + 1;
  rule.last = performance.now();
  if (rule.action === "beep") beep();
  if (rule.action === "message") banner(rule.text || answer.name);
  if (rule.action === "send" && /^https?:\/\//.test(rule.text || "")) {
    // Sent as plain text so it reaches services like ntfy or IFTTT without extra permission.
    fetch(rule.text, { method: "POST", mode: "no-cors", body: JSON.stringify({ model: S.name, category: answer.name, chance: answer.chance, time: new Date().toISOString() }) }).catch(() => {});
  }
  window.firedRules = (window.firedRules || 0) + 1;
  drawRules();
}

function check(answers) {
  for (const rule of rules) {
    // A rule looks at every answer, so a detector's rule fires for any box of its thing.
    const top = (answers || []).find((a) => a.name === rule.category);
    const sure = !!top && top.chance * 100 >= rule.threshold;
    rule.streak = sure ? (rule.streak || 0) + 1 : Math.min(0, (rule.streak || 0) - 1);
    // It fires once when it becomes sure, then waits until it has been unsure for a moment.
    if (rule.streak >= FRAMES_TO_FIRE && !rule.on && !(rule.last && performance.now() - rule.last < COOLDOWN_MS)) { rule.on = true; fire(rule, top); }
    if (rule.streak <= -FRAMES_TO_FIRE && rule.on) { rule.on = false; drawRules(); }
  }
}

function drawRules() {
  $("rules").replaceChildren(...rules.map((rule, index) => {
    const row = document.createElement("li");
    row.className = `rule${rule.on ? " firing" : ""}`;
    const category = document.createElement("select");
    category.setAttribute("aria-label", "Category");
    S.labels.forEach((name) => category.add(new Option(name, name, false, name === rule.category)));
    category.onchange = () => { rule.category = category.value; save(); };
    const threshold = document.createElement("input");
    Object.assign(threshold, { type: "number", min: RULE.min, max: 99, value: rule.threshold });
    threshold.setAttribute("aria-label", "How sure, in percent");
    threshold.onchange = () => { rule.threshold = Math.min(99, Math.max(RULE.min, Number(threshold.value) || RULE.start)); save(); };
    const action = document.createElement("select");
    action.setAttribute("aria-label", "Action");
    Object.entries(ACTIONS).forEach(([value, label]) => action.add(new Option(label, value, false, value === rule.action)));
    action.onchange = () => { rule.action = action.value; save(); drawRules(); };
    const parts = [text(RULE.words[0]), category, text(RULE.words[1]), threshold, text(RULE.words[2]), action];
    if (rule.action === "message" || rule.action === "send") {
      const extra = document.createElement("input");
      Object.assign(extra, { type: rule.action === "send" ? "url" : "text", value: rule.text || "", placeholder: rule.action === "send" ? "https://ntfy.sh/your-topic" : "Message to show" });
      extra.setAttribute("aria-label", rule.action === "send" ? "Web address" : "Message");
      extra.onchange = () => { rule.text = extra.value; save(); };
      parts.push(extra);
    }
    const counted = text(`${rule.count || 0} ${rule.count === 1 ? "time" : "times"}`);
    counted.className = "count";
    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.onclick = () => { rules.splice(index, 1); save(); drawRules(); };
    row.append(...parts, counted, remove);
    return row;
  }));
}

function text(value) { const span = document.createElement("span"); span.textContent = value; return span; }

$("add").onclick = () => { rules.push({ category: S.labels[0], threshold: RULE.start, action: "beep", text: "" }); save(); drawRules(); };
drawRules();

load().catch((error) => say(`The model couldn't start: ${error && error.message ? error.message : error}`, "bad"));
</script>
</body>
</html>
"""
