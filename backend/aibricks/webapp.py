"""Turns a trained photo model into a single web page that runs on its own, in any browser.

The model is exported to ONNX, an open format for trained models, and embedded in the page. The
page runs it with ONNX Runtime Web, so photos are checked on the device and never uploaded. It
also checks itself when it opens: it runs a fixed test pattern and compares the answer with the
one PyTorch gave, so a model that converted wrongly says so instead of guessing quietly.
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


def _export(job: training.Job) -> dict:
    """Exports the model once. A trained model never changes, so later pages reuse the file."""
    folder = training._folder(job)
    info_path, model_path = folder / APP_INFO, folder / APP_FILE
    if info_path.is_file() and model_path.is_file():
        return json.loads(info_path.read_text(encoding="utf-8"))
    try:
        with training._predictions:
            done = subprocess.run(
                [sys.executable, "-I", "-c", _EXPORT], cwd=folder, env=training._environment(folder) | {"OMP_NUM_THREADS": "2"},
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


def _job(job_id: str) -> training.Job:
    job = training.stored(job_id)
    if job.task != "image":
        raise TrainingError(422, "Apps can be made from photo models for now.")
    if job.status == "running":
        raise TrainingError(409, "This model is still training.")
    if not job.hasModel:
        raise TrainingError(409, "This run didn't save a model.")
    return job


def page(job_id: str) -> tuple[str, str]:
    """The finished page and a file name for it."""
    job = _job(job_id)
    info = _export(job)
    model = (training._folder(job) / APP_FILE).read_bytes()
    if len(model) > MAX_APP_MODEL_BYTES:
        raise TrainingError(413, f"This model is {len(model) // (1024 * 1024)} MB, too big to put in a web page. Smaller models, like the ones built from blocks, work best.")
    # No "<" at all, so nothing in a name can end the script block or change how it's read.
    settings = json.dumps({**info, "name": job.projectName, "trainedAt": job.startedAt}).replace("<", "\\u003c")
    text = (_PAGE
            .replace("__TITLE__", html.escape(job.projectName))
            .replace("__RUNTIME__", RUNTIME_URL)
            .replace("__INTEGRITY__", RUNTIME_INTEGRITY)
            .replace("__SETTINGS__", settings)
            .replace("__MODEL__", base64.b64encode(model).decode()))
    return text, f"{job.slug or 'model'}-app.html"


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
    Made with AI Bricks. Photos are checked on this device and never uploaded, unless a rule sends a result to a web address you typed.
  </footer>
</main>
<div class="banner" id="banner" hidden role="alert"></div>

<script id="settings" type="application/json">__SETTINGS__</script>
<script id="model" type="application/octet-stream">__MODEL__</script>
<script src="__RUNTIME__ort.min.js" integrity="__INTEGRITY__" crossorigin="anonymous"></script>
<script>
"use strict";
const S = JSON.parse(document.getElementById("settings").textContent);
const $ = (id) => document.getElementById(id);
const FRAMES_TO_FIRE = 3;
const COOLDOWN_MS = 2000;
let session = null, stream = null, running = false;

$("about").textContent = `Sorts photos into: ${S.labels.join(", ")}. Trained ${new Date(S.trainedAt * 1000).toLocaleDateString()}.`;

function say(text, tone) {
  const box = $("status");
  box.textContent = text;
  box.className = `small status ${tone || ""}`;
}

// ---------------------------------------------------------------- the model

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

function ranked(logits) {
  const top = Math.max(...logits), exps = logits.map((v) => Math.exp(v - top)), total = exps.reduce((a, b) => a + b, 0);
  return S.labels.map((name, i) => ({ name, chance: exps[i] / total })).sort((a, b) => b.chance - a.chance);
}

function show(answers) {
  $("answers").replaceChildren(...answers.map((a) => {
    const row = document.createElement("li");
    row.innerHTML = `<span class="name"></span><span class="track"><span></span></span><span class="value"></span>`;
    row.children[0].textContent = a.name;
    row.children[1].firstChild.style.width = `${a.chance * 100}%`;
    row.children[2].textContent = `${Math.round(a.chance * 100)}%`;
    return row;
  }));
  check(answers[0]);
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
  $("note").hidden = false;
  $("note").textContent = "Camera off.";
  $("rate").textContent = "";
}

async function loop() {
  let frames = 0, since = performance.now();
  while (running) {
    const video = $("video");
    if (video.videoWidth && !document.hidden) {
      show(ranked(await run(prepare(video, video.videoWidth, video.videoHeight))));
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
  show(ranked(await run(prepare(image, image.naturalWidth, image.naturalHeight))));
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

function check(top) {
  for (const rule of rules) {
    const sure = top.name === rule.category && top.chance * 100 >= rule.threshold;
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
    Object.assign(threshold, { type: "number", min: 50, max: 99, value: rule.threshold });
    threshold.setAttribute("aria-label", "How sure, in percent");
    threshold.onchange = () => { rule.threshold = Math.min(99, Math.max(50, Number(threshold.value) || 80)); save(); };
    const action = document.createElement("select");
    action.setAttribute("aria-label", "Action");
    Object.entries(ACTIONS).forEach(([value, label]) => action.add(new Option(label, value, false, value === rule.action)));
    action.onchange = () => { rule.action = action.value; save(); drawRules(); };
    const parts = [text("When it sees"), category, text("at least"), threshold, text("% sure,"), action];
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

$("add").onclick = () => { rules.push({ category: S.labels[0], threshold: 80, action: "beep", text: "" }); save(); drawRules(); };
drawRules();

load().catch((error) => say(`The model couldn't start: ${error && error.message ? error.message : error}`, "bad"));
</script>
</body>
</html>
"""
