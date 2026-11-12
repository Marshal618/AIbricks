"""Packs a trained photo, pose or fast find-things model into a kit for a Raspberry Pi.

The kit is a zip with the model in ONNX form (the same file the web app embeds), its settings, and a
small Python script that runs it on the Pi's camera with ONNX Runtime. No PyTorch is needed on the
Pi, which keeps the install small. The script can switch a GPIO pin or run a command while it sees
something, so a model can drive a light, a buzzer or a motor. It checks itself against the answers
PyTorch gave, like the web app does, so a model that converted wrongly says so.
"""
from __future__ import annotations

import io
import json
import zipfile

from . import training, webapp

_RUN_PY = r'''"""Runs an AI Bricks model on a Raspberry Pi, or any computer with a camera.

    python run.py                      watch the camera and print what it sees
    python run.py --photo cat.jpg      check photos instead
    python run.py --when Cup --pin 17  switch GPIO pin 17 on while it sees Cup
    python run.py --when Cup --command "aplay beep.wav"
    python run.py --check              check the model gives the answers it gave when it was made
"""
import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "settings.json").read_text(encoding="utf-8"))
KIND = S.get("kind", "image")  # "image", "pose", "detect" or "outline"
LABELS = S["labels"]
SEEN, HAND_POINTS = 0.5, 21


class Model:
    def __init__(self):
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(str(HERE / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self.finder = None
        if KIND == "pose":
            self.finder = PointFinder(HERE / "points.task")

    # ------------------------------------------------------------ the network on its own
    def run(self, values):
        if KIND == "detect":
            boxes, scores, labels = self.session.run(None, {"image": values})
            return boxes, scores, labels
        if KIND == "outline":
            return self.session.run(None, {"image": values})[0][0]
        name = "features" if KIND == "pose" else "image"
        return self.session.run(None, {name: values})[0][0]

    # ------------------------------------------------------------ a photo in, answers out
    def answer(self, image, sure):
        """A list of (name, chance, count), surest first. Empty when nothing is there. For an outlining
        model the chance is how much of the photo each thing covers."""
        if KIND == "outline":
            side = S["size"]
            pixels = np.asarray(image.resize((side, side), Image.BILINEAR), dtype=np.float32) / 255
            labels = self.run(np.ascontiguousarray(pixels.transpose(2, 0, 1))[None]).argmax(0)
            shares = [(name, float((labels == i + 1).mean()), 1) for i, name in enumerate(LABELS)]
            return sorted((row for row in shares if row[1] >= 0.005), key=lambda row: -row[1])
        if KIND == "detect":
            side = S["size"]
            pixels = np.asarray(image.resize((side, side), Image.BILINEAR), dtype=np.float32) / 255
            boxes, scores, labels = self.run(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
            found = {}
            for score, label in zip(scores.tolist(), labels.tolist()):
                if score < sure:
                    continue
                name = LABELS[int(label) - 1]
                count, best = found.get(name, (0, 0.0))
                found[name] = (count + 1, max(best, score))
            return sorted(((name, best, count) for name, (count, best) in found.items()), key=lambda row: (-row[2], -row[1]))
        if KIND == "pose":
            features = self.finder.find(image)
            if features is None:
                return []
            logits = self.run(features[None])
        else:
            side = S["size"]
            small = image.resize((side, side), Image.BILINEAR)
            if S["channels"] == 1:
                small = small.convert("L")
            pixels = np.asarray(small, dtype=np.float32) / 255
            pixels = pixels[:, :, None] if pixels.ndim == 2 else pixels
            if S.get("mean"):
                pixels = (pixels - np.array(S["mean"], dtype=np.float32)) / np.array(S["std"], dtype=np.float32)
            logits = self.run(np.ascontiguousarray(pixels.transpose(2, 0, 1))[None].astype(np.float32))
        chances = np.exp(logits - logits.max())
        chances = chances / chances.sum()
        return [(LABELS[i], float(chances[i]), 1) for i in np.argsort(-chances)]


# ---------------------------------------------------------------- poses: MediaPipe finds the points
def scaled(points, width, height):
    side = max(width, height)
    return np.array([[p["x"] * width / side, p["y"] * height / side, p["z"] * width / side] for p in points], dtype=np.float32)


def body_features(xyz, seen):
    centre = (xyz[11] + xyz[12]) / 2
    moved = xyz - centre
    visible = moved[seen > 0]
    scale = float(np.sqrt((visible[:, :2] ** 2).sum(axis=1).mean())) if len(visible) else 1.0
    moved = moved / max(scale, 1e-3) * seen[:, None]
    return np.concatenate([moved.reshape(-1), seen]).astype(np.float32)


def hand_features(hands):
    parts = []
    for side_name in ("Left", "Right"):
        points = hands.get(side_name)
        if points is None:
            parts.append(np.zeros(HAND_POINTS * 3 + 1, dtype=np.float32))
            continue
        moved = points - points[0]
        scale = float(np.sqrt((moved[:, :2] ** 2).sum(axis=1)).max())
        parts.append(np.concatenate([(moved / max(scale, 1e-3)).reshape(-1), [1.0]]).astype(np.float32))
    return np.concatenate(parts)


def features(found, handedness, width, height):
    """Measures points exactly as training did. Points are dicts with x, y, z and visibility."""
    if not found:
        return None
    if S["points"] == "body":
        points = found[0]
        return body_features(scaled(points, width, height), np.array([(p.get("visibility") or 0) > SEEN for p in points], dtype=np.float32))
    hands = {}
    for i, points in enumerate(found):
        side_name = handedness[i] if i < len(handedness) and handedness[i] else "Right"
        if side_name in hands:  # two hands that look like the same side: the second takes the other place
            side_name = "Left" if side_name == "Right" else "Right"
        hands[side_name] = scaled(points, width, height)
    return hand_features(hands)


class PointFinder:
    def __init__(self, path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        options = BaseOptions(model_asset_path=str(path))
        if S["points"] == "body":
            self.finder = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=options, running_mode=vision.RunningMode.IMAGE, num_poses=1))
        else:
            self.finder = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=options, running_mode=vision.RunningMode.IMAGE, num_hands=2))

    def find(self, image):
        frame = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(image)))
        result = self.finder.detect(frame)
        as_dicts = lambda points: [{"x": p.x, "y": p.y, "z": p.z, "visibility": p.visibility} for p in points]
        if S["points"] == "body":
            return features([as_dicts(p) for p in result.pose_landmarks], [], *image.size)
        sides = [handed[0].category_name if handed else None for handed in result.handedness]
        return features([as_dicts(p) for p in result.hand_landmarks], sides, *image.size)


# ---------------------------------------------------------------- the self-check
def check(model):
    """The largest gap between the answers here and the ones PyTorch gave when the kit was made."""
    if KIND == "outline":
        side = S["size"]
        pattern = np.array([math.sin(i * 0.37) * 0.5 + 0.5 for i in range(3 * side * side)], dtype=np.float32).reshape(1, 3, side, side)
        got = model.run(pattern).reshape(-1)[::S["check"]["step"]]
        return max(abs(a - b) / max(1.0, abs(a)) for a, b in zip(S["check"]["values"], got.tolist()))
    if KIND == "detect":
        side = S["size"]
        pattern = np.array([math.sin(i * 0.37) * 0.5 + 0.5 for i in range(3 * side * side)], dtype=np.float32).reshape(3, side, side)
        boxes, scores, _ = model.run(pattern)
        want_scores, want_boxes = S["check"]["scores"], S["check"]["boxes"]
        if len(scores) < len(want_scores):
            return 1.0
        gap = max([abs(a - b) for a, b in zip(want_scores, scores.tolist())], default=0.0)
        return max([gap] + [abs(a - b) / side for a, b in zip(want_boxes, boxes.reshape(-1).tolist())])
    if KIND == "pose":
        pattern = np.array([math.sin(i * 0.37) for i in range(S["features"])], dtype=np.float32)[None]
        sample = S["sample"]
        result = sample["result"]
        sides = [h[0]["categoryName"] for h in result.get("handedness", [])]
        measured = features(result["landmarks"], sides, sample["width"], sample["height"])
        point_gap = float(np.abs(measured - np.array(sample["features"], dtype=np.float32)).max())
    else:
        side, channels = S["size"], S["channels"]
        pattern = np.array([math.sin(i * 0.37) for i in range(channels * side * side)], dtype=np.float32).reshape(1, channels, side, side)
        point_gap = 0.0
    got = model.run(pattern)
    gap = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(got.tolist(), S["check"]))
    return max(gap, point_gap)


# ---------------------------------------------------------------- the camera
class Camera:
    """The Raspberry Pi camera through Picamera2 when there is one, otherwise a USB webcam through OpenCV."""

    def __init__(self, usb=False, index=0):
        self.picam = self.capture = None
        if not usb:
            try:
                from picamera2 import Picamera2
                self.picam = Picamera2()
                self.picam.configure(self.picam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
                self.picam.start()
                time.sleep(1)  # let the exposure settle
                return
            except Exception:
                self.picam = None
        try:
            import cv2
        except ImportError:
            sys.exit("No camera found. Plug in the Raspberry Pi camera, or install OpenCV for a USB webcam: pip install opencv-python-headless")
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            sys.exit(f"Couldn't open camera {index}. Check it's plugged in.")

    def photo(self):
        if self.picam is not None:
            # Picamera2's "RGB888" is stored blue-green-red, so it's flipped here.
            return Image.fromarray(self.picam.capture_array()[:, :, ::-1].copy())
        ok, frame = self.capture.read()
        if not ok:
            return None
        return Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))

    def close(self):
        if self.picam is not None:
            self.picam.stop()
        if self.capture is not None:
            self.capture.release()


# ---------------------------------------------------------------- rules
class Rule:
    """Switches a pin on while the thing is seen, and runs a command each time it starts being seen."""

    def __init__(self, when, pin, command, wait):
        self.when, self.command, self.wait = when, command, wait
        self.on, self.streak, self.last_run, self.pin = False, 0, 0.0, None
        if pin is not None:
            try:
                from gpiozero import LED
            except ImportError:
                sys.exit("Switching pins needs gpiozero, which comes with Raspberry Pi OS. Run setup.sh on the Pi.")
            self.pin = LED(pin)

    def update(self, seen):
        # Two photos in a row to switch on and three to switch off, so one odd guess doesn't flicker it.
        self.streak = max(self.streak, 0) + 1 if seen else min(self.streak, 0) - 1
        if not self.on and self.streak >= 2:
            self.on = True
            if self.pin is not None:
                self.pin.on()
            if self.command and time.time() - self.last_run >= self.wait:
                self.last_run = time.time()
                subprocess.Popen(shlex.split(self.command))
            return f"→ {self.when}: on"
        if self.on and self.streak <= -3:
            self.on = False
            if self.pin is not None:
                self.pin.off()
            return f"→ {self.when}: off"
        return None


def _rounded(answers):
    """An outlining model's shares to the nearest 5%, so it reports changes without printing every photo."""
    return [(name, round(share * 20)) for name, share, _ in answers]


def describe(answers):
    if not answers:
        return S.get("nothing") or "Nothing found."
    if KIND == "detect":
        return ", ".join(f"{name} × {count}" for name, _, count in answers)
    if KIND == "outline":
        return ", ".join(f"{name} {round(share * 100)}%" for name, share, _ in answers)
    name, chance, _ = answers[0]
    return f"{name} {round(chance * 100)}%"


def is_seen(answers, when, sure):
    if KIND == "outline":
        return any(name == when and share >= sure for name, share, _ in answers)
    if KIND == "detect":
        return any(name == when for name, _, _ in answers)
    return bool(answers) and answers[0][0] == when and answers[0][1] >= sure


def main():
    parser = argparse.ArgumentParser(description=f"Runs {S.get('name', 'an AI Bricks model')}. {S.get('about', '')}")
    parser.add_argument("--photo", nargs="+", help="check these photos instead of the camera")
    parser.add_argument("--check", action="store_true", help="check the model gives the same answers as when it was made")
    parser.add_argument("--when", help=f"the name to watch for: one of {', '.join(LABELS)}")
    parser.add_argument("--pin", type=int, help="GPIO pin to switch on while it's seen (BCM number, like 17)")
    parser.add_argument("--command", help="a command to run each time it starts being seen")
    parser.add_argument("--wait", type=float, default=5, help="seconds before the command can run again (default 5)")
    parser.add_argument("--sure", type=float, default=S.get("sure", 0.7),
                        help="how much of the view it must cover, from 0 to 1" if KIND == "outline" else "how sure it must be, from 0 to 1")
    parser.add_argument("--usb", action="store_true", help="use a USB webcam even if there's a Raspberry Pi camera")
    parser.add_argument("--camera", type=int, default=0, help="which USB webcam, when there are several (default 0)")
    parser.add_argument("--json", action="store_true", help="print each answer as a line of JSON, for other programs")
    args = parser.parse_args()
    if args.when is not None and args.when not in LABELS:
        sys.exit(f"--when must be one of: {', '.join(LABELS)}")
    if (args.pin is not None or args.command) and args.when is None:
        sys.exit("Say what to watch for with --when.")

    model = Model()
    if args.check:
        gap = check(model)
        print(f"Largest gap from PyTorch: {gap:.2g}")
        if gap >= 1e-3:
            sys.exit("This model gives different answers here than when it was made, so its guesses may be wrong.")
        print("✓ The model works on this device.")
        return

    def report(answers, where=None):
        if args.json:
            print(json.dumps({"photo": where, "answers": [{"name": n, "chance": round(c, 4), "count": k} for n, c, k in answers]}), flush=True)
        else:
            print(f"{where}: {describe(answers)}" if where else f"{time.strftime('%H:%M:%S')}  {describe(answers)}", flush=True)

    if args.photo:
        for path in args.photo:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            report(model.answer(image, args.sure), path)
        return

    rule = Rule(args.when, args.pin, args.command, args.wait) if args.when else None
    camera = Camera(args.usb, args.camera)
    print(f"Watching. {S.get('about', '')} Press Ctrl+C to stop.", flush=True)
    last, last_answers = None, []
    try:
        while True:
            image = camera.photo()
            if image is None:
                time.sleep(0.1)
                continue
            answers = model.answer(image, args.sure)
            text = describe(answers)
            if args.json or text.split(" ")[0] != (last or "").split(" ")[0] or KIND == "detect" and text != last \
                    or KIND == "outline" and _rounded(answers) != _rounded(last_answers):
                report(answers)
                last, last_answers = text, answers
            if rule:
                change = rule.update(is_seen(answers, args.when, args.sure))
                if change:
                    print(change, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        if rule and rule.pin is not None:
            rule.pin.off()


if __name__ == "__main__":
    main()
'''

_SETUP_SH = r'''#!/bin/sh
# Sets up this model on a Raspberry Pi (64-bit Raspberry Pi OS). Run it once: sh setup.sh
set -e
cd "$(dirname "$0")"
# System packages let the script use the Pi's camera (picamera2) and pins (gpiozero), which come with Raspberry Pi OS.
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
if ! .venv/bin/python -c "import picamera2" 2>/dev/null; then
  echo "No Raspberry Pi camera library found, so installing OpenCV for USB webcams."
  .venv/bin/pip install opencv-python-headless
fi
.venv/bin/python run.py --check
echo
echo "Ready. Start it with: .venv/bin/python run.py"
'''

_README = """# __NAME__ on a Raspberry Pi

__ABOUT__ This folder runs the model on a Raspberry Pi with its camera, without the internet and
without AI Bricks.

## What you need

- A Raspberry Pi 4 or 5 running **64-bit** Raspberry Pi OS. A Pi 3 or Zero 2 W works too, more slowly.
- The Raspberry Pi camera, or a USB webcam.

## Set it up

Copy this folder to the Pi, then in a terminal in the folder:

```
sh setup.sh
```

It installs what the model needs (__NEEDS__) and checks the model gives the same answers it gave in
AI Bricks.

## Use it

```
.venv/bin/python run.py                      # watch the camera and print what it sees
.venv/bin/python run.py --photo photo.jpg    # check a photo
```

__RULES__

Add `--usb` to use a USB webcam when a Raspberry Pi camera is also plugged in, and `--json` to print
each answer as JSON for another program. `--help` lists everything.

## Switching things on

Pins are numbered the Broadcom way (GPIO 17 is physical pin 11). An LED with a 330 Ω resistor between
GPIO 17 and a ground pin is the simplest test. A pin gives 3.3 volts and very little current, so use a
relay or transistor board for anything bigger, like a motor or a lamp.

It switches on after two photos in a row agree, and off after three that don't, so a single odd guess
doesn't make it flicker.

## Start it when the Pi starts

```
crontab -e
```

and add a line like this, with the folder's real path:

```
@reboot cd /home/pi/__FOLDER__ && .venv/bin/python run.py __EXAMPLE__ >> run.log 2>&1
```

## What's here

- `model.onnx`: the trained model in ONNX, an open format, run by ONNX Runtime.
- `settings.json`: its labels and how it reads photos.
__POINTS_FILE__- `run.py`: the script that runs it. It's short and yours to change.
"""


def kit(job_id: str) -> tuple[bytes, str]:
    """The zip and a file name for it."""
    job = webapp._job(job_id, where="a Raspberry Pi", made="a Raspberry Pi kit")
    info = webapp._export(job)
    folder = training._folder(job)
    detect, pose, outlining = job.task == "detect", job.task == "pose", job.task == "outline"
    labels = info["labels"]
    if outlining:
        about, nothing = f"Outlines {', '.join(labels)}, and says how much of the view each one covers.", "Nothing outlined."
        example, sure = labels[0], 0.2
    elif detect:
        about, nothing = f"Finds and counts: {', '.join(labels)}.", "Nothing found."
        example, sure = labels[0], webapp._detect_settings(job).get("sure", 0.5)
    elif pose:
        hands = info.get("points") == "hands"
        about, nothing = f"Sorts {'hand signs' if hands else 'poses'} into: {', '.join(labels)}.", "No hands in view." if hands else "No one in view."
        example, sure = labels[0], 0.7
    else:
        about, nothing, example, sure = f"Sorts photos into: {', '.join(labels)}.", "", labels[0], 0.7
    settings = {"kind": "image", **info, "name": job.projectName, "trainedAt": job.startedAt, "about": about, "nothing": nothing, "sure": sure}
    slug = job.slug or "model"
    name = f"{slug}-pi"
    quoted = f'"{example}"' if any(c in example for c in " '\"$`\\") else example
    rule = f"--when {quoted} --pin 17"
    rules = (f"To switch on GPIO pin 17 while {example} covers a fifth of the view, or play a sound each time it starts:\n\n" if outlining else
             f"To switch on GPIO pin 17 while it {'finds' if detect else 'sees'} {example}, or play a sound each time it starts:\n\n")
    rules += (
             f"```\n.venv/bin/python run.py {rule}\n"
             f".venv/bin/python run.py --when {quoted} --command \"aplay /usr/share/sounds/alsa/Front_Center.wav\"\n```\n\n"
             f"`--when` takes one of: {', '.join(labels)}. "
             + ("Add `--sure 0.5` to need it to cover half the view (it uses 0.2 now)." if outlining
                else f"Add `--sure 0.8` to need more certainty (it uses {sure:g} now)." if detect else "Add `--sure 0.9` to need more certainty (it uses 0.7 now)."))
    readme = (_README
              .replace("__NAME__", job.projectName)
              .replace("__ABOUT__", about)
              .replace("__NEEDS__", "ONNX Runtime and MediaPipe" if pose else "ONNX Runtime")
              .replace("__RULES__", rules)
              .replace("__FOLDER__", name)
              .replace("__EXAMPLE__", rule)
              .replace("__POINTS_FILE__", "- `points.task`: MediaPipe's model that finds the points.\n" if pose else ""))
    requirements = "onnxruntime>=1.17\nnumpy\npillow\n" + ("mediapipe>=0.10.14\n" if pose else "")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        def add(path: str, data: bytes | str, mode: int = 0o644) -> None:
            entry = zipfile.ZipInfo(f"{name}/{path}", date_time=(2020, 1, 1, 0, 0, 0))
            entry.external_attr = mode << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
        add("model.onnx", (folder / webapp.APP_FILE).read_bytes())
        add("settings.json", json.dumps(settings, indent=1))
        if pose:
            add("points.task", (folder / "model" / "points.task").read_bytes())
        add("run.py", _RUN_PY, 0o755)
        add("setup.sh", _SETUP_SH, 0o755)
        add("requirements.txt", requirements)
        add("README.md", readme)
    return buffer.getvalue(), f"{name}.zip"
