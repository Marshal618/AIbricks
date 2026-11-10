"""Sort body poses and hand signs: photos sorted into categories, like any photo dataset.

MediaPipe, running on this computer, finds the points of a body (shoulders, elbows, knees and so on)
or of each hand (every finger joint) in a photo. A small network then learns which category those
points make. Points don't depend on clothes, light or background, which is why a few dozen photos
are enough. The point-finding models are downloaded once, about 6 MB for bodies or 8 MB for hands,
and kept with each trained model, so it never needs the internet again.
"""
from __future__ import annotations

from . import codegen, datasets, training, versions
from .training import TrainingError

POINTS = ("body", "hands")
MIN_PER_CATEGORY = training.MIN_PER_CATEGORY
SUGGESTED_PER_CATEGORY = 20

_MODEL_PY = '''"""Sorts __WHAT__. MediaPipe finds the points, then a small network names the category they make."""
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

POINTS = "__POINTS__"  # "body" or "hands"
FINDERS = {
    "body": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
             "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "hands": ("hand_landmarker.task",
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
              "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
}
BODY_POINTS, HAND_POINTS = 33, 21
FEATURES = BODY_POINTS * 4 if POINTS == "body" else 2 * (HAND_POINTS * 3 + 1)
SEEN = 0.5  # how sure MediaPipe must be that a body point is in view


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finder_file(model_folder=Path("model")):
    """The MediaPipe model that finds the points: kept with the trained model, or downloaded once."""
    name, url, sha = FINDERS[POINTS]
    kept = Path(model_folder) / "points.task"
    if kept.is_file():
        return kept
    cache = Path(os.environ.get("AI_BRICKS_POINTS_CACHE") or Path.home() / ".cache" / "aibricks-points")
    path = cache / name
    if path.is_file() and _sha256(path) == sha:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=60) as reply, open(partial, "wb") as file:
        shutil.copyfileobj(reply, file)
    if _sha256(partial) != sha:
        partial.unlink()
        raise RuntimeError("The point-finding model that downloaded isn't the expected one.")
    os.replace(partial, path)
    return path


def load_image(source):
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class PointFinder:
    """Finds a body's 33 points, or up to two hands' 21 points each."""

    def __init__(self, path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        options = BaseOptions(model_asset_path=str(path))
        if POINTS == "body":
            self.finder = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=options, running_mode=vision.RunningMode.IMAGE, num_poses=1))
        else:
            self.finder = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=options, running_mode=vision.RunningMode.IMAGE, num_hands=2))

    def find(self, image):
        """Returns (features or None when nothing was found, the points to draw as 0-1 shares of the photo)."""
        width, height = image.size
        frame = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(image)))
        found = self.finder.detect(frame)
        # Points come as shares of the width and height. Scaling both by the longer side keeps shapes true.
        side = max(width, height)
        sx, sy = width / side, height / side
        if POINTS == "body":
            if not found.pose_landmarks:
                return None, []
            points = found.pose_landmarks[0]
            xyz = np.array([[p.x * sx, p.y * sy, p.z * sx] for p in points], dtype=np.float32)
            seen = np.array([(p.visibility or 0.0) > SEEN for p in points], dtype=np.float32)
            draw = [[[round(p.x, 4), round(p.y, 4), round(float(p.visibility or 0.0), 3)] for p in points]]
            return body_features(xyz, seen), draw
        if not found.hand_landmarks:
            return None, []
        hands, draw = {}, []
        for points, handed in zip(found.hand_landmarks, found.handedness):
            side_name = handed[0].category_name if handed else "Right"
            if side_name in hands:  # two hands that look like the same side: the second takes the other place
                side_name = "Left" if side_name == "Right" else "Right"
            hands[side_name] = np.array([[p.x * sx, p.y * sy, p.z * sx] for p in points], dtype=np.float32)
            draw.append([[round(p.x, 4), round(p.y, 4), 1.0] for p in points])
        return hand_features(hands), draw


def body_features(xyz, seen):
    """Points measured from between the shoulders, in units of the body's size, so where a person
    stands and how far from the camera they are don't matter. Points out of view are zeros."""
    centre = (xyz[11] + xyz[12]) / 2
    moved = xyz - centre
    visible = moved[seen > 0]
    scale = float(np.sqrt((visible[:, :2] ** 2).sum(axis=1).mean())) if len(visible) else 1.0
    moved = moved / max(scale, 1e-3) * seen[:, None]
    return np.concatenate([moved.reshape(-1), seen]).astype(np.float32)


def hand_features(hands):
    """Each hand's points measured from its wrist, in units of the hand's size, with a mark for each
    hand that's in view. The left hand always comes first."""
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


def build(categories):
    return nn.Sequential(
        nn.Linear(FEATURES, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, len(categories)),
    )


def answers(network, features, categories):
    with torch.no_grad():
        chances = torch.softmax(network(torch.from_numpy(features)[None]), dim=1)[0].tolist()
    ranked = sorted(zip(categories, chances), key=lambda pair: -pair[1])
    return [{"category": name, "chance": round(chance, 4)} for name, chance in ranked]


class PoseSorter:
    def __init__(self, folder="model"):
        folder = Path(folder)
        settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        self.categories = settings["categories"]
        self.network = build(self.categories)
        self.network.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        self.network.eval()
        self.finder = PointFinder(finder_file(folder))

    def check(self, image):
        """{"found", "answers", "points"}. Nothing is guessed when no body or hand is in view."""
        features, points = self.finder.find(image)
        if features is None:
            return {"found": False, "answers": [], "points": []}
        return {"found": True, "answers": answers(self.network, features, self.categories), "points": points}
'''

_TRAIN_PY = '''"""Learns to sort __WHAT__. Put photos in data/images/<category>/, then run: python train.py"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import POINTS, PointFinder, answers, build, finder_file, load_image

DATA = Path("data/images")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HOLD_BACK = 0.2
EPOCHS = 300
REPORTS = 20
NOTHING = "Nothing found"
WHAT = "body" if POINTS == "body" else "hands"


def fingerprint(path):
    return hashlib.sha256(path.name.encode("utf-8")).hexdigest()


def main():
    torch.manual_seed(0)
    categories = sorted(folder.name for folder in DATA.iterdir() if folder.is_dir()) if DATA.is_dir() else []
    train, check = [], []
    for index, name in enumerate(categories):
        photos = sorted((path for path in (DATA / name).iterdir() if path.suffix.lower() in EXTENSIONS), key=fingerprint)
        held = max(1, round(len(photos) * HOLD_BACK)) if len(photos) >= 2 else 0
        check += [(path, index) for path in photos[:held]]
        train += [(path, index) for path in photos[held:]]
    if len(categories) < 2 or not train:
        sys.exit("Add photos to at least 2 categories.")
    try:
        path = finder_file()
    except Exception as problem:
        sys.exit(f"The model that finds the points couldn't be downloaded. The first run needs the internet once. ({type(problem).__name__})")
    finder = PointFinder(path)
    print(f"Training on {len(train)} samples, checking on {len(check)}, using cpu.", flush=True)

    everything = train + check
    found = []
    for step, (photo, _) in enumerate(everything, 1):
        try:
            features, _ = finder.find(load_image(photo))
        except Exception:
            features = None
        found.append(features)
        print(f"Step {step}/{len(everything)}", flush=True)
    train_found = [(features, label) for features, (_, label) in zip(found, train) if features is not None]
    check_found = [(features, label) for features, (_, label) in zip(found[len(train):], check) if features is not None]
    missing = len(train) - len(train_found)
    if len({label for _, label in train_found}) < 2:
        sys.exit(f"No {WHAT} could be seen in most photos, so there's nothing to learn from. "
                 f"Take photos where the {'whole body' if POINTS == 'body' else 'hands'} can be seen clearly.")

    x = torch.from_numpy(np.stack([features for features, _ in train_found]))
    y = torch.tensor([label for _, label in train_found])
    cx = torch.from_numpy(np.stack([features for features, _ in check_found])) if check_found else None
    cy = torch.tensor([label for _, label in check_found]) if check_found else None
    network = build(categories)
    optimiser = torch.optim.AdamW(network.parameters(), lr=3e-3, weight_decay=1e-3)
    # Rare categories count for as much as common ones.
    counts = torch.bincount(y, minlength=len(categories)).clamp(min=1).float()
    loss_function = nn.CrossEntropyLoss(weight=counts.sum() / (len(categories) * counts))
    plain = nn.CrossEntropyLoss()
    best, best_loss = None, float("inf")
    for epoch in range(1, EPOCHS + 1):
        network.train()
        # A little jitter, as no two people hold a pose in exactly the same way.
        noisy = x + torch.randn_like(x) * 0.03 * (x != 0)
        loss = loss_function(network(noisy), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if epoch % (EPOCHS // REPORTS) == 0:
            network.eval()
            with torch.no_grad():
                train_loss = plain(network(x), y).item()
                if cx is not None:
                    outputs = network(cx)
                    check_loss = plain(outputs, cy).item()
                    right = (outputs.argmax(1) == cy).sum().item()
                else:
                    check_loss, right = train_loss, 0
            # Photos where nothing was found count as wrong, as the model couldn't answer.
            accuracy = 100 * right / len(check) if check else 0.0
            if check_loss < best_loss:
                best_loss, best = check_loss, {key: value.clone() for key, value in network.state_dict().items()}
            print(f"Epoch {epoch // (EPOCHS // REPORTS)}/{REPORTS}  training loss {train_loss:.4f}  check loss {check_loss:.4f}  check accuracy {accuracy:.1f}%", flush=True)

    network.load_state_dict(best)
    network.eval()
    results = []
    for (photo, label), features in zip(check, found[len(train):]):
        if features is None:
            results.append({"file": photo.name, "actual": categories[label], "guess": NOTHING, "chance": 0.0})
            continue
        top = answers(network, features, categories)[0]
        results.append({"file": photo.name, "actual": categories[label], "guess": top["category"], "chance": top["chance"]})

    partial = Path("model.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    torch.save(best, partial / "weights.pt")
    shutil.copy2(path, partial / "points.task")
    (partial / "settings.json").write_text(json.dumps({"categories": categories, "points": POINTS}), encoding="utf-8")
    shutil.rmtree("model", ignore_errors=True)
    os.replace(partial, "model")
    with open("data/review.json", "w", encoding="utf-8") as file:
        json.dump(results, file)
    with open("data/metrics.json", "w", encoding="utf-8") as file:
        json.dump({
            "task": "pose", "points": POINTS, "total": len(results), "correct": sum(row["guess"] == row["actual"] for row in results),
            "nothingFoundTraining": missing, "nothingFoundChecking": sum(row["guess"] == NOTHING for row in results),
            "trainedOn": len(train_found),
        }, file)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''"""Sorts __WHAT__ in photos. Train first, then run: python predict.py photo.jpg"""
import json
import sys

from model import PoseSorter, load_image


def main():
    paths = [item for item in sys.argv[1:] if item != "--json"]
    if not paths:
        sys.exit("Name at least one photo, for example: python predict.py photo.jpg")
    sorter = PoseSorter()
    results = [{"file": path, **sorter.check(load_image(path))} for path in paths]
    if "--json" in sys.argv:
        print(json.dumps(results))
        return
    for result in results:
        if not result["found"]:
            print(f"{result['file']}: nothing found")
            continue
        top = result["answers"][0]
        print(f"{result['file']}: {top['category']} ({top['chance']:.0%})")


if __name__ == "__main__":
    main()
'''

_README = """# {name}

Sorts {what}, trained in AI Bricks. MediaPipe finds the {points_text} in each photo, then a small
network learns which category they make. Your photos aren't included in this download.

## Use it

```
pip install -r requirements.txt
python predict.py photo.jpg
```

The point-finding model is in `model/points.task`, so it works without the internet.
MediaPipe's models are from Google, under the Apache 2.0 licence.

## Train it again

Put photos in `data/images/<category>/`, one folder per category, then run `python train.py`.
"""

LIVE_RUNNER = """
import base64, contextlib, io, json, sys
sys.path.insert(0, '.')
with contextlib.redirect_stdout(sys.stderr):
    from model import PoseSorter, load_image
    sorter = PoseSorter()
print('ready', flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        print(json.dumps(sorter.check(load_image(io.BytesIO(base64.b64decode(request['data']))))), flush=True)
    except Exception as problem:
        print(json.dumps({'error': f'{type(problem).__name__}: {problem}'[:300]}), flush=True)
"""


def _what(points: str) -> str:
    return "body poses" if points == "body" else "hand signs"


def project_files(name: str, points: str) -> dict[str, str]:
    what = _what(points)
    fill = lambda text: text.replace("__POINTS__", points).replace("__WHAT__", what)  # noqa: E731
    return {
        "model.py": fill(_MODEL_PY), "train.py": fill(_TRAIN_PY), "predict.py": fill(_PREDICT_PY),
        "README.md": _README.format(name=name, what=what, points_text="33 body points" if points == "body" else "21 points of each hand"),
        "requirements.txt": "torch>=2.2\nmediapipe>=0.10.14\nnumpy\npillow\n",
    }


def is_pose(meta: dict) -> bool:
    return meta.get("kind") == "image" and meta.get("goal") == "pose"


def points_of(meta: dict) -> str:
    return meta.get("points") if meta.get("points") in POINTS else "body"


def start(dataset: str) -> training.Job:
    with training._lock:
        info = datasets.get(dataset)
        if not is_pose(info):
            raise TrainingError(422, "This trainer is for “sort poses and hand signs” datasets.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        names = info["categories"]
        if len(names) < 2:
            raise TrainingError(422, "Add at least 2 categories.")
        problems = training.readiness_problems(names, dataset)
        if problems:
            raise TrainingError(422, " ".join(problems))
        amounts = {name: info["counts"].get(name, 0) for name in names}
        files = project_files(info["name"], points_of(info))
        return training.start_run(info["name"], codegen.slug(info["name"]), codegen.build_hash(files), files,
                                  amounts, dataset, versions.freeze, "pose")


def mistakes(job_id: str) -> dict:
    """The held-back photos the newest pose model got wrong, most confident first."""
    job = training.stored(job_id)
    if job.task != "pose" or not job.hasModel:
        raise TrainingError(404, "That model doesn't sort poses.")
    review = job.review or {}
    return {"mistakes": review.get("mistakes", []), "total": review.get("mistakeCount", 0)}
