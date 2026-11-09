"""Spot what's unusual: a model that learns from photos of normal things only.

It follows PatchCore (Roth et al., 2022). A pretrained ResNet-18 describes every small patch of each
normal photo, and a sample of those descriptions is kept as a memory. A new photo's patches are
compared with the memory: a patch unlike anything normal is unusual, and so is the photo if its most
unusual patch is far enough from normal.

The line between normal and unusual is set from the training photos, never from the photos it's
checked on: each half of them is scored against a memory of the other half, which shows how far
unseen normal photos usually are. The held-back normal photos, and any Unusual photos, are kept
apart to check it honestly. Unusual photos are never learned from.
"""
from __future__ import annotations

import json

from . import codegen, datasets, training, versions
from .training import TrainingError

NORMAL, UNUSUAL = "Normal", "Unusual"
CATEGORIES = [NORMAL, UNUSUAL]
MIN_NORMAL = 10
SUGGESTED_NORMAL = 30
MIN_FACTOR, MAX_FACTOR = 0.5, 3.0

_MODEL_PY = '''"""Spots unusual photos after learning from normal ones only, following PatchCore (Roth et al., 2022).

A pretrained ResNet-18 describes each small patch of a photo. A sample of the patches from normal photos
is kept as a memory. A new photo's unusualness is how far its least normal patch is from that memory.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

IMAGE_SIZE = 224
MEMORY_SHARE = 0.1  # the share of normal patches kept, chosen to cover them evenly
MAX_PATCHES = 60000
LABELS = ["Unusual", "Normal"]


def best_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_image(source):
    return TRANSFORM(Image.open(source).convert("RGB"))


class PatchFeatures(nn.Module):
    """Mid-level ResNet-18 features for each patch, each blended with its neighbours."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        middle = self.layer2(self.stem(images))
        deep = self.layer3(middle)
        middle = F.avg_pool2d(middle, 3, 1, 1)
        deep = F.interpolate(F.avg_pool2d(deep, 3, 1, 1), size=middle.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([middle, deep], dim=1)  # images x 384 x 28 x 28


def patches(extractor, images, device, batch_size=16, progress=None):
    """Every patch of every image, as rows, with the grid size."""
    rows, grid = [], None
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            features = extractor(torch.stack(images[start:start + batch_size]).to(device)).cpu()
            grid = features.shape[-2:]
            rows.append(features.permute(0, 2, 3, 1).reshape(len(features), -1, features.shape[1]))
            if progress:
                progress(min(start + batch_size, len(images)))
    return torch.cat(rows), grid


def coreset(features, share=MEMORY_SHARE, seed=0):
    """Greedy k-centre selection: each pick is the patch farthest from those already picked."""
    generator = torch.Generator().manual_seed(seed)
    if len(features) > MAX_PATCHES:
        features = features[torch.randperm(len(features), generator=generator)[:MAX_PATCHES]]
    count = max(1, int(len(features) * share))
    if count >= len(features):
        return features
    reduced = features @ (torch.randn(features.shape[1], 128, generator=generator) / math.sqrt(128))
    chosen = [0]
    distance = ((reduced - reduced[0]) ** 2).sum(dim=1)
    for _ in range(count - 1):
        index = int(distance.argmax())
        chosen.append(index)
        distance = torch.minimum(distance, ((reduced - reduced[index]) ** 2).sum(dim=1))
    return features[chosen]


def patch_scores(image_patches, memory, chunk=4096):
    """Each patch's distance to the nearest remembered normal patch."""
    best = torch.full((len(image_patches),), float("inf"))
    for start in range(0, len(memory), chunk):
        best = torch.minimum(best, torch.cdist(image_patches, memory[start:start + chunk]).min(dim=1).values)
    return best


def answer(scores, grid, threshold):
    """Normal or unusual, how sure, and where the unusual parts are."""
    worst = float(scores.max())
    ratio = worst / threshold
    unusual = 1 / (1 + math.exp(-6 * (ratio - 1)))
    ranked = sorted([{"category": "Unusual", "chance": unusual}, {"category": "Normal", "chance": 1 - unusual}], key=lambda a: -a["chance"])
    heat = (scores / threshold).clamp(0, 1.5) / 1.5
    return {
        "answers": ranked, "unusualness": round(ratio, 3),
        "heatmap": {"width": int(grid[1]), "height": int(grid[0]), "values": [round(v, 3) for v in heat.tolist()],
                    "method": "patchcore", "category": ranked[0]["category"]},
    }


class Spotter:
    def __init__(self, folder="model", device="cpu"):
        folder = Path(folder)
        self.device = device
        self.extractor = PatchFeatures().to(device)
        self.memory = torch.load(folder / "memory.pt", map_location="cpu", weights_only=True)
        self.settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))

    @property
    def threshold(self):
        return self.settings["threshold"] * self.settings.get("factor", 1.0)

    def check(self, image):
        rows, grid = patches(self.extractor, [image], self.device)
        return answer(patch_scores(rows[0], self.memory), grid, self.threshold)
'''

_TRAIN_PY = '''"""Learns what normal looks like. Put normal photos in data/images/Normal/ and, optionally, a few
unusual ones in data/images/Unusual/ to check it with. Then run: python train.py"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from model import IMAGE_SIZE, PatchFeatures, best_device, coreset, load_image, patch_scores, patches, answer

DATA = Path("data/images")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HOLD_BACK = 0.2


def files(name):
    folder = DATA / name
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in EXTENSIONS) if folder.is_dir() else []


def fingerprint(path):
    return hashlib.sha256(path.name.encode("utf-8")).hexdigest()


def main():
    normal, unusual = files("Normal"), files("Unusual")
    if len(normal) < 5:
        sys.exit("Add at least 5 normal photos.")
    normal = sorted(normal, key=fingerprint)
    held = max(1, int(len(normal) * HOLD_BACK))
    check_normal, train = normal[:held], normal[held:]
    device = best_device()
    try:
        extractor = PatchFeatures().to(device)
    except Exception as problem:
        sys.exit(f"The pretrained ResNet-18 couldn't be loaded. The first run downloads it, about 45 MB. ({type(problem).__name__})")
    print(f"Training on {len(train)} samples, checking on {len(check_normal) + len(unusual)}, using {device}.", flush=True)
    everything = train + check_normal + unusual
    total = len(everything)
    rows, grid = patches(extractor, [load_image(path) for path in everything], device, progress=lambda done: print(f"Step {done}/{total}", flush=True))
    train_rows = rows[:len(train)]

    # How far unseen normal photos usually are: each half scored against a memory of the other half.
    half = len(train) // 2
    unseen = []
    if half >= 1:
        for learn, score in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
            memory = coreset(train_rows[learn].reshape(-1, train_rows.shape[-1]))
            unseen += [float(patch_scores(image, memory).max()) for image in train_rows[score]]
    memory = coreset(train_rows.reshape(-1, train_rows.shape[-1]))
    if unseen:
        unseen.sort()
        threshold = unseen[-1] * 1.05  # just above the most unusual unseen normal photo
    else:
        threshold = float(patch_scores(train_rows[0], memory).max()) * 2 + 1e-6

    results = []
    for path, image, actual in zip(check_normal + unusual, rows[len(train):], ["Normal"] * len(check_normal) + ["Unusual"] * len(unusual)):
        found = answer(patch_scores(image, memory), grid, threshold)
        top = found["answers"][0]
        results.append({"file": path.name, "actual": actual, "guess": top["category"], "chance": top["chance"], "unusualness": found["unusualness"]})
    right = sum(row["guess"] == row["actual"] for row in results)
    normals = [row for row in results if row["actual"] == "Normal"]
    odd = [row for row in results if row["actual"] == "Unusual"]
    pairs = [(o["unusualness"], n["unusualness"]) for o in odd for n in normals]
    auroc = (sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)) if pairs else None

    partial = Path("model.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    torch.save(memory, partial / "memory.pt")
    (partial / "settings.json").write_text(json.dumps({"threshold": threshold, "factor": 1.0, "size": IMAGE_SIZE}), encoding="utf-8")
    shutil.rmtree("model", ignore_errors=True)
    os.replace(partial, "model")
    with open("data/review.json", "w", encoding="utf-8") as file:
        json.dump(results, file)
    with open("data/metrics.json", "w", encoding="utf-8") as file:
        json.dump({
            "task": "unusual", "total": len(results), "correct": right, "auroc": auroc, "memory": len(memory),
            "falseAlarms": sum(row["guess"] == "Unusual" for row in normals), "checkedNormal": len(normals),
            "caught": sum(row["guess"] == "Unusual" for row in odd), "checkedUnusual": len(odd),
        }, file)
    print(f"Epoch 1/1  training loss 0.0000  check loss 0.0000  check accuracy {100 * right / len(results):.1f}%", flush=True)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''"""Says whether photos look normal. Train first, then run: python predict.py photo.jpg"""
import json
import sys

from model import Spotter, load_image


def main():
    paths = [item for item in sys.argv[1:] if item != "--json"]
    if not paths:
        sys.exit("Name at least one photo, for example: python predict.py photo.jpg")
    spotter = Spotter()
    results = [{"file": path, **spotter.check(load_image(path))} for path in paths]
    if "--json" in sys.argv:
        print(json.dumps(results))
        return
    for result in results:
        top = result["answers"][0]
        print(f"{result['file']}: {top['category']} ({result['unusualness']:.2f} times the line for unusual)")


if __name__ == "__main__":
    main()
'''

_README = """# {name}

Spots unusual photos after learning from normal ones only, trained in AI Bricks. It follows PatchCore
(Roth et al., 2022), using a pretrained ResNet-18. Your photos aren't included in this download.

## Use it

```
pip install -r requirements.txt
python predict.py photo.jpg
```

The first run downloads ResNet-18, about 45 MB. The line between normal and unusual is in
`model/settings.json`: raise `factor` above 1 for fewer false alarms, or lower it to catch more.

## Train it again

Put normal photos in `data/images/Normal/`, and optionally a few unusual ones in
`data/images/Unusual/` to check it with, then run `python train.py`.
"""

LIVE_RUNNER = """
import base64, contextlib, io, json, sys
sys.path.insert(0, '.')
with contextlib.redirect_stdout(sys.stderr):
    from model import Spotter, load_image, best_device
    spotter = Spotter()
print('ready', flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        print(json.dumps(spotter.check(load_image(io.BytesIO(base64.b64decode(request['data']))))), flush=True)
    except Exception as problem:
        print(json.dumps({'error': f'{type(problem).__name__}: {problem}'[:300]}), flush=True)
"""


def project_files(name: str) -> dict[str, str]:
    return {
        "model.py": _MODEL_PY, "train.py": _TRAIN_PY, "predict.py": _PREDICT_PY,
        "README.md": _README.format(name=name), "requirements.txt": "torch>=2.2\ntorchvision>=0.17\npillow\n",
    }


def is_unusual(meta: dict) -> bool:
    return meta.get("kind") == "image" and meta.get("goal") == "unusual"


def start(dataset: str) -> training.Job:
    with training._lock:
        info = datasets.get(dataset)
        if not is_unusual(info):
            raise TrainingError(422, "This trainer is for “spot what's unusual” datasets.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        amounts = {name: info["counts"].get(name, 0) for name in CATEGORIES}
        if amounts[NORMAL] < MIN_NORMAL:
            raise TrainingError(422, f"Add at least {MIN_NORMAL} normal photos. It has {amounts[NORMAL]}.")
        files = project_files(info["name"])
        names = [name for name in CATEGORIES if amounts[name]]
        held = max(1, int(amounts[NORMAL] * 0.2))
        baseline = max(held, amounts[UNUSUAL]) / (held + amounts[UNUSUAL])
        return training.start_run(info["name"], codegen.slug(info["name"]), codegen.build_hash(files), files,
                                  {name: amounts[name] for name in names}, dataset, versions.freeze, "anomaly", baseline)


def set_factor(job_id: str, factor: float) -> dict:
    """Moves the line for unusual: above 1 means fewer false alarms, below 1 catches more."""
    if not MIN_FACTOR <= factor <= MAX_FACTOR:
        raise TrainingError(422, f"Choose a value between {MIN_FACTOR} and {MAX_FACTOR}.")
    job = training.stored(job_id)
    if job.task != "anomaly" or not job.hasModel:
        raise TrainingError(404, "That model doesn't spot unusual photos.")
    path = training._folder(job) / "model" / "settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    settings["factor"] = factor
    path.write_text(json.dumps(settings), encoding="utf-8")
    training.close_live(job.id)
    return {"factor": factor, "threshold": settings["threshold"]}


def settings(job_id: str) -> dict:
    job = training.stored(job_id)
    if job.task != "anomaly" or not job.hasModel:
        raise TrainingError(404, "That model doesn't spot unusual photos.")
    data = json.loads((training._folder(job) / "model" / "settings.json").read_text(encoding="utf-8"))
    review = json.loads(json.dumps(job.review or {}))
    return {"factor": data.get("factor", 1.0), "metrics": job.metrics, "review": review}
