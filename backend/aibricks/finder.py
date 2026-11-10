"""Find things in photos: draw boxes around each thing, then train a model that draws them itself.

The model is a Faster R-CNN detector (Ren et al., 2015) with a MobileNetV3 backbone, pretrained on
COCO by torchvision. Only its last layer is replaced to suit the dataset's own things, so a few dozen
boxed photos are enough to start (fine-tuning, as in torchvision's detection tutorial). It runs on
this computer; the pretrained weights, about 75 MB, are downloaded once.

Photos wait in the dataset's photo folder (the same inbox unsorted photos use). A photo is done once
its boxes are saved, even with no boxes: "nothing here" photos teach it what not to find. Boxes are
stored as shares of the photo's width and height, so they don't depend on its size.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from . import codegen, datasets, labelling, training, versions
from .training import TrainingError

MIN_PHOTOS = 10
MIN_BOXES = 5
SUGGESTED_BOXES = 30
MAX_BOXES = 100
MIN_SURE, MAX_SURE = 0.2, 0.95


def is_find(meta: dict) -> bool:
    return meta.get("kind") == "image" and meta.get("goal") == "find"


def _boxes_file(dataset: str) -> Path:
    return training.root(dataset) / "find" / "boxes.json"


def read_boxes(dataset: str) -> dict[str, list[dict]]:
    """Every done photo's boxes, keyed by photo. Photos that were removed are left out."""
    try:
        saved = json.loads(_boxes_file(dataset).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    folder = labelling.inbox(dataset)
    return {name: boxes for name, boxes in saved.items() if (folder / name).is_file()}


def write_boxes(dataset: str, boxes: dict[str, list[dict]]) -> None:
    path = _boxes_file(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(boxes), encoding="utf-8")
    os.replace(temporary, path)


def photos(dataset: str) -> list[str]:
    folder = labelling.inbox(dataset)
    return sorted(path.name for path in training._images_in(folder) if labelling._IMAGE_ID.match(path.name))


def counts(meta: dict) -> tuple[dict[str, int], int, int]:
    """Boxes of each thing, photos done, and photos still to box."""
    boxes = read_boxes(meta["id"])
    amounts = {name: 0 for name in meta["categories"]}
    for found in boxes.values():
        for box in found:
            if box["category"] in amounts:
                amounts[box["category"]] += 1
    return amounts, len(boxes), len(photos(meta["id"])) - len(boxes)


def _clean(boxes: list[dict], categories: list[str]) -> list[dict]:
    if len(boxes) > MAX_BOXES:
        raise TrainingError(422, f"A photo can have at most {MAX_BOXES} boxes.")
    cleaned = []
    for box in boxes:
        if box.get("category") not in categories:
            raise TrainingError(422, f'"{str(box.get("category"))[:40]}" isn\'t one of the things this dataset finds. Refresh the page.')
        try:
            x, y, w, h = (float(box[key]) for key in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            raise TrainingError(422, "A box is missing its position.") from None
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0 - x), min(max(h, 0.0), 1.0 - y)
        if w < 0.005 or h < 0.005:
            continue  # a slip of the mouse rather than a box
        cleaned.append({"category": box["category"], "x": round(x, 5), "y": round(y, 5), "w": round(w, 5), "h": round(h, 5)})
    return cleaned


def listing(dataset: str) -> dict:
    meta = datasets._read(dataset)
    if not is_find(meta):
        raise TrainingError(422, "Boxes are for “find things in photos” datasets.")
    boxes = read_boxes(dataset)
    names = photos(dataset)
    # Photos still to box come first, so the next one is always at the top.
    names.sort(key=lambda name: (name in boxes, name))
    return {"photos": [{"id": name, "boxes": boxes.get(name)} for name in names], "done": len(boxes), "total": len(names)}


def save_boxes(dataset: str, image_id: str, boxes: list[dict]) -> dict:
    with training._lock:
        meta = datasets._read(dataset)
        if not is_find(meta):
            raise TrainingError(422, "Boxes are for “find things in photos” datasets.")
        labelling.image_file(image_id, dataset)
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        saved = read_boxes(dataset)
        saved[image_id] = _clean(boxes, meta["categories"])
        write_boxes(dataset, saved)
        return {"boxes": saved[image_id]}


def clear_boxes(dataset: str, image_id: str) -> None:
    """Marks a photo as not done again."""
    with training._lock:
        saved = read_boxes(dataset)
        if saved.pop(image_id, None) is not None:
            write_boxes(dataset, saved)


def rename_category(dataset: str, old: str, new: str) -> None:
    saved = read_boxes(dataset)
    for found in saved.values():
        for box in found:
            if box["category"] == old:
                box["category"] = new
    write_boxes(dataset, saved)


def remove_category(dataset: str, name: str) -> None:
    saved = read_boxes(dataset)
    write_boxes(dataset, {photo: [box for box in found if box["category"] != name] for photo, found in saved.items()})


# --------------------------------------------------------------------- the project

_MODEL_PY = '''"""Finds things in photos and draws a box around each one.

A Faster R-CNN detector (Ren et al., 2015) with a MobileNetV3 backbone, pretrained by torchvision on
the COCO photos. Its last layer is replaced to find your own things, then the whole detector is
fine-tuned on your boxed photos.
"""
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor


def best_device():
    # The detector's box steps run faster on the CPU than on a Mac's GPU, so only CUDA is used.
    return "cuda" if torch.cuda.is_available() else "cpu"


def build(categories, pretrained=True):
    """The detector, with one output for each thing plus one for background."""
    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, len(categories) + 1)
        return model
    return fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, num_classes=len(categories) + 1)


def load_image(source):
    """A photo the right way up, as the model reads it."""
    return to_tensor(ImageOps.exif_transpose(Image.open(source)).convert("RGB"))


def boxes_from(output, categories, width, height, sure):
    """The model's boxes as shares of the photo's size, surest first."""
    found = []
    for box, label, score in zip(output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()):
        if score < sure:
            continue
        x1, y1, x2, y2 = box
        found.append({"category": categories[label - 1], "chance": round(score, 4),
                      "x": round(x1 / width, 5), "y": round(y1 / height, 5),
                      "w": round((x2 - x1) / width, 5), "h": round((y2 - y1) / height, 5)})
    return found


def answer(found, categories):
    counts = {name: sum(box["category"] == name for box in found) for name in categories}
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    total = sum(counts.values()) or 1
    return {"boxes": found, "counts": counts,
            "answers": [{"category": name, "chance": amount / total} for name, amount in ranked]}


class Finder:
    def __init__(self, folder="model", device="cpu"):
        folder = Path(folder)
        self.settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        self.categories = self.settings["categories"]
        self.device = device
        self.model = build(self.categories, pretrained=False)
        self.model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        self.model.to(device).eval()

    def find(self, image):
        with torch.no_grad():
            output = self.model([image.to(self.device)])[0]
        found = boxes_from({key: value.cpu() for key, value in output.items()}, self.categories,
                           image.shape[2], image.shape[1], self.settings.get("sure", 0.5))
        return answer(found, self.categories)
'''

_TRAIN_PY = '''"""Teaches the detector to find your things. The photos are in data/images/ and their boxes in
data/boxes.json, as {"photo.jpg": [{"category": "...", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}]},
with positions as shares of the photo's width and height. Then run: python train.py"""
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch

from model import best_device, boxes_from, build, load_image

DATA = Path("data")
HOLD_BACK = 0.2
SURE = 0.5
MATCH = 0.5  # how much a found box must overlap the real one to count as finding it


def fingerprint(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def target(boxes, categories, width, height):
    rows = [[b["x"] * width, b["y"] * height, (b["x"] + b["w"]) * width, (b["y"] + b["h"]) * height] for b in boxes]
    return {"boxes": torch.tensor(rows, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([categories.index(b["category"]) + 1 for b in boxes], dtype=torch.int64)}


def load(names, boxes, categories):
    examples = []
    for name in names:
        image = load_image(DATA / "images" / name)
        examples.append((name, image, target(boxes[name], categories, image.shape[2], image.shape[1])))
    return examples


def flipped(image, goal):
    """The photo mirrored left to right, with its boxes mirrored too."""
    width = image.shape[2]
    rows = goal["boxes"].clone()
    rows[:, [0, 2]] = width - goal["boxes"][:, [2, 0]]
    return image.flip(-1), {"boxes": rows, "labels": goal["labels"]}


def overlap(a, b):
    ax, ay, aw, ah = a["x"], a["y"], a["w"], a["h"]
    bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
    width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    shared = width * height
    union = aw * ah + bw * bh - shared
    return shared / union if union > 0 else 0.0


def match(found, real):
    """Pairs each found box with the real box of the same thing it overlaps most."""
    used, hits = set(), []
    for box in sorted(found, key=lambda item: -item.get("chance", 1)):
        best, best_overlap = None, MATCH
        for index, other in enumerate(real):
            if index in used or other["category"] != box["category"]:
                continue
            amount = overlap(box, other)
            if amount >= best_overlap:
                best, best_overlap = index, amount
        if best is not None:
            used.add(best)
        hits.append(best is not None)
    return hits, len(real) - len(used)


def average_precision(results, categories):
    """The usual detection score, mAP at 50% overlap, from every box above a low sureness."""
    scores = []
    for name in categories:
        real = sum(1 for row in results for box in row["real"] if box["category"] == name)
        if not real:
            continue
        ranked = []
        for row in results:
            found = [box for box in row["all"] if box["category"] == name]
            hits, _ = match(found, [box for box in row["real"] if box["category"] == name])
            ranked += [(box["chance"], hit) for box, hit in zip(sorted(found, key=lambda item: -item["chance"]), hits)]
        ranked.sort(key=lambda item: -item[0])
        found_so_far, precision_sum = 0, 0.0
        for rank, (_, hit) in enumerate(ranked, 1):
            if hit:
                found_so_far += 1
                precision_sum += found_so_far / rank
        scores.append(precision_sum / real)
    return sum(scores) / len(scores) if scores else None


def check(model, examples, categories, device):
    """Each held-back photo's real boxes, and the boxes the model found."""
    model.eval()
    results = []
    with torch.no_grad():
        for name, image, goal in examples:
            output = {key: value.cpu() for key, value in model([image.to(device)])[0].items()}
            height, width = image.shape[1], image.shape[2]
            every = boxes_from(output, categories, width, height, 0.05)
            real = [{"category": categories[label - 1], "x": x1 / width, "y": y1 / height, "w": (x2 - x1) / width, "h": (y2 - y1) / height}
                    for (x1, y1, x2, y2), label in zip(goal["boxes"].tolist(), goal["labels"].tolist())]
            results.append({"file": name, "real": real, "all": every, "found": [box for box in every if box["chance"] >= SURE]})
    for row in results:
        hits, missed = match(row["found"], row["real"])
        row["right"], row["falseAlarms"], row["missed"] = sum(hits), len(hits) - sum(hits), missed
    return results


def check_loss(model, examples, device):
    """The detector's own loss on the held-back photos. Its batch norm is frozen, so this changes nothing."""
    model.train()
    total = 0.0
    with torch.no_grad():
        for _, image, goal in examples:
            losses = model([image.to(device)], [{key: value.to(device) for key, value in goal.items()}])
            total += float(sum(losses.values()))
    return total / max(1, len(examples))


def main():
    saved = json.loads((DATA / "boxes.json").read_text(encoding="utf-8"))
    categories, boxes = saved["categories"], saved["photos"]
    names = sorted(boxes, key=fingerprint)
    if len(names) < 5:
        sys.exit("Box at least 5 photos.")
    held = max(1, int(len(names) * HOLD_BACK))
    check_names, train_names = names[:held], names[held:]
    device = best_device()
    try:
        model = build(categories).to(device)
    except Exception as problem:
        sys.exit(f"The pretrained detector couldn't be loaded. The first run downloads it, about 75 MB. ({type(problem).__name__})")
    train, held_back = load(train_names, boxes, categories), load(check_names, boxes, categories)
    epochs = int(os.environ.get("EPOCHS", "0")) or (24 if len(train) < 60 else 16)
    batch = 4
    steps_per_epoch = math.ceil(len(train) / batch)
    total_steps = epochs * steps_per_epoch
    print(f"Training on {len(train)} samples, checking on {len(held_back)}, using {device}.", flush=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=1e-4)
    warmup = min(steps_per_epoch, 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
    random.seed(0)
    torch.manual_seed(0)
    best, step = None, 0
    partial = Path("model.partial")
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(train)
        random.shuffle(order)
        running = 0.0
        for start in range(0, len(order), batch):
            images, goals = [], []
            for _, image, goal in order[start:start + batch]:
                if random.random() < 0.5:
                    image, goal = flipped(image, goal)
                images.append(image.to(device))
                goals.append({key: value.to(device) for key, value in goal.items()})
            loss = sum(model(images, goals).values())
            if not torch.isfinite(loss):
                sys.exit("The loss became invalid.")
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimiser.step()
            schedule.step()
            running += float(loss) * len(images)
            step += 1
            print(f"Step {step}/{total_steps}", flush=True)
        loss_on_check = check_loss(model, held_back, device)
        results = check(model, held_back, categories, device)
        right = sum(row["right"] for row in results)
        wrong = sum(row["falseAlarms"] + row["missed"] for row in results)
        share = right / (right + wrong) if right + wrong else 1.0
        print(f"Epoch {epoch}/{epochs}  training loss {running / len(train):.4f}  check loss {loss_on_check:.4f}  check accuracy {100 * share:.1f}%", flush=True)
        if best is None or loss_on_check < best:
            # The version with the lowest check loss is kept, as with every model AI Bricks trains.
            best = loss_on_check
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir()
            torch.save(model.state_dict(), partial / "model.pt")
            (partial / "settings.json").write_text(json.dumps({"categories": categories, "sure": SURE}), encoding="utf-8")
    shutil.rmtree("model", ignore_errors=True)
    os.replace(partial, "model")

    model.load_state_dict(torch.load("model/model.pt", map_location="cpu", weights_only=True))
    model.to(device)
    results = check(model, held_back, categories, device)
    per = {name: {"found": 0, "missed": 0, "falseAlarms": 0} for name in categories}
    for row in results:
        for name in categories:
            hits, missed = match([box for box in row["found"] if box["category"] == name], [box for box in row["real"] if box["category"] == name])
            per[name]["found"] += sum(hits)
            per[name]["falseAlarms"] += len(hits) - sum(hits)
            per[name]["missed"] += missed
    metrics = {
        "task": "detect", "photos": len(results), "found": sum(row["right"] for row in results),
        "missed": sum(row["missed"] for row in results), "falseAlarms": sum(row["falseAlarms"] for row in results),
        "allRight": sum(row["missed"] == 0 and row["falseAlarms"] == 0 for row in results),
        "map50": average_precision(results, categories), "perCategory": per, "sure": SURE,
    }
    for row in results:
        del row["all"]
    with open(DATA / "review.json", "w", encoding="utf-8") as file:
        json.dump(results, file)
    with open(DATA / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''"""Finds things in photos. Train first, then run: python predict.py photo.jpg"""
import json
import sys

from model import Finder, load_image


def main():
    paths = [item for item in sys.argv[1:] if item != "--json"]
    if not paths:
        sys.exit("Name at least one photo, for example: python predict.py photo.jpg")
    finder = Finder()
    results = [{"file": path, **finder.find(load_image(path))} for path in paths]
    if "--json" in sys.argv:
        print(json.dumps(results))
        return
    for result in results:
        found = ", ".join(f"{amount} {name}" for name, amount in result["counts"].items() if amount) or "nothing"
        print(f"{result['file']}: {found}")


if __name__ == "__main__":
    main()
'''

_README = """# {name}

Finds things in photos and draws a box around each one, trained in AI Bricks. It's a Faster R-CNN
detector (Ren et al., 2015) with a MobileNetV3 backbone, pretrained by torchvision on the COCO photos
and fine-tuned on your boxed photos. Your photos aren't included in this download.

## Use it

```
pip install -r requirements.txt
python predict.py photo.jpg
```

It shows boxes it's at least 50% sure of. Change `sure` in `model/settings.json` to show more or fewer.

## Train it again

Put photos in `data/images/` and their boxes in `data/boxes.json`, as described at the top of
`train.py`, then run `python train.py`. The first run downloads the pretrained detector, about 75 MB.
"""

LIVE_RUNNER = """
import base64, contextlib, io, json, sys
sys.path.insert(0, '.')
with contextlib.redirect_stdout(sys.stderr):
    from model import Finder, load_image, best_device
    finder = Finder(device=best_device())
print('ready', flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        print(json.dumps(finder.find(load_image(io.BytesIO(base64.b64decode(request['data']))))), flush=True)
    except Exception as problem:
        print(json.dumps({'error': f'{type(problem).__name__}: {problem}'[:300]}), flush=True)
"""


def project_files(name: str) -> dict[str, str]:
    return {
        "model.py": _MODEL_PY, "train.py": _TRAIN_PY, "predict.py": _PREDICT_PY,
        "README.md": _README.format(name=name), "requirements.txt": "torch>=2.2\ntorchvision>=0.17\npillow\n",
    }


def readiness(meta: dict) -> list[str]:
    amounts, done, _ = counts(meta)
    problems = []
    if not meta["categories"]:
        problems.append("Name at least one thing to find.")
    if done < MIN_PHOTOS:
        problems.append(f"Box at least {MIN_PHOTOS} photos. {done} {'is' if done == 1 else 'are'} done.")
    short = [name for name, amount in amounts.items() if amount < MIN_BOXES]
    if short:
        problems.append(f"Draw at least {MIN_BOXES} boxes around each thing. Short: {', '.join(short)}.")
    return problems


def freeze(dataset: str, run: Path) -> None:
    """Copies the done photos and their boxes for one training run."""
    meta = datasets._read(dataset)
    boxes = read_boxes(dataset)
    target = run / "data" / "images"
    target.mkdir(parents=True, exist_ok=True)
    folder = labelling.inbox(dataset)
    for name in boxes:
        versions._link(folder / name, target / name)
    (run / "data" / "boxes.json").write_text(json.dumps({"categories": meta["categories"], "photos": boxes}), encoding="utf-8")


def start(dataset: str) -> training.Job:
    with training._lock:
        meta = datasets._read(dataset)
        if not is_find(meta):
            raise TrainingError(422, "This trainer is for “find things in photos” datasets.")
        problem = datasets._busy(dataset)
        if problem:
            raise TrainingError(409, problem)
        problems = readiness(meta)
        if problems:
            raise TrainingError(422, " ".join(problems))
        amounts, _, _ = counts(meta)
    files = project_files(meta["name"])
    return training.start_run(meta["name"], codegen.slug(meta["name"]), codegen.build_hash(files), files,
                              amounts, dataset, versions.freeze, "detect", 0.0)


def set_sure(job_id: str, sure: float) -> dict:
    """How sure the model must be before it shows a box."""
    if not MIN_SURE <= sure <= MAX_SURE:
        raise TrainingError(422, f"Choose a value between {MIN_SURE} and {MAX_SURE}.")
    job = training.stored(job_id)
    if job.task != "detect" or not job.hasModel:
        raise TrainingError(404, "That model doesn't find things in photos.")
    path = training._folder(job) / "model" / "settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    settings["sure"] = round(sure, 3)
    path.write_text(json.dumps(settings), encoding="utf-8")
    training.close_live(job.id)
    return {"sure": settings["sure"]}


def details(job_id: str) -> dict:
    job = training.stored(job_id)
    if job.task != "detect" or not job.hasModel:
        raise TrainingError(404, "That model doesn't find things in photos.")
    settings = json.loads((training._folder(job) / "model" / "settings.json").read_text(encoding="utf-8"))
    return {"sure": settings.get("sure", 0.5), "categories": settings["categories"], "metrics": job.metrics}
