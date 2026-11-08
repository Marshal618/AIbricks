"""Scores every labelled example with a model that never saw it, to find labels worth a second look.

Runs on its own: python health_check.py work_folder. It reads manifest.json there and writes
result.json. It uses the confident learning rule (Northcutt et al.): an example is flagged when the
model is confidently sure it belongs to a different category than its label.
"""
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FOLDS = 5
SEED = 42
BATCH = 32
MAX_PAIRS = 300
NEAR = {"image": 0.97, "text": 0.97}


def step(number, total):
    print(f"Step {number}/{total}", flush=True)


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def text_embeddings(texts, where):
    from transformers import AutoModel, AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        model = AutoModel.from_pretrained(TEXT_MODEL, use_safetensors=True).to(where).eval()
    except (OSError, ValueError) as problem:
        sys.exit(f"The text model couldn't be loaded. The first check needs an internet connection. ({type(problem).__name__})")
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = tokenizer(texts[start:start + BATCH], padding=True, truncation=True, max_length=128, return_tensors="pt").to(where)
            words = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            rows.append(F.normalize((words * mask).sum(1) / mask.sum(1).clamp(min=1e-9), dim=1).cpu())
    return torch.cat(rows)


def image_embeddings(paths, where):
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18
    try:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    except Exception as problem:  # torchvision raises several kinds of error when offline.
        sys.exit(f"The image model couldn't be downloaded. The first check needs an internet connection. ({type(problem).__name__})")
    model.fc = torch.nn.Identity()
    model = model.to(where).eval()
    prepare = weights.transforms()
    rows = []
    with torch.no_grad():
        for start in range(0, len(paths), BATCH):
            pictures = []
            for path in paths[start:start + BATCH]:
                with Image.open(path) as picture:
                    pictures.append(prepare(picture.convert("RGB")))
            rows.append(F.normalize(model(torch.stack(pictures).to(where)), dim=1).cpu())
    return torch.cat(rows)


PENALTY = 1e-3  # Tested against standardised features, which flagged far more correct labels on real data.


def fit(features, targets, classes):
    """Logistic regression on the normalised embeddings."""
    layer = torch.nn.Linear(features.shape[1], classes)
    optimizer = torch.optim.LBFGS(layer.parameters(), max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(layer(features), targets) + PENALTY * layer.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return layer


def held_out_chances(features, targets, classes):
    """Each example's chances come from a model trained on the other folds, never on itself."""
    rng = random.Random(SEED)
    fold_of = [0] * len(targets)
    for label in range(classes):
        members = [index for index, target in enumerate(targets.tolist()) if target == label]
        rng.shuffle(members)
        for position, index in enumerate(members):
            fold_of[index] = position % FOLDS
    chances = torch.zeros(len(targets), classes)
    for fold in range(FOLDS):
        train = torch.tensor([value != fold for value in fold_of])
        if train.all() or not train.any():
            continue
        predict = fit(features[train], targets[train], classes)
        with torch.no_grad():
            chances[~train] = F.softmax(predict(features[~train]), dim=1)
    return chances


def label_issues(chances, targets, names):
    """Confident learning: thresholds are each category's average confidence on its own examples."""
    thresholds = torch.tensor([
        chances[targets == label, label].mean().item() if (targets == label).any() else 1.0 for label in range(len(names))
    ])
    issues = []
    for index in range(len(targets)):
        confident = (chances[index] >= thresholds).nonzero().flatten().tolist()
        if not confident:
            continue
        best = max(confident, key=lambda label: chances[index, label].item())
        given = int(targets[index])
        if best != given:
            issues.append({
                "index": index, "suggested": names[best], "confidence": round(chances[index, best].item(), 4),
                "givenChance": round(chances[index, given].item(), 4),
            })
    return sorted(issues, key=lambda issue: issue["givenChance"] - issue["confidence"])


def near_duplicates(features, limit):
    pairs = []
    for start in range(0, len(features), 512):
        similar = features[start:start + 512] @ features.T
        for offset, row in enumerate(similar):
            first = start + offset
            for second in (row >= limit).nonzero().flatten().tolist():
                if second > first:
                    pairs.append((first, second, round(row[second].item(), 4)))
    return sorted(pairs, key=lambda pair: -pair[2])


def main():
    work = Path(sys.argv[1])
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    kind, items, names = manifest["kind"], manifest["items"], manifest["categories"]
    torch.manual_seed(SEED)
    where = device()
    print(f"Checking {len(items)} examples, using {where}.", flush=True)
    step(1, 3)
    if kind == "text":
        features = text_embeddings([item["text"] for item in items], where)
    else:
        features = image_embeddings([item["path"] for item in items], where)
    step(2, 3)
    targets = torch.tensor([names.index(item["label"]) for item in items])
    usable = [label for label in range(len(names)) if (targets == label).sum() >= 2]
    chances = held_out_chances(features, targets, len(names)) if len(usable) >= 2 else None
    issues = label_issues(chances, targets, names) if chances is not None else []
    agreement = float((chances.argmax(1) == targets).float().mean()) if chances is not None else None
    step(3, 3)
    pairs = near_duplicates(features, NEAR[kind])
    result = {"agreement": agreement, "issues": []}
    for issue in issues:
        item = items[issue.pop("index")]
        result["issues"].append({"id": item["id"], "given": item["label"], **issue})
    result["duplicates"] = [
        {"first": items[a]["id"], "second": items[b]["id"], "firstLabel": items[a]["label"], "secondLabel": items[b]["label"], "similarity": similarity}
        for a, b, similarity in pairs if items[a]["id"] != items[b]["id"]
    ][:MAX_PAIRS]
    result["duplicateCount"] = len(pairs)
    (work / "result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
