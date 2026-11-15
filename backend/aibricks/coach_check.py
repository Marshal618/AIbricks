"""The data coach: how much more data would help, and which examples to label next.

Runs on its own: python coach_check.py work_folder. It reads manifest.json there and writes result.json.

How much more data. A quick stand-in model (logistic regression on what a pretrained network sees in
each example, as the label check uses) is trained on growing shares of the labelled data and scored on
examples it never saw, in 5 folds. Its error usually falls along a power law as data grows (Hestness et
al., 2017; Kaplan et al., 2020), so a curve of that shape is fitted and carried forward. Fitting each
fold on its own gives a range. It's an estimate for the stand-in, not for the model you design: a
different model may score higher or lower, but how it improves with more data tends to look alike.

What to label next. The stand-in, trained on everything labelled, guesses each unlabelled example.
Those it's least sure about teach the most (uncertainty sampling, Lewis and Gale, 1994), as long as
they aren't all alike, so each pick must also differ from what's labelled and already picked (as in
core-set selection, Sener and Savarese, 2018).
"""
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health_check import SEED, device, fit, image_embeddings, step, text_embeddings  # noqa: E402

FOLDS = 5
SHARES = [0.1, 0.17, 0.25, 0.4, 0.6, 0.8, 1.0]
REPEATS = 3  # fresh random picks for each share below the whole, to steady the small ones
PROJECT = [1.5, 2, 4]
PICKS = 24
CANDIDATES = 300
EXPONENTS = [0.05 * step for step in range(1, 31)]  # 0.05 to 1.5


def folds_of(targets, classes, rng):
    fold_of = [0] * len(targets)
    for label in range(classes):
        members = [index for index, target in enumerate(targets) if target == label]
        rng.shuffle(members)
        for position, index in enumerate(members):
            fold_of[index] = position % FOLDS
    return fold_of


def subset(members_by_class, share, rng):
    """The same share of each category, at least one of each."""
    chosen = []
    for members in members_by_class:
        if members:
            chosen += rng.sample(members, max(1, round(share * len(members))))
    return chosen


def fit_curve(points):
    """Fits error = floor + scale * examples ** -exponent to (examples, error) points.

    The floor is the error it would level off at; it's kept between 0 and the lowest error seen.
    """
    points = [(n, e) for n, e in points if n > 0]
    if len(points) < 3:
        return None
    lowest = min(e for _, e in points)
    best = None
    for exponent in EXPONENTS:
        xs = [n ** -exponent for n, _ in points]
        for floor_step in range(0, 21):
            floor = lowest * floor_step / 20
            ys = [e - floor for _, e in points]
            scale = max(0.0, sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs))
            misfit = sum((scale * x - y) ** 2 for x, y in zip(xs, ys))
            if best is None or misfit < best[0]:
                best = (misfit, floor, scale, exponent)
    _, floor, scale, exponent = best
    return {"floor": floor, "scale": scale, "exponent": exponent}


def error_at(curve, examples):
    return min(1.0, curve["floor"] + curve["scale"] * examples ** -curve["exponent"])


def learning_curves(features, targets, classes):
    """Scores of the stand-in at each share of the training folds, overall and for each category."""
    rng = random.Random(SEED)
    fold_of = folds_of(targets.tolist(), classes, rng)
    rows = []  # each run: its fold, share, training examples, accuracy, and {category: (its training examples, share right)}
    for fold in range(FOLDS):
        held = [index for index, value in enumerate(fold_of) if value == fold]
        if not held:
            continue
        members_by_class = [[index for index, value in enumerate(fold_of) if value != fold and targets[index] == label] for label in range(classes)]
        held_index = torch.tensor(held)
        for share in SHARES:
            for _ in range(1 if share == 1.0 else REPEATS):
                chosen = subset(members_by_class, share, rng)
                present = sorted({int(targets[index]) for index in chosen})
                if len(present) < 2:
                    continue
                layer = fit(features[chosen], targets[chosen], classes)
                with torch.no_grad():
                    guesses = layer(features[held_index]).argmax(1)
                real = targets[held_index]
                per = {}
                for label in range(classes):
                    mine = real == label
                    if mine.any():
                        per[label] = (sum(int(targets[index]) == label for index in chosen), float((guesses[mine] == label).float().mean()))
                rows.append({"fold": fold, "share": share, "examples": len(chosen), "accuracy": float((guesses == real).float().mean()), "per": per})
        step(2 + fold, FOLDS + 3)
    return rows


def averaged(points):
    """(examples, value) pairs averaged over the same number of examples."""
    grouped = {}
    for n, value in points:
        grouped.setdefault(n, []).append(value)
    return sorted((n, sum(values) / len(values)) for n, values in grouped.items())


def project(rows, labelled, train_share):
    """The fitted curve's accuracy now and with more data, with a range from fitting each fold alone."""
    every = fit_curve(averaged([(row["examples"], 1 - row["accuracy"]) for row in rows]))
    by_fold = [fit_curve(averaged([(row["examples"], 1 - row["accuracy"]) for row in rows if row["fold"] == f])) for f in range(FOLDS)]
    by_fold = [curve for curve in by_fold if curve]
    if every is None:
        return None

    def at(total):
        examples = total * train_share
        middle = 1 - error_at(every, examples)
        spread = [1 - error_at(curve, examples) for curve in by_fold] or [middle]
        return {"examples": round(total), "accuracy": round(middle, 4), "low": round(min(spread + [middle]), 4), "high": round(max(spread + [middle]), 4)}

    seen = []
    for share in SHARES:
        mine = [row for row in rows if row["share"] == share]
        if mine:
            seen.append((sum(row["examples"] for row in mine) / len(mine), sum(row["accuracy"] for row in mine) / len(mine)))
    return {
        "measured": [{"examples": round(n / train_share), "accuracy": round(accuracy, 4)} for n, accuracy in seen],
        "now": at(labelled),
        "more": [at(labelled * times) for times in PROJECT],
        "levelsOffAt": round(1 - every["floor"], 4),
        "fitted": [at(total) for total in sorted({max(2, round(labelled * k / 20)) for k in range(1, 81)})],
    }


def per_category(rows, names, counts, train_share):
    """Each category's share right with all the data, and how much doubling its examples might add."""
    found = []
    for label, name in enumerate(names):
        points = averaged([(row["per"][label][0], 1 - row["per"][label][1]) for row in rows if label in row["per"]])
        full = [row["per"][label][1] for row in rows if row["share"] == 1.0 and label in row["per"]]
        curve = fit_curve(points)
        gain = None
        if curve is not None:
            gain = max(0.0, error_at(curve, counts[label] * train_share) - error_at(curve, counts[label] * 2 * train_share))
        found.append({"category": name, "count": counts[label], "accuracy": round(sum(full) / len(full), 4) if full else None,
                      "gainIfDoubled": None if gain is None else round(gain, 4)})
    return found


def next_to_label(features, targets, classes, pool, names):
    """Unlabelled examples, most useful first, each with the stand-in's guess."""
    if not len(pool):
        return []
    layer = fit(features, targets, classes)
    with torch.no_grad():
        chances = F.softmax(layer(pool), dim=1)
    top = chances.topk(min(2, classes), dim=1)
    margin = top.values[:, 0] - (top.values[:, 1] if classes > 1 else 0)
    unsure = (1 - margin).tolist()
    order = sorted(range(len(pool)), key=lambda index: -unsure[index])
    candidates = order[:CANDIDATES]
    # How far each candidate is from the nearest labelled example, then from each pick as well.
    nearest = (1 - (pool[candidates] @ features.T).max(dim=1).values).clamp(min=0)
    picks = []
    for _ in range(min(PICKS, len(candidates))):
        widest = float(nearest.max())
        if widest <= 1e-6:
            break  # everything left is a copy of something labelled or picked
        scores = [unsure[index] * float(nearest[at]) / widest if index not in picks else -1.0 for at, index in enumerate(candidates)]
        best = max(range(len(candidates)), key=lambda at: scores[at])
        if scores[best] <= 0:
            break
        picks.append(candidates[best])
        nearest = torch.minimum(nearest, (1 - pool[candidates] @ pool[candidates[best]]).clamp(min=0))
    rest = [index for index in order if index not in set(picks)]
    return [{"index": index, "guess": names[int(top.indices[index, 0])], "chance": round(float(top.values[index, 0]), 4),
             "unsure": round(unsure[index], 4), "picked": position < len(picks)}
            for position, index in enumerate(picks + rest)]


def main():
    work = Path(sys.argv[1])
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    kind, items, unlabelled, names = manifest["kind"], manifest["items"], manifest["unlabelled"], manifest["categories"]
    torch.manual_seed(SEED)
    where = device()
    print(f"Coaching on {len(items)} labelled and {len(unlabelled)} unlabelled examples, using {where}.", flush=True)
    step(1, FOLDS + 3)
    if kind == "text":
        everything = text_embeddings([item["text"] for item in items + unlabelled], where)
    else:
        everything = image_embeddings([item["path"] for item in items + unlabelled], where)
    features, pool = everything[:len(items)], everything[len(items):]
    targets = torch.tensor([names.index(item["label"]) for item in items])
    counts = [int((targets == label).sum()) for label in range(len(names))]
    train_share = (FOLDS - 1) / FOLDS
    rows = learning_curves(features, targets, len(names))
    overall = project(rows, len(items), train_share)
    categories = per_category(rows, names, counts, train_share)
    step(FOLDS + 2, FOLDS + 3)
    ranked = next_to_label(features, targets, len(names), pool, names)
    for row in ranked:
        row["id"] = unlabelled[row.pop("index")]["id"]
    step(FOLDS + 3, FOLDS + 3)
    result = {"labelled": len(items), "curve": overall, "categories": categories, "next": ranked}
    (work / "result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
