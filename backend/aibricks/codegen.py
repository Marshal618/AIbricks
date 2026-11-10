"""Turns a validated build into a complete, readable PyTorch project."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from . import augment as augmentation
from .brick_sources import SOURCES
from .catalogue import IMAGE, SERIES, TEXT, TRAINING_DIAL
from .graph import Analysis, GraphIn, Shape, Step, parse_names

VAR_NAMES = {
    "conv": "conv", "pool": "pool", "residual": "residual", "upsample": "upsample", "backbone": "backbone",
    "cnn_decoder": "cnn_decoder", "lstm": "lstm", "gru": "gru", "rnn": "rnn", "lstm_decoder": "lstm_decoder",
    "global_pool": "global_pool", "flatten": "flatten", "last_step": "last_step", "dense": "dense",
    "dropout": "dropout", "combine": "combine", "classify": "classify_head", "predict_number": "number_head",
    "forecast": "forecast_head", "detect_anomalies": "rebuild_head", "generate": "generate_head",
    "embedding": "embedding", "positional": "positional", "attention": "attention", "mean_pool": "mean_pool",
    "transformer_encoder": "encoder", "transformer_decoder": "decoder",
    "pretrained_language": "language_model", "pretrained_vision": "vision_model",
}

PRETRAINED = {"backbone", "pretrained_language", "pretrained_vision"}
# Bricks that skip padding when told where it is.
USES_PADDING_MASK = {"attention", "transformer_encoder", "transformer_decoder", "mean_pool", "last_step"}
NEEDS_TORCH = {"combine", "last_step", "mean_pool", "positional", "decoder_only", "decoder_memory"}


def _fill(template: str, **values: object) -> str:
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", str(value))
    return template


def class_name(project: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", project)
    name = "".join(w[:1].upper() + w[1:] for w in words) or "MyModel"
    return f"Model{name}" if name[0].isdigit() else name


def slug(project: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", project.lower())) or "my_model"


def source_key(step: Step) -> str:
    brick = step.brick.id
    if brick == "forecast":
        return f"forecast_{'series' if step.input_shapes[0].kind == SERIES else 'summary'}"
    if brick == "detect_anomalies":
        return f"rebuild_{'image' if step.input_shapes[0].kind == IMAGE else 'series'}"
    if brick == "transformer_decoder":
        return "decoder_memory" if len(step.inputs) == 2 else "decoder_only"
    return brick


def _constructor(step: Step, socket: Shape, target: Shape | None = None) -> str:
    t, p = step.brick.id, step.params
    x = step.input_shapes[0] if step.input_shapes else None
    cls = SOURCES[source_key(step)][0]
    if t == "conv":
        return f"{cls}({x.channels}, {p['filters']}, kernel_size={p['kernel_size']})"
    if t == "embedding":
        return f"{cls}(vocab_size={x.features}, size={p['size']})"
    if t == "positional":
        return f"{cls}({x.features}, max_length={x.steps})"
    if t == "attention":
        return f"{cls}({x.features}, heads={p['heads']})"
    if t == "transformer_encoder":
        return f"{cls}({x.features}, heads={p['heads']}, layers={p['layers']})"
    if t == "transformer_decoder":
        if len(step.inputs) == 2:
            return f"{cls}({x.features}, heads={p['heads']}, layers={p['layers']}, memory_size={step.input_shapes[1].features})"
        return f"{cls}({x.features}, heads={p['heads']}, layers={p['layers']})"
    if t in ("pretrained_language", "pretrained_vision"):
        return f"{cls}(pretrained=use_pretrained_weights, freeze={p['freeze']})"
    if t == "mean_pool":
        return f"{cls}()"
    if t == "generate":
        return f"{cls}({x.features}, vocab_size={(target or socket).features})"
    if t == "pool":
        return f'{cls}(mode="{p["mode"]}")'
    if t == "residual":
        return f"{cls}({x.channels})"
    if t == "upsample":
        return f"{cls}({x.channels}, {p['filters']})"
    if t == "backbone":
        return f"{cls}(pretrained=use_pretrained_weights, freeze={p['freeze']})"
    if t == "cnn_decoder":
        return f"{cls}({x.size}, {p['filters']}, height={step.output.height}, width={step.output.width})"
    if t in ("lstm", "gru", "rnn"):
        return f"{cls}({x.features}, {p['hidden']}, layers={p['layers']}, bidirectional={p['bidirectional']})"
    if t == "lstm_decoder":
        return f"{cls}({x.size}, {p['hidden']}, steps={step.output.steps})"
    if t in ("global_pool", "flatten", "last_step", "combine"):
        return f"{cls}()"
    if t == "dense":
        return f"{cls}({x.size}, {p['size']})"
    if t == "dropout":
        return f"{cls}({p['percent'] / 100})"
    if t == "classify":
        return f"{cls}({x.size}, num_classes=len(CLASS_NAMES))"
    if t == "predict_number":
        return f"{cls}({x.size})"
    if t == "forecast":
        if x.kind == SERIES:
            return f"{cls}({x.features}, num_features={socket.features})"
        return f"{cls}({x.size}, horizon={p['horizon']}, num_features={socket.features})"
    if t == "detect_anomalies":
        return f"{cls}({x.channels}, {socket.channels})" if x.kind == IMAGE else f"{cls}({x.features}, {socket.features})"
    raise ValueError(t)  # pragma: no cover


def _model_py(graph: GraphIn, analysis: Analysis) -> str:
    socket, head, target = analysis.socket, analysis.head, analysis.target
    body = [s for s in analysis.steps if s.brick.group != "socket"]
    used = sorted({SOURCES[source_key(s)][0] for s in body})
    counts = Counter(s.brick.id for s in body)
    seen: Counter = Counter()
    names: dict[str, str] = {}
    for step in body:
        seen[step.brick.id] += 1
        base = VAR_NAMES[step.brick.id]
        names[step.id] = f"{base}_{seen[step.brick.id]}" if counts[step.brick.id] > 1 else base

    linear = all(len(s.inputs) == 1 for s in body) and len({s.inputs[0] for s in body}) == len(body)
    variable = {socket.id: "x"}
    if target:
        variable[target.id] = "target"
    # Text carries padding, so bricks that care are told where it is.
    mask_of: dict[str, str | None] = {socket.id: "padding_mask" if socket.output.kind == TEXT else None}
    if target:
        mask_of[target.id] = "target_padding_mask"
    has_backbone = any(s.brick.id in PRETRAINED for s in body)
    cls = class_name(graph.projectName)

    lines = ['"""The model you designed in AI Bricks, written out as plain PyTorch."""', "from torch import nn", ""]
    lines.append("from bricks import " + ", ".join(used))
    if head.brick.id == "classify":
        lines += ["", f"CLASS_NAMES = {parse_names(head.params['classes'])!r}"]
    lines += ["", "", f"class {cls}(nn.Module):"]
    lines.append("    def __init__(self, use_pretrained_weights=True):" if has_backbone else "    def __init__(self):")
    lines.append("        super().__init__()")
    for step in body:
        lines.append(f"        self.{names[step.id]} = {_constructor(step, socket.output, target.output if target else None)}")
    signature = "self, x, target" if target else "self, x"
    lines += ["", f"    def forward({signature}):", f"        # x starts as {socket.output.code()}"]
    if target:
        lines.append(f"        # target starts as {target.output.code()}")
    if socket.output.kind == TEXT:
        lines.append("        padding_mask = x == 0  # token 0 is padding")
    if target:
        lines.append("        target_padding_mask = target == 0")
    for step in body:
        out = "x" if linear else f"{names[step.id]}_out"
        variable[step.id] = out
        call_args = [variable[i] for i in step.inputs]
        masks = [mask_of.get(i) for i in step.inputs]
        if step.brick.id in USES_PADDING_MASK and masks[0]:
            call_args.append(f"padding_mask={masks[0]}")
        if step.brick.id == "transformer_decoder" and len(masks) == 2 and masks[1]:
            call_args.append(f"memory_padding_mask={masks[1]}")
        keeps_steps = step.output is not None and step.output.kind == SERIES and step.output.steps == step.input_shapes[0].steps
        mask_of[step.id] = masks[0] if keeps_steps else None
        call = f"{out} = self.{names[step.id]}({', '.join(call_args)})"
        if step.output is not None:
            lines.append(f"        {call}  # {step.output.code()}")
        else:
            lines.append(f"        {call}")
    lines.append(f"        return {variable[head.id]}")
    lines += ["", ""]
    if has_backbone:
        lines += ["def build_model(use_pretrained_weights=True):", f"    return {cls}(use_pretrained_weights)", ""]
    else:
        lines += ["def build_model(use_pretrained_weights=True):", f"    return {cls}()  # this build has no pretrained parts", ""]
    return "\n".join(lines)


def _bricks_py(analysis: Analysis) -> str:
    keys = []
    for step in analysis.steps:
        if step.brick.group == "socket":
            continue
        key = source_key(step)
        if key not in keys:
            keys.append(key)
    header = "from torch import nn\n"
    if NEEDS_TORCH & set(keys):
        header = "import torch\n" + header
    stdlib = []
    if "positional" in keys:
        stdlib.append("import math")
    if {"backbone", "pretrained_vision"} & set(keys):
        stdlib.append("from urllib.error import URLError")
    if stdlib:
        header = "\n".join(stdlib) + "\n\n" + header
    parts = ['"""One small, readable PyTorch module for each brick in your build."""\n' + header]
    parts += [SOURCES[k][1] for k in keys]
    return "\n\n".join(parts)


# --------------------------------------------------------------------- data.py

_BANNER = """# ======================================================================
# PUT YOUR DATA HERE
@@LINES@@
# ======================================================================
"""

_IMAGE_COMMON = '''
IMAGE_SIZE = @@SIZE@@
CHANNELS = @@CHANNELS@@
AUGMENT = @@AUGMENT@@  # vary the training photos, never the held-back ones
VARY = @@VARY@@
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@@VARIED_STEPS@@

def _image_transform(augment=False):
    steps = varied_steps(IMAGE_SIZE, VARY, transforms) if augment else [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if CHANNELS == 1:
        steps.append(transforms.Grayscale(1))
    steps.append(transforms.ToTensor())@@NORMALISE@@
    return transforms.Compose(steps)


def _load_image(path, transform):
    return transform(Image.open(path).convert("RGB"))


def _hash(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _split(dataset):
    """Holds back 20% of the images to check the model on data it never trained on.

    The images held back are picked by a fingerprint of their file name, not at random, so models
    trained on the same images are checked on the same ones and can be compared fairly.
    """
    if len(dataset) < 5:
        raise ValueError(f"Found only {len(dataset)} images. Add at least 5 before training.")
    val_size = max(1, len(dataset) // 5)
    paths = [sample[0] for sample in dataset.samples] if hasattr(dataset, "samples") else dataset.paths
    order = sorted(range(len(paths)), key=lambda index: _hash(Path(paths[index]).name))
    return Subset(dataset, sorted(order[val_size:])), Subset(dataset, sorted(order[:val_size]))
'''

_IMAGE_CLASSIFY = '''"""Loads your images. Edit the marked section, then run: python train.py"""
import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from model import CLASS_NAMES

@@BANNER@@@@COMMON@@

class ImageClassDataset(Dataset):
    """Reads <DATA_FOLDER>/<category>/<image>."""

    normalisation = None

    def __init__(self):
        self.transform = _image_transform()
        self.varied = _image_transform(augment=True)
        self.augmented = set()  # the training photos, when AUGMENT is on
        self.samples = []
        root = Path(DATA_FOLDER)
        if not root.is_dir():
            raise FileNotFoundError(f"Data folder not found: {root.resolve()}")
        for label, name in enumerate(CLASS_NAMES):
            folder = root / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected a folder for category {name!r}: {folder.resolve()}")
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in EXTENSIONS:
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return _load_image(path, self.varied if index in self.augmented else self.transform), label


def load_datasets():
    dataset = ImageClassDataset()
    train, val = _split(dataset)
    if AUGMENT:
        dataset.augmented = set(train.indices)
    train.normalisation = None
    return train, val
'''

_IMAGE_NUMBER = '''"""Loads your images and their numbers. Edit the marked section, then run: python train.py"""
import csv
import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

@@BANNER@@@@COMMON@@

class ImageNumberDataset(Dataset):
    """Reads LABELS_FILE, a CSV with the columns: filename,value"""

    def __init__(self):
        self.transform = _image_transform()
        self.samples = []
        with open(LABELS_FILE, newline="") as file:
            for row in csv.DictReader(file):
                self.samples.append((Path(DATA_FOLDER) / row["filename"], float(row["value"])))
        values = torch.tensor([value for _, value in self.samples])
        if len(values) < 2:
            raise ValueError("Add at least 2 rows to the labels file.")
        # Scale the target to around 0. Convert predictions back with: prediction * std + mean
        self.mean = values.mean().item()
        self.std = values.std().item() or 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, value = self.samples[index]
        target = torch.tensor([(value - self.mean) / self.std], dtype=torch.float32)
        return _load_image(path, self.transform), target


def load_datasets():
    dataset = ImageNumberDataset()
    train, val = _split(dataset)
    train.normalisation = {"target_mean": dataset.mean, "target_std": dataset.std}
    return train, val
'''

_IMAGE_REBUILD = '''"""Loads your normal images. Edit the marked section, then run: python train.py"""
import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

@@BANNER@@@@COMMON@@

class NormalImageDataset(Dataset):
    """Reads every image in DATA_FOLDER. The target is the image itself."""

    def __init__(self):
        self.transform = _image_transform()
        root = Path(DATA_FOLDER)
        if not root.is_dir():
            raise FileNotFoundError(f"Data folder not found: {root.resolve()}")
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = _load_image(self.paths[index], self.transform)
        return image, image


def load_datasets():
    train, val = _split(NormalImageDataset())
    train.normalisation = None
    return train, val
'''

_SERIES_COMMON = '''"""Loads your time series. Edit the marked section, then run: python train.py"""
import csv

import torch
from torch.utils.data import Dataset
@@EXTRA_IMPORT@@
@@BANNER@@
COLUMNS = @@COLUMNS@@
SEQUENCE_LENGTH = @@LENGTH@@
HORIZON = @@HORIZON@@  # future rows each sample predicts
TRAIN_FRACTION = 0.8  # the earliest rows train; the latest rows check the model


def _read_table():
    """Reads the CSV into a dictionary of column name to list of text values."""
    with open(DATA_FILE, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{DATA_FILE} has no rows.")
    needed = COLUMNS + EXTRA_COLUMNS
    missing = [name for name in needed if name not in rows[0]]
    if missing:
        raise ValueError(f"{DATA_FILE} is missing the columns {missing}. It has {list(rows[0])}.")
    return {name: [row[name] for row in rows] for name in needed}


def _numbers(table, name):
    try:
        return [float(value) for value in table[name]]
    except ValueError as error:
        raise ValueError(f"Column {name!r} has a value that isn't a number: {error}") from error


def _time_split(total_rows):
    """Splits by time, so the check uses later rows than training."""
    cut = int(total_rows * TRAIN_FRACTION)
    needed = SEQUENCE_LENGTH + HORIZON
    if cut < needed or total_rows - cut < needed:
        minimum = int(needed / min(TRAIN_FRACTION, 1 - TRAIN_FRACTION)) + 1
        raise ValueError(f"Found {total_rows} rows, but this model needs at least {minimum}.")
    return cut


class WindowDataset(Dataset):
    """Slides a window over the rows. Each window is one sample."""

    def __init__(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs) - SEQUENCE_LENGTH - HORIZON + 1

    def __getitem__(self, index):
        x = self.inputs[index : index + SEQUENCE_LENGTH]
        return x, @@TARGET@@

'''

_SERIES_LOAD_PLAIN = '''
def load_datasets():
    table = _read_table()
    rows = torch.tensor(list(zip(*(_numbers(table, name) for name in COLUMNS))), dtype=torch.float32)
    cut = _time_split(len(rows))
    # Scale each column to around 0 using the training rows only.
    mean, std = rows[:cut].mean(dim=0), rows[:cut].std(dim=0).clamp_min(1e-6)
    rows = (rows - mean) / std
    train = WindowDataset(rows[:cut])
    train.normalisation = {"mean": mean.tolist(), "std": std.tolist()}
    return train, WindowDataset(rows[cut:])
'''

_SERIES_LOAD_LABEL = '''
def load_datasets():
    table = _read_table()
    rows = torch.tensor(list(zip(*(_numbers(table, name) for name in COLUMNS))), dtype=torch.float32)
    unknown = sorted(set(table[LABEL_COLUMN]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Column {LABEL_COLUMN!r} has categories that aren't in the model: {unknown[:5]}")
    labels = torch.tensor([CLASS_NAMES.index(value) for value in table[LABEL_COLUMN]])
    cut = _time_split(len(rows))
    mean, std = rows[:cut].mean(dim=0), rows[:cut].std(dim=0).clamp_min(1e-6)
    rows = (rows - mean) / std
    train = WindowDataset(rows[:cut], labels[:cut])
    train.normalisation = {"mean": mean.tolist(), "std": std.tolist()}
    return train, WindowDataset(rows[cut:], labels[cut:])
'''

_SERIES_LOAD_NUMBER = '''
def load_datasets():
    table = _read_table()
    rows = torch.tensor(list(zip(*(_numbers(table, name) for name in COLUMNS))), dtype=torch.float32)
    values = torch.tensor(_numbers(table, TARGET_COLUMN), dtype=torch.float32).unsqueeze(1)
    cut = _time_split(len(rows))
    mean, std = rows[:cut].mean(dim=0), rows[:cut].std(dim=0).clamp_min(1e-6)
    target_mean, target_std = values[:cut].mean().item(), values[:cut].std().item() or 1.0
    rows = (rows - mean) / std
    values = (values - target_mean) / target_std  # convert predictions back with: prediction * std + mean
    train = WindowDataset(rows[:cut], values[:cut])
    train.normalisation = {
        "mean": mean.tolist(), "std": std.tolist(),
        "target_mean": target_mean, "target_std": target_std,
    }
    return train, WindowDataset(rows[cut:], values[cut:])
'''


def _data_py(analysis: Analysis) -> str:
    socket, head = analysis.socket, analysis.head
    task = head.brick.id
    if socket.output.kind == TEXT:
        return _text_data_py(analysis)
    if socket.output.kind == IMAGE:
        pretrained = any(s.brick.id in PRETRAINED for s in analysis.steps)
        normalise = (
            "\n    # The pretrained model expects images scaled this way.\n"
            "    steps.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))"
        ) if pretrained else ""
        augment = bool(socket.params.get("augment", False)) and task == "classify"
        common = _fill(_IMAGE_COMMON, SIZE=socket.output.height, CHANNELS=socket.output.channels, NORMALISE=normalise, AUGMENT=augment,
                       VARY=repr(augmentation.settings_from(socket.params)), VARIED_STEPS=augmentation.SOURCE)
        if task == "classify":
            names = parse_names(head.params["classes"])
            banner = [
                "# One folder per category, for example:",
                *[f"#   data/images/{n}/photo_001.jpg" for n in names[:3]],
                f"# Images of any size work. They are resized to {socket.output.height}×{socket.output.height}.",
            ]
            template, config = _IMAGE_CLASSIFY, 'DATA_FOLDER = "data/images"'
        elif task == "predict_number":
            banner = [
                "# A folder of images, plus a CSV file listing each image's number:",
                "#   filename,value",
                "#   photo_001.jpg,3.5",
            ]
            template, config = _IMAGE_NUMBER, 'DATA_FOLDER = "data/images"\nLABELS_FILE = "data/labels.csv"'
        else:
            banner = ["# A folder of NORMAL images only. The model learns what normal looks like."]
            template, config = _IMAGE_REBUILD, 'DATA_FOLDER = "data/images"'
        full_banner = _fill(_BANNER, LINES="\n".join(banner)) + config + "\n"
        return _fill(template, BANNER=full_banner, COMMON=common)

    columns = parse_names(socket.params["columns"])
    horizon = 0
    extra_import = ""
    extra = "EXTRA_COLUMNS = []"
    example_extra = ""
    banner = ["# A CSV file with one row per time step, oldest first, and these columns:"]
    if task == "forecast":
        horizon = head.params["horizon"]
        target = "self.inputs[index + SEQUENCE_LENGTH : index + SEQUENCE_LENGTH + HORIZON]"
        load = _SERIES_LOAD_PLAIN
    elif task == "detect_anomalies":
        target = "x"
        load = _SERIES_LOAD_PLAIN
        banner.append("# Use NORMAL readings only. The model learns what normal looks like.")
    elif task == "classify":
        target = "self.targets[index + SEQUENCE_LENGTH - 1]"
        load = _SERIES_LOAD_LABEL
        extra_import = "\nfrom model import CLASS_NAMES\n"
        extra = 'LABEL_COLUMN = "label"  # category at each row\nEXTRA_COLUMNS = [LABEL_COLUMN]'
        example_extra = ",label"
    else:
        target = "self.targets[index + SEQUENCE_LENGTH - 1]"
        load = _SERIES_LOAD_NUMBER
        extra = 'TARGET_COLUMN = "target"  # number to predict at each row\nEXTRA_COLUMNS = [TARGET_COLUMN]'
        example_extra = ",target"
    banner.append("#   " + ",".join(columns) + example_extra)
    config = _fill(_BANNER, LINES="\n".join(banner)) + 'DATA_FILE = "data/series.csv"\n' + extra + "\n"
    return _fill(
        _SERIES_COMMON,
        EXTRA_IMPORT=extra_import,
        BANNER=config,
        COLUMNS=repr(columns),
        LENGTH=socket.output.steps,
        HORIZON=horizon,
        TARGET=target,
    ) + load.lstrip("\n")


# --------------------------------------------------------------- sample_data.py

_SAMPLE_HEADER = '''"""Generated sample data, used only by the test run. It is not a substitute for your data."""
import math

import torch

BATCH_SIZE = 8
'''

_SAMPLES = {
    ("image", "classify"): '''
NUM_CLASSES = @@CLASSES@@
SIZE, CHANNELS = @@SIZE@@, @@CHANNELS@@


def make_sample_batch():
    """Each category is a bright square in its own position."""
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(BATCH_SIZE, CHANNELS, SIZE, SIZE, generator=generator) * 0.1
    labels = torch.arange(BATCH_SIZE) % NUM_CLASSES
    columns = math.ceil(math.sqrt(NUM_CLASSES))
    cell = max(SIZE // columns, 1)
    for index, label in enumerate(labels.tolist()):
        row, column = divmod(label, columns)
        images[index, :, row * cell : (row + 1) * cell, column * cell : (column + 1) * cell] = 1.0
    return images, labels
''',
    ("image", "predict_number"): '''
SIZE, CHANNELS = @@SIZE@@, @@CHANNELS@@


def make_sample_batch():
    """The number to predict is each image's brightness."""
    generator = torch.Generator().manual_seed(0)
    values = torch.linspace(-1, 1, BATCH_SIZE).unsqueeze(1)
    noise = torch.rand(BATCH_SIZE, CHANNELS, SIZE, SIZE, generator=generator) * 0.1
    return (values.view(-1, 1, 1, 1) + 1) / 2 + noise, values
''',
    ("image", "detect_anomalies"): '''
SIZE, CHANNELS = @@SIZE@@, @@CHANNELS@@


def make_sample_batch():
    """Smooth gradients, which are easy to learn and rebuild."""
    ramp = torch.linspace(0, 1, SIZE)
    images = []
    for index in range(BATCH_SIZE):
        strength = (index + 1) / BATCH_SIZE
        images.append((ramp.view(1, -1) * strength).expand(SIZE, SIZE).expand(CHANNELS, SIZE, SIZE))
    images = torch.stack(images)
    return images, images
''',
    ("series", "classify"): '''
NUM_CLASSES = @@CLASSES@@
STEPS, FEATURES = @@STEPS@@, @@FEATURES@@


def make_sample_batch():
    """Each category is a wave with its own frequency."""
    labels = torch.arange(BATCH_SIZE) % NUM_CLASSES
    time = torch.linspace(0, 2 * math.pi, STEPS)
    waves = torch.stack([torch.sin((label + 1) * time) for label in labels.tolist()])
    return waves.unsqueeze(2).repeat(1, 1, FEATURES), labels
''',
    ("series", "predict_number"): '''
STEPS, FEATURES = @@STEPS@@, @@FEATURES@@


def make_sample_batch():
    """The number to predict is each wave's height."""
    values = torch.linspace(0.2, 1, BATCH_SIZE).unsqueeze(1)
    time = torch.linspace(0, 2 * math.pi, STEPS)
    waves = values * torch.sin(time)
    return waves.unsqueeze(2).repeat(1, 1, FEATURES), values * 2 - 1.2
''',
    ("series", "forecast"): '''
STEPS, FEATURES, HORIZON = @@STEPS@@, @@FEATURES@@, @@HORIZON@@


def make_sample_batch():
    """Waves that start at different points. The model continues each wave."""
    start = torch.linspace(0, 2 * math.pi, BATCH_SIZE).unsqueeze(1)
    time = torch.arange(STEPS + HORIZON, dtype=torch.float32).unsqueeze(0) * (2 * math.pi / 24)
    waves = torch.sin(start + time).unsqueeze(2).repeat(1, 1, FEATURES)
    return waves[:, :STEPS], waves[:, STEPS:]
''',
    ("series", "detect_anomalies"): '''
STEPS, FEATURES = @@STEPS@@, @@FEATURES@@


def make_sample_batch():
    """Regular waves that start at different points."""
    start = torch.linspace(0, 2 * math.pi, BATCH_SIZE).unsqueeze(1)
    time = torch.linspace(0, 2 * math.pi, STEPS).unsqueeze(0)
    waves = torch.sin(start + time).unsqueeze(2).repeat(1, 1, FEATURES)
    return waves, waves
''',
}


def _sample_py(analysis: Analysis) -> str:
    socket, head = analysis.socket, analysis.head
    s = socket.output
    if s.kind == TEXT:
        return _text_sample_py(analysis)
    classes = len(parse_names(head.params["classes"])) if head.brick.id == "classify" else 0
    return _SAMPLE_HEADER + _fill(
        _SAMPLES[(s.kind, head.brick.id)],
        CLASSES=classes,
        SIZE=s.height,
        CHANNELS=s.channels,
        STEPS=s.steps,
        FEATURES=s.features,
        HORIZON=head.params.get("horizon", 0),
    )


# ------------------------------------------------------------- text data and samples

_TOKENIZER = '''
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def split_words(text):
    return re.findall(r"\\w+|[^\\w\\s]", text.lower())


class SimpleTokenizer:
    """Gives each common word a number. The word list is saved with the model."""

    def __init__(self, words):
        self.words = words
        self.index = {word: number for number, word in enumerate(words)}

    @classmethod
    def build(cls, word_lists, vocab_size):
        counts = Counter(word for words in word_lists for word in words)
        common = [word for word, _ in counts.most_common(vocab_size - len(SPECIAL_TOKENS))]
        return cls(SPECIAL_TOKENS + common)

    def encode(self, text):
        return [self.index.get(word, UNK) for word in split_words(text)] or [UNK]

    def decode(self, numbers):
        return " ".join(self.words[n] for n in numbers if n >= len(SPECIAL_TOKENS))


def pad(numbers, length):
    numbers = numbers[:length]
    return numbers + [PAD] * (length - len(numbers))
'''

_HF_TOKENIZER = '''

class PretrainedTokenizer:
    """The tokenizer DistilBERT was trained with."""

    words = None

    def __init__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def encode(self, text):
        return self.tokenizer(text, truncation=True, max_length=MAX_LENGTH)["input_ids"]
'''

_TEXT_ROWS = '''"""Loads your text. Edit the marked section, then run: python train.py"""
import csv
import random
import re
from collections import Counter

import torch
from torch.utils.data import Dataset
@@EXTRA_IMPORT@@
@@BANNER@@
MAX_LENGTH = @@LENGTH@@
VOCAB_SIZE = @@VOCAB@@
@@TOKENIZER@@@@HF@@

def _read_rows():
    with open(DATA_FILE, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 5:
        raise ValueError(f"{DATA_FILE} has {len(rows)} rows. Add at least 5 before training.")
    missing = [name for name in (TEXT_COLUMN, ANSWER_COLUMN) if name not in rows[0]]
    if missing:
        raise ValueError(f"{DATA_FILE} is missing the columns {missing}. It has {list(rows[0])}.")
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, answer):
        self.tokens = torch.tensor([pad(tokenizer.encode(row[TEXT_COLUMN]), MAX_LENGTH) for row in rows])
        self.answers = [answer(row[ANSWER_COLUMN]) for row in rows]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index], self.answers[index]


def load_datasets():
    rows = _read_rows()
    random.Random(0).shuffle(rows)
    cut = len(rows) - max(1, len(rows) // 5)  # hold back 20% to check the model
    train_rows, val_rows = rows[:cut], rows[cut:]
    tokenizer = @@MAKE_TOKENIZER@@
@@LOAD_BODY@@
    train.vocabulary = tokenizer.words
    return train, val
'''

_TEXT_LOAD_CLASSIFY = '''
    def answer(value):
        if value not in CLASS_NAMES:
            raise ValueError(f"Column {ANSWER_COLUMN!r} has a category that isn't in the model: {value!r}")
        return CLASS_NAMES.index(value)

    train, val = TextDataset(train_rows, tokenizer, answer), TextDataset(val_rows, tokenizer, answer)'''

_TEXT_LOAD_NUMBER = '''
    values = torch.tensor([float(row[ANSWER_COLUMN]) for row in train_rows])
    mean, std = values.mean().item(), values.std().item() or 1.0

    def answer(value):  # convert predictions back with: prediction * std + mean
        return torch.tensor([(float(value) - mean) / std], dtype=torch.float32)

    train, val = TextDataset(train_rows, tokenizer, answer), TextDataset(val_rows, tokenizer, answer)
    train.normalisation = {"target_mean": mean, "target_std": std}'''

_TEXT_CORPUS = '''"""Loads your text. Edit the marked section, then run: python train.py"""
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

@@BANNER@@
MAX_LENGTH = @@LENGTH@@
VOCAB_SIZE = @@VOCAB@@
TRAIN_FRACTION = 0.9  # the rest of the text checks the model
@@TOKENIZER@@

class NextTokenDataset(Dataset):
    """Each sample is a window of text. The answer is the same window moved one token later."""

    def __init__(self, numbers):
        self.numbers = torch.tensor(numbers)

    def __len__(self):
        return max(0, (len(self.numbers) - MAX_LENGTH - 1) // (MAX_LENGTH // 2 or 1) + 1)

    def __getitem__(self, index):
        start = index * (MAX_LENGTH // 2 or 1)
        return self.numbers[start : start + MAX_LENGTH], self.numbers[start + 1 : start + MAX_LENGTH + 1]


def load_datasets():
    with open(DATA_FILE, encoding="utf-8") as file:
        words = split_words(file.read())
    cut = int(len(words) * TRAIN_FRACTION)
    if cut <= MAX_LENGTH or len(words) - cut <= MAX_LENGTH:
        minimum = (MAX_LENGTH + 1) * 10 + 1
        raise ValueError(f"Found {len(words)} tokens, but this model needs at least {minimum}.")
    tokenizer = SimpleTokenizer.build([words[:cut]], VOCAB_SIZE)
    numbers = [tokenizer.index.get(word, UNK) for word in words]
    train = NextTokenDataset(numbers[:cut])
    train.vocabulary = tokenizer.words
    return train, NextTokenDataset(numbers[cut:])
'''

_TEXT_PAIRS = '''"""Loads your text pairs. Edit the marked section, then run: python train.py"""
import csv
import random
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

@@BANNER@@
SOURCE_LENGTH, SOURCE_VOCAB_SIZE = @@S_LENGTH@@, @@S_VOCAB@@
TARGET_LENGTH, TARGET_VOCAB_SIZE = @@T_LENGTH@@, @@T_VOCAB@@
@@TOKENIZER@@

class PairDataset(Dataset):
    """While training, the decoder reads the answer so far and predicts its next token."""

    def __init__(self, rows, source_tokenizer, target_tokenizer):
        self.source = torch.tensor([pad(source_tokenizer.encode(row[SOURCE_COLUMN]), SOURCE_LENGTH) for row in rows])
        targets = []
        for row in rows:
            numbers = [BOS] + target_tokenizer.encode(row[TARGET_COLUMN])[: TARGET_LENGTH - 1] + [EOS]
            targets.append(pad(numbers, TARGET_LENGTH + 1))
        self.target = torch.tensor(targets)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return (self.source[index], self.target[index, :-1]), self.target[index, 1:]


def load_datasets():
    with open(DATA_FILE, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 5:
        raise ValueError(f"{DATA_FILE} has {len(rows)} rows. Add at least 5 before training.")
    missing = [name for name in (SOURCE_COLUMN, TARGET_COLUMN) if name not in rows[0]]
    if missing:
        raise ValueError(f"{DATA_FILE} is missing the columns {missing}. It has {list(rows[0])}.")
    random.Random(0).shuffle(rows)
    cut = len(rows) - max(1, len(rows) // 5)
    train_rows = rows[:cut]
    source_tokenizer = SimpleTokenizer.build([split_words(row[SOURCE_COLUMN]) for row in train_rows], SOURCE_VOCAB_SIZE)
    target_tokenizer = SimpleTokenizer.build([split_words(row[TARGET_COLUMN]) for row in train_rows], TARGET_VOCAB_SIZE)
    train = PairDataset(train_rows, source_tokenizer, target_tokenizer)
    train.vocabulary = {"source": source_tokenizer.words, "target": target_tokenizer.words}
    return train, PairDataset(rows[cut:], source_tokenizer, target_tokenizer)
'''

_GENERATE_NEXT = '''"""Writes text with your trained model. Run: python generate.py "your opening words"
"""
import sys

import torch

from data import MAX_LENGTH, UNK, SimpleTokenizer, pad
from model import build_model

NEW_TOKENS = 40
TEMPERATURE = 0.8  # lower is safer, higher is more varied


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    checkpoint = torch.load("model.pt", weights_only=True)
    tokenizer = SimpleTokenizer(checkpoint["vocabulary"])
    model = build_model()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    numbers = tokenizer.encode(prompt) if prompt else [UNK]
    with torch.no_grad():
        for _ in range(NEW_TOKENS):
            window = numbers[-MAX_LENGTH:]
            scores = model(torch.tensor([pad(window, MAX_LENGTH)]))[0, len(window) - 1]
            scores[:4] = float("-inf")  # never write special tokens
            scores[len(tokenizer.words) :] = float("-inf")  # stay within the words seen in training
            numbers.append(torch.multinomial((scores / TEMPERATURE).softmax(dim=0), 1).item())
    print(tokenizer.decode(numbers))


if __name__ == "__main__":
    main()
'''

_GENERATE_PAIR = '''"""Translates text with your trained model. Run: python generate.py "text to translate"
"""
import sys

import torch

from data import BOS, EOS, SOURCE_LENGTH, TARGET_LENGTH, SimpleTokenizer, pad
from model import build_model


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "hello"
    checkpoint = torch.load("model.pt", weights_only=True)
    source_tokenizer = SimpleTokenizer(checkpoint["vocabulary"]["source"])
    target_tokenizer = SimpleTokenizer(checkpoint["vocabulary"]["target"])
    model = build_model()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    source = torch.tensor([pad(source_tokenizer.encode(text), SOURCE_LENGTH)])
    answer = [BOS]
    with torch.no_grad():
        while len(answer) <= TARGET_LENGTH:
            target = torch.tensor([pad(answer, TARGET_LENGTH)])
            scores = model(source, target)[0, len(answer) - 1]
            scores[len(target_tokenizer.words) :] = float("-inf")  # stay within the words seen in training
            next_token = scores.argmax().item()
            if next_token == EOS:
                break
            answer.append(next_token)
    print(target_tokenizer.decode(answer) or "(The model stopped without writing anything. It may need more data or training.)")


if __name__ == "__main__":
    main()
'''


_PREDICT = '''"""Asks the trained model about one image. Train first, then run: python predict.py photo.jpg"""
import argparse
import json
from pathlib import Path

import torch

from data import EXTENSIONS, _image_transform, _load_image
from model import CLASS_NAMES, build_model

MODEL_FILE = "model.pt"


def load_trained_model():
    try:
        checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise SystemExit(f"{MODEL_FILE} wasn't found. Train the model first with: python train.py") from None
    model = build_model(use_pretrained_weights=False)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def answer(model, image_file):
    """Returns (category, chance) pairs, most likely first. image_file is a path or an open file."""
    image = _load_image(image_file, _image_transform()).unsqueeze(0)
    with torch.no_grad():
        chances = model(image).softmax(dim=1)[0].tolist()
    return sorted(zip(CLASS_NAMES, chances), key=lambda pair: pair[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*")
    parser.add_argument("--folder", help="also ask about every image in this folder")
    parser.add_argument("--json", action="store_true", help="print the answers as JSON")
    arguments = parser.parse_args()
    paths = list(arguments.images)
    if arguments.folder:
        paths += sorted(str(path) for path in Path(arguments.folder).iterdir() if path.suffix.lower() in EXTENSIONS)
    if not paths:
        parser.error("name at least one image, or use --folder")
    model = load_trained_model()
    results = [(path, answer(model, path)) for path in paths]
    if arguments.json:
        print(json.dumps([
            {"file": path, "answers": [{"category": name, "chance": chance} for name, chance in answers]}
            for path, answers in results
        ]))
        return
    for path, answers in results:
        if len(results) > 1:
            print(path)
        for name, chance in answers:
            print(f"{chance:6.1%}  {name}")


if __name__ == "__main__":
    main()
'''

_REVIEW = '''"""Shows which held-back images the trained model got wrong. Train first, then run: python review.py"""
import argparse
import json

from data import load_datasets
from model import CLASS_NAMES
from predict import answer, load_trained_model


def review():
    """Asks the model about every image that training held back."""
    model = load_trained_model()
    _, check_data = load_datasets()
    rows = []
    for index in check_data.indices:
        path, label = check_data.dataset.samples[index]
        guess, chance = answer(model, path)[0]
        rows.append({"path": str(path), "actual": CLASS_NAMES[label], "guess": guess, "chance": chance})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print every result as JSON")
    arguments = parser.parse_args()
    rows = review()
    if arguments.json:
        print(json.dumps(rows))
        return
    mistakes = [row for row in rows if row["guess"] != row["actual"]]
    print(f"{len(rows) - len(mistakes)} of {len(rows)} held-back images were correct.")
    for row in sorted(mistakes, key=lambda row: row["chance"], reverse=True):
        print(f"{row['path']}: guessed {row['guess']} ({row['chance']:.0%}), actually {row['actual']}")


if __name__ == "__main__":
    main()
'''

_SERVE = '''"""A small web API for your trained model. Train first, then run: python serve.py

Then ask it about an image from another terminal:
    curl --data-binary @photo.jpg -H "Content-Type: application/octet-stream" http://127.0.0.1:8000/predict
"""
import argparse
import io

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from predict import answer, load_trained_model

MAX_IMAGE_BYTES = 25 * 1024 * 1024

app = FastAPI(title="AI Bricks model")
model = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(request: Request):
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "The image is larger than 25 MB.")
    try:
        answers = await run_in_threadpool(answer, model, io.BytesIO(bytes(data)))
    except OSError:
        raise HTTPException(422, "The request body isn't a readable image.") from None
    return {
        "best": answers[0][0],
        "answers": [{"category": name, "chance": chance} for name, chance in answers],
    }


def main():
    global model
    parser = argparse.ArgumentParser(description="Serves the trained model over HTTP.")
    parser.add_argument("--port", type=int, default=8000)
    # This only accepts requests from this computer. Before opening it to a network,
    # change the host to "0.0.0.0" and put a login and rate limit in front of it.
    parser.add_argument("--host", default="127.0.0.1")
    arguments = parser.parse_args()
    model = load_trained_model()
    uvicorn.run(app, host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
'''


def trainable_categories(analysis: Analysis) -> list[str] | None:
    """The category names if this build can train inside the app, otherwise None."""
    if analysis.socket.output.kind == IMAGE and analysis.head.brick.id == "classify":
        return parse_names(analysis.head.params["classes"])
    return None


def labelling_info(analysis: Analysis) -> tuple[str | None, list[str] | None]:
    """What the labelling tool can sort for this build: its kind of data and its categories."""
    kind = analysis.socket.output.kind
    if analysis.head.brick.id == "classify" and kind in (IMAGE, TEXT) and not analysis.target:
        return ("image" if kind == IMAGE else "text"), parse_names(analysis.head.params["classes"])
    return None, None


def _generate_py(analysis: Analysis) -> str:
    return _GENERATE_PAIR if analysis.target else _GENERATE_NEXT


def _text_data_py(analysis: Analysis) -> str:
    socket, head, target = analysis.socket.output, analysis.head, analysis.target
    task = head.brick.id
    tokenizer = _TOKENIZER
    if target:
        config = _fill(_BANNER, LINES="# A CSV file with one pair per row:\n#   source,target\n#   \"hello there\",\"bonjour\"")
        config += 'DATA_FILE = "data/pairs.csv"\nSOURCE_COLUMN = "source"\nTARGET_COLUMN = "target"'
        return _fill(_TEXT_PAIRS, BANNER=config, S_LENGTH=socket.steps, S_VOCAB=socket.features,
                     T_LENGTH=target.output.steps, T_VOCAB=target.output.features, TOKENIZER=tokenizer)
    if task == "generate":
        config = _fill(_BANNER, LINES="# A plain text file containing all your training text.") + 'DATA_FILE = "data/corpus.txt"'
        return _fill(_TEXT_CORPUS, BANNER=config, LENGTH=socket.steps, VOCAB=socket.features, TOKENIZER=tokenizer)

    pretrained = any(s.brick.id == "pretrained_language" for s in analysis.steps)
    answer = "label" if task == "classify" else "target"
    example = "Billing" if task == "classify" else "3.5"
    config = _fill(_BANNER, LINES=f"# A CSV file with one example per row:\n#   text,{answer}\n#   \"My invoice is wrong\",{example}")
    config += f'DATA_FILE = "data/texts.csv"\nTEXT_COLUMN = "text"\nANSWER_COLUMN = "{answer}"'
    return _fill(
        _TEXT_ROWS,
        EXTRA_IMPORT="\nfrom model import CLASS_NAMES\n" if task == "classify" else "",
        BANNER=config,
        LENGTH=socket.steps,
        VOCAB=socket.features,
        TOKENIZER=tokenizer,
        HF=_HF_TOKENIZER if pretrained else "",
        MAKE_TOKENIZER="PretrainedTokenizer()" if pretrained else "SimpleTokenizer.build([split_words(row[TEXT_COLUMN]) for row in train_rows], VOCAB_SIZE)",
        LOAD_BODY=_TEXT_LOAD_CLASSIFY if task == "classify" else _TEXT_LOAD_NUMBER,
    )


_TEXT_SAMPLES = {
    "classify": '''
LENGTH, VOCAB = @@LENGTH@@, @@VOCAB@@
NUM_CLASSES = @@CLASSES@@


def make_sample_batch():
    """Each category uses its own token. The last position is padding, as in real text."""
    labels = torch.arange(BATCH_SIZE) % NUM_CLASSES
    tokens = (4 + labels % (VOCAB - 4)).unsqueeze(1).repeat(1, LENGTH)
    tokens[:, -1] = 0
    return tokens, labels
''',
    "predict_number": '''
LENGTH, VOCAB = @@LENGTH@@, @@VOCAB@@


def make_sample_batch():
    """The number to predict depends on which token each text uses."""
    index = torch.arange(BATCH_SIZE)
    tokens = (4 + index % (VOCAB - 4)).unsqueeze(1).repeat(1, LENGTH)
    tokens[:, -1] = 0
    return tokens, torch.linspace(-1, 1, BATCH_SIZE).unsqueeze(1)
''',
    "generate": '''
LENGTH, VOCAB = @@LENGTH@@, @@VOCAB@@
CYCLE = min(8, VOCAB - 4)


def make_sample_batch():
    """Tokens that repeat in a fixed cycle, so the next token is learnable."""
    start = torch.arange(BATCH_SIZE).unsqueeze(1)
    position = torch.arange(LENGTH + 1).unsqueeze(0)
    tokens = 4 + (start + position) % CYCLE
    return tokens[:, :-1], tokens[:, 1:]
''',
    "pairs": '''
SOURCE_LENGTH, SOURCE_VOCAB = @@LENGTH@@, @@VOCAB@@
TARGET_LENGTH, TARGET_VOCAB = @@T_LENGTH@@, @@T_VOCAB@@
CYCLE = min(8, SOURCE_VOCAB - 4, TARGET_VOCAB - 4)


def make_sample_batch():
    """The answer repeats a cycle chosen by the source, so it is learnable."""
    start = torch.arange(BATCH_SIZE).unsqueeze(1)
    source = 4 + (start + torch.arange(SOURCE_LENGTH).unsqueeze(0)) % CYCLE
    answer = 4 + (start + torch.arange(TARGET_LENGTH).unsqueeze(0)) % CYCLE
    target_in = torch.cat([torch.full((BATCH_SIZE, 1), 2), answer[:, :-1]], dim=1)  # 2 starts the answer
    return (source, target_in), answer
''',
}


def _text_sample_py(analysis: Analysis) -> str:
    socket, head, target = analysis.socket.output, analysis.head, analysis.target
    key = "pairs" if target else head.brick.id
    classes = len(parse_names(head.params["classes"])) if head.brick.id == "classify" else 0
    return _SAMPLE_HEADER.replace("import math\n\n", "") + _fill(
        _TEXT_SAMPLES[key],
        LENGTH=socket.steps,
        VOCAB=socket.features,
        CLASSES=classes,
        T_LENGTH=target.output.steps if target else 0,
        T_VOCAB=target.output.features if target else 0,
    )


# ----------------------------------------------------------- train.py and smoke_test.py

_SMOKE = '''"""The test run: a few training steps on generated sample data.

This proves the model is wired correctly and can learn something.
It does NOT prove the model will be accurate on your data.
"""
import json

import torch
from torch import nn

from model import build_model
from sample_data import make_sample_batch

STEPS = 60
@@LOSS_DEF@@


def report(**event):
    print(json.dumps(event), flush=True)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    # Pretrained weights are skipped here so the test run needs no download.
    model = build_model(use_pretrained_weights=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    report(event="model", parameters=sum(p.numel() for p in model.parameters()),
           trainable=sum(p.numel() for p in trainable))

    inputs, targets = make_sample_batch()
    model.train()
    outputs = @@CALL@@
    report(event="check", check="connections", passed=True)

    expected = @@EXPECTED@@
    if tuple(outputs.shape) != tuple(expected):
        report(event="check", check="shapes", passed=False,
               detail=f"Output is {tuple(outputs.shape)} but should be {tuple(expected)}.")
        raise SystemExit(1)
    report(event="check", check="shapes", passed=True)

    optimizer = torch.optim.Adam(trainable, lr=3e-3)
    losses = []
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        loss = LOSS_FUNCTION(@@CALL@@, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        report(event="loss", step=step, loss=loss.item())

    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    went_down = torch.isfinite(torch.tensor(losses)).all().item() and last < first * 0.9
    report(event="check", check="loss", passed=bool(went_down), first=first, last=last)
    raise SystemExit(0 if went_down else 1)


if __name__ == "__main__":
    main()
'''

_TRAIN = '''"""Trains the model on your data. Fill in data.py first, then run: python train.py"""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import load_datasets
from model import build_model

# Your "@@DIAL@@" training setting. Change these freely.
EPOCHS = @@EPOCHS@@
BATCH_SIZE = @@BATCH@@
LEARNING_RATE = @@LR@@
@@LOSS_DEF@@
SAVE_TO = "model.pt"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save(checkpoint):
    """Writes the file in one step, so stopping early can't leave a broken model."""
    torch.save(checkpoint, SAVE_TO + ".partial")
    os.replace(SAVE_TO + ".partial", SAVE_TO)


def evaluate(model, loader, device):
    """Returns the average loss@@EVAL_DOC@@ on data the model did not train on."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            @@TO_DEVICE@@
            outputs = @@CALL@@
            total_loss += LOSS_FUNCTION(outputs, targets).item() * len(targets)
            count += len(targets)
@@EVAL_EXTRA@@
    return total_loss / count, correct / count


def main():
    device = pick_device()
    train_data, val_data = load_datasets()
    print(f"Training on {len(train_data)} samples, checking on {len(val_data)}, using {device}.")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=len(train_data) > BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = build_model().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE)

    checkpoint = {
        "normalisation": getattr(train_data, "normalisation", None),
        "vocabulary": getattr(train_data, "vocabulary", None),
    }
    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, count = 0.0, 0
        for inputs, targets in train_loader:
            @@TO_DEVICE@@
            optimizer.zero_grad()
            loss = LOSS_FUNCTION(@@CALL@@, targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        val_loss, accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{EPOCHS}  training loss {total / count:.4f}  check loss {val_loss:.4f}@@EPOCH_EXTRA@@")
        # Keep the version that did best on the check data.
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint["model"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save(checkpoint)
@@AFTER@@
    save(checkpoint)
    print(f"Saved the best model to {SAVE_TO}.")


if __name__ == "__main__":
    main()
'''

_ANOMALY_AFTER = '''
    # Rebuild errors far above normal are flagged as anomalies.
    model.load_state_dict(checkpoint["model"])
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, targets in DataLoader(train_data, batch_size=BATCH_SIZE):
            inputs, targets = inputs.to(device), targets.to(device)
            difference = (model(inputs) - targets) ** 2
            errors.append(difference.flatten(start_dim=1).mean(dim=1).cpu())
    errors = torch.cat(errors)
    std = errors.std().item() if len(errors) > 1 else 0.0
    checkpoint["anomaly_threshold"] = errors.mean().item() + 3 * std
    print(f"Inputs with a rebuild error above {checkpoint['anomaly_threshold']:.5f} count as anomalies.")
'''


def _expected_shape(analysis: Analysis) -> str:
    socket, head = analysis.socket.output, analysis.head
    t = head.brick.id
    if t == "classify":
        return f"(BATCH_SIZE, {len(parse_names(head.params['classes']))})"
    if t == "predict_number":
        return "(BATCH_SIZE, 1)"
    if t == "forecast":
        return f"(BATCH_SIZE, {head.params['horizon']}, {socket.features})"
    if t == "generate":
        expected = analysis.target.output if analysis.target else socket
        return f"(BATCH_SIZE, {expected.steps}, {expected.features})"
    if socket.kind == IMAGE:
        return f"(BATCH_SIZE, {socket.channels}, {socket.height}, {socket.width})"
    return f"(BATCH_SIZE, {socket.steps}, {socket.features})"


def _loss_def(analysis: Analysis) -> str:
    task = analysis.head.brick.id
    if task == "generate":
        return (
            "_cross_entropy = nn.CrossEntropyLoss(ignore_index=0)  # token 0 is padding\n\n\n"
            "def LOSS_FUNCTION(outputs, targets):\n"
            "    return _cross_entropy(outputs.transpose(1, 2), targets)"
        )
    return "LOSS_FUNCTION = nn.CrossEntropyLoss()" if task == "classify" else "LOSS_FUNCTION = nn.MSELoss()"


def _smoke_py(analysis: Analysis) -> str:
    call = "model(*inputs)" if analysis.target else "model(inputs)"
    text = _fill(_SMOKE, LOSS_DEF=_loss_def(analysis), CALL=call, EXPECTED=_expected_shape(analysis))
    return text.replace("from sample_data import make_sample_batch", "from sample_data import BATCH_SIZE, make_sample_batch")


def training_settings(graph: GraphIn) -> dict:
    dial = TRAINING_DIAL[graph.training] | (graph.trainingOverride or {})
    if graph.trainingOverride:
        dial["name"] = "Custom"
    dial["epochs"], dial["batch_size"] = int(dial["epochs"]), int(dial["batch_size"])
    return dial


def _train_py(graph: GraphIn, analysis: Analysis) -> str:
    dial = training_settings(graph)
    task = analysis.head.brick.id
    is_classify = task == "classify"
    return _fill(
        _TRAIN,
        DIAL=dial["name"],
        EPOCHS=dial["epochs"],
        BATCH=dial["batch_size"],
        LR=dial["learning_rate"],
        LOSS_DEF=_loss_def(analysis),
        CALL="model(*inputs)" if analysis.target else "model(inputs)",
        TO_DEVICE=(
            "inputs, targets = [item.to(device) for item in inputs], targets.to(device)"
            if analysis.target else "inputs, targets = inputs.to(device), targets.to(device)"
        ),
        EVAL_DOC=" and accuracy" if is_classify else "",
        EVAL_EXTRA="            correct += (outputs.argmax(dim=1) == targets).sum().item()" if is_classify else "",
        EPOCH_EXTRA="  check accuracy {accuracy:.1%}" if is_classify else "",
        AFTER=_ANOMALY_AFTER if task == "detect_anomalies" else "",
    )


def _data_description(analysis: Analysis) -> list[str]:
    socket, head = analysis.socket, analysis.head
    s = socket.output
    task = head.brick.id
    if s.kind == TEXT:
        if analysis.target:
            return [
                "A CSV file with the columns source and target, one pair per row.",
                f"Inputs are cut to {s.steps} tokens and outputs to {analysis.target.output.steps} tokens.",
            ]
        if task == "generate":
            return [
                "A plain text file containing all your training text.",
                f"The model reads {s.steps} tokens at a time and learns to predict the next one.",
            ]
        lines = [f"A CSV file with a text column. Each text is cut to {s.steps} tokens."]
        if task == "classify":
            lines.append("Add a label column containing one of: " + ", ".join(parse_names(head.params["classes"])) + ".")
        else:
            lines.append("Add a target column containing the number to predict.")
        return lines
    if s.kind == IMAGE:
        colour = "colour" if s.channels == 3 else "greyscale"
        lines = [f"Images of any size. They are resized to {s.height}×{s.width} pixels and converted to {colour}."]
        if task == "classify":
            names = ", ".join(parse_names(head.params["classes"]))
            lines.append(f"One folder per category, named exactly: {names}.")
        elif task == "predict_number":
            lines.append("A CSV file with the columns filename and value.")
        else:
            lines.append("Normal images only. The model learns what normal looks like.")
    else:
        columns = ", ".join(parse_names(socket.params["columns"]))
        lines = [f"A CSV file with one row per time step, oldest first, and the columns: {columns}."]
        lines.append(f"The model looks at {s.steps} rows at a time.")
        if task == "forecast":
            lines.append(f"It predicts the next {head.params['horizon']} rows of the same columns.")
        elif task == "classify":
            names = ", ".join(parse_names(head.params["classes"]))
            lines.append(f"Add a label column containing one of: {names}.")
        elif task == "predict_number":
            lines.append("Add a target column containing the number to predict.")
        else:
            lines.append("Use normal readings only. The model learns what normal looks like.")
    return lines


def data_description(analysis: Analysis) -> list[str]:
    return _data_description(analysis)


def _readme(graph: GraphIn, analysis: Analysis) -> str:
    bullets = "\n".join(f"- {line}" for line in _data_description(analysis))
    pretrained = any(s.brick.id in PRETRAINED for s in analysis.steps)
    note = "\nThe first training run downloads the pretrained weights.\n" if pretrained else ""
    generate_row = "| `generate.py` | Produces text from the trained model |\n" if analysis.head.brick.id == "generate" else ""
    predict_row = ""
    predict_note = ""
    if trainable_categories(analysis):
        predict_row = (
            "| `predict.py` | Asks the trained model about one image |\n"
            "| `review.py` | Lists the held-back images the model got wrong |\n"
            "| `serve.py` | Runs a small web API for the trained model |\n"
        )
        predict_note = (
            "\n## How to use it\n\n"
            "- `python predict.py photo.jpg` asks the trained model about one image.\n"
            "- `python review.py` lists the held-back images it got wrong. It needs your data folder in place.\n"
            "- `python serve.py` starts a web API on this computer. Send it an image with:\n\n"
            "  ```\n"
            '  curl --data-binary @photo.jpg -H "Content-Type: application/octet-stream" http://127.0.0.1:8000/predict\n'
            "  ```\n"
        )
    title = " ".join(graph.projectName.split()) or "My model"
    return f"""# {title}

Designed in AI Bricks. This is plain PyTorch, so you can read and change all of it.

## What data it expects

{bullets}

## How to train it

1. Install the requirements:

   ```
   pip install -r requirements.txt
   ```

2. Open `data.py` and fill in the section marked `PUT YOUR DATA HERE`.
3. Start training:

   ```
   python train.py
   ```
{note}
The best version is saved to `model.pt`, along with any scaling values you need to prepare new data.
{predict_note}
## Files

| File | What it does |
|---|---|
| `model.py` | Connects your bricks into one model |
| `bricks.py` | Holds one small module for each kind of brick |
| `data.py` | Loads your data |
| `train.py` | Trains the model |
{generate_row}{predict_row}| `smoke_test.py` | Repeats the test run |
| `sample_data.py` | Generates the test run's sample data |

## What the test run proved

The build passed a 60-step test on generated sample data. That shows the bricks connect correctly and the loss goes down.
It does not show how accurate the model will be on your data. Only training and checking on your own data can show that.

Run it again at any time:

```
python smoke_test.py
```
"""


def _requirements(analysis: Analysis) -> str:
    lines = ["torch>=2.2"]
    if analysis.socket.output.kind == IMAGE:
        lines += ["torchvision>=0.17", "pillow>=10"]
    if any(s.brick.id == "pretrained_language" for s in analysis.steps):
        lines.append("transformers>=4.40")
    if trainable_categories(analysis):
        lines += ["fastapi>=0.115", "uvicorn>=0.30"]
    return "\n".join(lines) + "\n"


def generate(graph: GraphIn, analysis: Analysis) -> dict[str, str]:
    """Returns {file name: contents}. The analysis must be free of issues."""
    if not analysis.ok:
        raise ValueError("Cannot generate code for a build with issues.")
    extra = {"generate.py": _generate_py(analysis)} if analysis.head.brick.id == "generate" else {}
    if trainable_categories(analysis):
        extra |= {"predict.py": _PREDICT, "review.py": _REVIEW, "serve.py": _SERVE}
    return {
        **extra,
        "README.md": _readme(graph, analysis),
        "model.py": _model_py(graph, analysis),
        "bricks.py": _bricks_py(analysis),
        "data.py": _data_py(analysis),
        "train.py": _train_py(graph, analysis),
        "smoke_test.py": _smoke_py(analysis),
        "sample_data.py": _sample_py(analysis),
        "requirements.txt": _requirements(analysis),
    }


def build_hash(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
