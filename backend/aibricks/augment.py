"""Varying training photos (data augmentation), and previews of what training will see.

Photos are changed fresh each time training shows them, and only the training photos: the held-back
photos are never changed. Saving altered copies into a dataset instead is a known way to fake a good
score, because copies of a training photo can land among the held-back ones.

`varied_steps` is copied into each generated data.py, so the preview, the app's training and the
downloaded project all vary photos in exactly the same way.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys

from . import training
from .training import TrainingError

PREVIEW_COUNT = 8
DEFAULTS = {"flip": True, "crop": 30, "rotate": 0, "light": 25, "auto": False}


def varied_steps(size, settings, transforms):
    """The training-only changes to each photo, from the Image folder block's settings."""
    steps = []
    zoom = settings["crop"] / 100
    if zoom > 0:
        steps.append(transforms.RandomResizedCrop((size, size), scale=(1 - zoom, 1.0), ratio=(0.85, 1.18)))
    else:
        steps.append(transforms.Resize((size, size)))
    if settings["flip"]:
        steps.append(transforms.RandomHorizontalFlip())
    if settings["rotate"]:
        steps.append(transforms.RandomRotation(settings["rotate"]))
    light = settings["light"] / 100
    if light:
        steps.append(transforms.ColorJitter(brightness=light, contrast=light, saturation=light))
    if settings["auto"]:
        # One random change of a random strength per photo (Müller and Hutter, 2021).
        steps.append(transforms.TrivialAugmentWide())
    return steps


SOURCE = inspect.getsource(varied_steps)


def settings_from(params: dict) -> dict:
    """The Image folder block's settings, as the plain values data.py uses."""
    return {
        "flip": bool(params.get("vary_flip", DEFAULTS["flip"])),
        "crop": int(params.get("vary_crop", DEFAULTS["crop"])),
        "rotate": int(params.get("vary_rotate", DEFAULTS["rotate"])),
        "light": int(params.get("vary_light", DEFAULTS["light"])),
        "auto": bool(params.get("vary_auto", DEFAULTS["auto"])),
    }


_PREVIEW = """
import base64, io, json, sys
from PIL import Image
from torchvision import transforms
__SOURCE__
request = json.loads(sys.stdin.read())
size, settings = request['size'], request['settings']
original = Image.open(request['path']).convert('RGB')
pipeline = transforms.Compose(varied_steps(size, settings, transforms) + ([transforms.Grayscale(1)] if request['channels'] == 1 else []))
def jpeg(image):
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, 'JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()
plain = original.resize((size, size))
plain = plain.convert('L') if request['channels'] == 1 else plain
print(json.dumps({'original': jpeg(plain), 'varied': [jpeg(pipeline(original)) for _ in range(request['count'])]}))
""".replace("__SOURCE__", SOURCE)


def _sample_photo(dataset: str):
    from . import datasets  # imported here, because datasets imports training
    meta = datasets._read(dataset)
    if meta["kind"] != "image":
        raise TrainingError(422, "Previews need a photo dataset.")
    for name in meta["categories"]:
        photos = sorted(training._images_in(training._category_folder(name, dataset)))
        if photos:
            return photos[len(photos) // 2]
    raise TrainingError(422, "Add some photos to this dataset first.")


def preview(dataset: str, size: int, channels: int, params: dict) -> dict:
    """One of the dataset's photos as the model sees it, and several varied versions of it."""
    path = _sample_photo(dataset)
    request = {"path": str(path), "size": size, "channels": channels, "settings": settings_from(params), "count": PREVIEW_COUNT}
    try:
        done = subprocess.run([sys.executable, "-I", "-c", _PREVIEW], input=json.dumps(request), capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise TrainingError(504, "Making the examples took too long.") from None
    if done.returncode != 0:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise TrainingError(500, lines[-1][:300] if lines else "The examples couldn't be made.")
    reply = json.loads(done.stdout)
    return {"original": f"data:image/jpeg;base64,{reply['original']}", "varied": [f"data:image/jpeg;base64,{item}" for item in reply["varied"]]}
