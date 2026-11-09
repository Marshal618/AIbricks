"""What this computer offers for training, and how much the workspace holds, for the home screen."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SIZE_CACHE_SECONDS = 30

_PROBE = """
import json, torch
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
name = torch.cuda.get_device_name(0) if device == 'cuda' else None
print(json.dumps({'torch': torch.__version__.split('+')[0], 'device': device, 'deviceName': name}))
"""

_probe: dict | None = None
_probe_lock = threading.Lock()
_sizes: tuple[float, dict] | None = None


def _accelerator() -> dict:
    """Asked once in a separate process, so the API itself never loads PyTorch."""
    global _probe
    with _probe_lock:
        if _probe is None:
            try:
                done = subprocess.run([sys.executable, "-I", "-c", _PROBE], capture_output=True, text=True, timeout=60)
                _probe = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else {}
            except (subprocess.TimeoutExpired, ValueError, IndexError):
                _probe = {}
        return _probe


def _folder_bytes(folder: Path) -> int:
    total = 0
    stack = [folder]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total


def _storage(root: Path) -> dict:
    global _sizes
    if _sizes is not None and time.time() - _sizes[0] < SIZE_CACHE_SECONDS and _sizes[1]["path"] == str(root):
        return _sizes[1]
    # Trained models live in runs, builds in projects and downloaded base models in cache. The rest is data.
    groups = {"runs": "models", "projects": "projects", "cache": "downloads"}
    parts = {"data": 0, "models": 0, "projects": 0, "downloads": 0}
    if root.exists():
        for entry in root.iterdir():
            size = _folder_bytes(entry) if entry.is_dir() else entry.stat().st_size
            parts[groups.get(entry.name, "data")] += size
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    result = {"path": str(root), "bytes": sum(parts.values()), "parts": parts, "freeBytes": free}
    _sizes = (time.time(), result)
    return result


def info() -> dict:
    from . import training

    device = _accelerator()
    labels = {"cuda": "NVIDIA GPU", "mps": "Apple GPU (Metal)", "cpu": "CPU"}
    return {
        "device": device.get("device", "unknown"),
        "deviceLabel": device.get("deviceName") or labels.get(device.get("device", ""), "Unknown"),
        "torch": device.get("torch"),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "cpus": os.cpu_count() or 1,
        "storage": _storage(training.workspace()),
    }
