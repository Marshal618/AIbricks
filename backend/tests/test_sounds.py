"""Sound datasets: recordings cut into clips, versions, and a sound sorter trained on them."""
import io
import subprocess
import sys
import wave
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_training import BINARY, picture, RED, workspace  # noqa: F401

from aibricks import sounds, training
from aibricks.main import app

client = TestClient(app)
JSON = {"Content-Type": "application/json"}


@pytest.fixture(autouse=True)
def shared_cache(monkeypatch):
    from test_text_models import CACHE
    monkeypatch.setenv("AI_BRICKS_CACHE", str(CACHE))


def recording(pitch=440.0, seconds=1.0, rate=16000, width=2, channels=1, seed=0, loud=0.4):
    """A tone with a little noise, as a WAV file."""
    rng = np.random.default_rng(seed)
    times = np.arange(int(rate * seconds)) / rate
    samples = loud * np.sin(2 * np.pi * pitch * times + rng.uniform(0, 6)) + 0.02 * rng.standard_normal(len(times))
    samples = np.clip(samples, -1, 1)
    if width == 1:
        raw = ((samples * 127) + 128).astype(np.uint8).tobytes()
    elif width == 2:
        raw = (samples * 32767).astype("<i2").tobytes()
    elif width == 3:
        values = (samples * (2**23 - 1)).astype(np.int32)
        raw = b"".join(int(value).to_bytes(3, "little", signed=True) for value in values)
    else:
        raw = (samples * (2**31 - 1)).astype("<i4").tobytes()
    if channels > 1:
        frame = len(raw) // len(samples)
        raw = b"".join(raw[i * frame:(i + 1) * frame] * channels for i in range(len(samples)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(width)
        file.setframerate(rate)
        file.writeframes(raw)
    return buffer.getvalue()


def create(categories=("Low", "High")):
    reply = client.post("/api/datasets", json={"name": "Whistles", "kind": "sound", "categories": list(categories)})
    assert reply.status_code == 200, reply.text
    return reply.json()["id"]


def add(dataset, category, data):
    return client.post(f"/api/datasets/{dataset}/sounds", params={"category": category}, content=data, headers=BINARY)


# ------------------------------------------------------------------ reading audio

@pytest.mark.parametrize("width, rate, channels", [(1, 8000, 1), (2, 44100, 2), (3, 48000, 1), (4, 22050, 2)])
def test_common_wav_formats_are_read_as_16khz_mono(width, rate, channels):
    samples = sounds.read_wav(io.BytesIO(recording(seconds=0.5, rate=rate, width=width, channels=channels)))
    assert samples.dtype == np.float32 and abs(len(samples) - 8000) <= 2
    assert 0.3 < np.abs(samples).max() <= 1.0
    # The tone's pitch survives resampling.
    peak = np.argmax(np.abs(np.fft.rfft(samples))) * 16000 / len(samples)
    assert abs(peak - 440) < 10


def test_recordings_are_cut_into_one_second_clips():
    samples = np.zeros(int(16000 * 2.5), np.float32)
    assert [len(clip) for clip in sounds.clips_of(samples)] == [16000, 16000, 16000]
    assert len(sounds.clips_of(np.zeros(16000 * 2 + 1000, np.float32))) == 2  # a tiny tail is dropped
    assert sounds.clips_of(np.zeros(3000, np.float32)) == []


def test_unusable_files_are_explained(workspace):  # noqa: F811
    dataset = create()
    assert "isn't a WAV" in add(dataset, "Low", b"ID3 not a wav at all").json()["detail"]["message"]
    floats = io.BytesIO()
    with wave.open(floats, "wb") as file:
        file.setnchannels(1), file.setsampwidth(2), file.setframerate(16000), file.writeframes(b"\0\0" * 16000)
    data = bytearray(floats.getvalue())
    data[20:22] = (3).to_bytes(2, "little")  # IEEE float format, which the wave module can't read
    assert add(dataset, "Low", bytes(data)).status_code == 422
    assert "quarter of a second" in add(dataset, "Low", recording(seconds=0.1)).json()["detail"]["message"]
    assert add(dataset, "Nope", recording()).status_code == 422
    words = client.post("/api/datasets", json={"name": "Words", "kind": "text", "categories": ["Low"]}).json()["id"]
    assert "can't hold recordings" in add(words, "Low", recording()).json()["detail"]["message"]
    assert client.post(f"/api/datasets/{dataset}/sounds", params={"category": "Low"}, content=recording()).status_code == 415


# ---------------------------------------------------------------------- storing

def test_clips_are_stored_listed_played_and_removed(workspace):  # noqa: F811
    dataset = create()
    reply = add(dataset, "Low", recording(seconds=2.5)).json()
    assert reply == {"added": 3, "duplicates": 0, "count": 3}
    assert add(dataset, "Low", recording(seconds=2.5)).json() == {"added": 0, "duplicates": 3, "count": 3}
    info = client.get(f"/api/datasets/{dataset}").json()
    assert info["kind"] == "sound" and info["counts"] == {"Low": 3, "High": 0} and info["labelled"] == 3

    listed = client.get(f"/api/datasets/{dataset}/sounds", params={"category": "Low"}).json()
    assert listed["total"] == 3 and len(listed["clips"]) == 3
    clip = listed["clips"][0]
    played = client.get(f"/api/datasets/{dataset}/sounds/{clip}", params={"category": "Low"})
    assert played.status_code == 200 and played.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(played.content)) as file:
        assert (file.getframerate(), file.getnchannels(), file.getnframes()) == (16000, 1, 16000)
    assert client.get(f"/api/datasets/{dataset}/sounds/../../x", params={"category": "Low"}).status_code == 404
    assert client.get(f"/api/datasets/{dataset}/sounds/{clip}", params={"category": "High"}).status_code == 404

    assert client.delete(f"/api/datasets/{dataset}/sounds/{clip}", params={"category": "Low"}).status_code == 200
    assert client.get(f"/api/datasets/{dataset}").json()["counts"]["Low"] == 2
    preview = client.get(f"/api/datasets/{dataset}/preview").json()
    assert [row["name"] for row in preview["categories"]] == ["Low", "High"] and len(preview["categories"][0]["samples"]) == 2


def test_versions_restore_clips_and_removing_a_category_keeps_a_copy(workspace):  # noqa: F811
    dataset = create()
    add(dataset, "Low", recording(seconds=3))
    add(dataset, "High", recording(1500, seconds=2))
    assert client.post(f"/api/datasets/{dataset}/versions", json={"message": "Start"}).status_code == 200
    clip = client.get(f"/api/datasets/{dataset}/sounds", params={"category": "Low"}).json()["clips"][0]
    client.delete(f"/api/datasets/{dataset}/sounds/{clip}", params={"category": "Low"})
    assert client.post(f"/api/datasets/{dataset}/versions/1/restore", headers=JSON).status_code == 200
    assert client.get(f"/api/datasets/{dataset}").json()["counts"] == {"Low": 3, "High": 2}

    add(dataset, "High", recording(1700, seed=4))  # unsaved, so removing High must save it first
    reply = client.delete(f"/api/datasets/{dataset}/categories", params={"name": "High"})
    assert reply.status_code == 200 and reply.json()["categories"] == ["Low"]
    versions = client.get(f"/api/datasets/{dataset}/versions").json()["versions"]
    assert versions[0]["message"] == "Saved automatically before removing High" and versions[0]["counts"]["High"] == 3  # newest first
    health = client.get(f"/api/datasets/{dataset}/health").json()
    assert health["kind"] == "sound"
    assert client.post(f"/api/datasets/{dataset}/health/check", headers=JSON).status_code == 422


# --------------------------------------------------------------------- training

def test_a_sound_sorter_learns_low_from_high(workspace):  # noqa: F811
    dataset = create()
    for seed in range(6):
        assert add(dataset, "Low", recording(220 + 15 * seed, seconds=2, seed=seed)).status_code == 200
        assert add(dataset, "High", recording(1800 + 60 * seed, seconds=2, seed=seed)).status_code == 200
    short = create(("A", "B"))
    add(short, "A", recording())
    assert "Record at least" in client.post(f"/api/datasets/{short}/train-sound", headers=JSON).json()["detail"]["message"]
    assert client.post(f"/api/datasets/{dataset}/train-sound").status_code == 415

    started = client.post(f"/api/datasets/{dataset}/train-sound", headers=JSON)
    assert started.status_code == 200, started.text
    job = wait_text_kind("sound")
    assert job["status"] == "finished", job["error"]
    assert job["task"] == "sound" and job["trainCount"] == 20 and job["checkCount"] == 4 and job["datasetVersion"] == 1
    assert job["best"]["accuracy"] == 1.0 and job["review"]["total"] == 4

    low = client.post(f"/api/models/{job['id']}/listen", params={"explain": "true"}, content=recording(260, seed=99), headers=BINARY)
    assert low.status_code == 200, low.text
    body = low.json()
    assert body["answers"][0]["category"] == "Low" and body["clips"] == 1
    assert body["spectrogram"]["height"] == 64 and body["spectrogram"]["width"] == 98
    assert len(body["spectrogram"]["values"]) == 64 * 98
    heat = body["heatmap"]
    assert heat["category"] == "Low" and len(heat["values"]) == heat["width"] * heat["height"] == 16 * 20
    high = client.post(f"/api/models/{job['id']}/listen", content=recording(2000, seconds=3, seed=98), headers=BINARY).json()
    assert high["answers"][0]["category"] == "High" and high["clips"] == 3 and high["heatmap"] is None

    assert client.post(f"/api/models/{job['id']}/listen", content=b"nope", headers=BINARY).status_code == 422
    assert client.post(f"/api/models/{job['id']}/listen", content=picture(RED), headers=BINARY).status_code == 422
    assert client.get(f"/api/models/{job['id']}/sound-mistakes").json() == {"mistakes": [], "total": 0}
    card = client.get(f"/api/models/{job['id']}/card").json()["markdown"]
    assert "Audio Spectrogram Transformer" in card and "clips" in card

    download = client.get(f"/api/train/{job['id']}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        names = archive.namelist()
        assert any(name.endswith("model/head.pt") for name in names) and not any("/data/" in name for name in names)
        archive.extractall(workspace / "download")
    training.close_live()
    project = next((workspace / "download").iterdir())
    (project / "try.wav").write_bytes(recording(1900, seed=5))
    done = subprocess.run([sys.executable, "predict.py", "try.wav", "--json"], cwd=project, capture_output=True, text=True, timeout=300,
                          env={"PATH": "/usr/bin:/bin", "HF_HOME": str(workspace.parent / "unused"), **_hf_env()})
    assert done.returncode == 0, done.stderr[-800:]
    assert '"category": "High"' in done.stdout.split("}")[0] + "}"


def test_planted_mistakes_can_be_heard_and_moved(workspace):  # noqa: F811
    dataset = create()
    for seed in range(8):
        add(dataset, "Low", recording(230 + 10 * seed, seconds=1, seed=seed))
        add(dataset, "High", recording(1900 + 40 * seed, seconds=1, seed=seed))
    # High tones filed as Low, picked so they land in the held-back fifth.
    planted = sorted((recording(2100 + 50 * seed, seed=100 + seed) for seed in range(12)), key=lambda data: _rank(data))
    for data in planted[:2]:
        add(dataset, "Low", data)
    client.post(f"/api/datasets/{dataset}/train-sound", headers=JSON)
    job = wait_text_kind("sound")
    assert job["status"] == "finished", job["error"]
    found = client.get(f"/api/models/{job['id']}/sound-mistakes").json()
    assert found["total"] >= 1
    mistake = found["mistakes"][0]
    assert mistake["actual"] == "Low" and mistake["guess"] == "High"
    clip = client.get(f"/api/train/{job['id']}/mistakes/{mistake['index']}")
    assert clip.status_code == 200 and clip.content[:4] == b"RIFF"
    moved = client.post(f"/api/train/{job['id']}/mistakes/{mistake['index']}/fix", json={"action": "move", "category": "High"})
    assert moved.status_code == 200
    assert client.get(f"/api/datasets/{dataset}").json()["counts"]["High"] == 9


def _rank(data):
    """Where a clip falls in training's held-back order: the lowest fingerprints are held back."""
    import hashlib
    clip = sounds.encode(sounds.clips_of(sounds.read_wav(io.BytesIO(data)))[0])
    name = hashlib.sha256(clip).hexdigest()[:32] + ".wav"
    return hashlib.sha256(name.encode()).hexdigest()


def _hf_env():
    from test_text_models import CACHE
    return {"HF_HOME": str(CACHE / "huggingface")}


def wait_text_kind(kind, limit=600):
    import time
    deadline = time.time() + limit
    while time.time() < deadline:
        job = client.get("/api/train/current", params={"kind": kind}).json()["job"]
        if job and job["status"] != "running":
            return job
        time.sleep(0.3)
    raise AssertionError("training did not finish")
