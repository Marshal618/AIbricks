"""Unpacks a Raspberry Pi kit and runs its script the way the Pi would, with this computer's Python."""
import io
import json
import subprocess
import sys
import zipfile


def unpack(reply, tmp_path):
    assert reply.status_code == 200, reply.text
    assert reply.headers["content-type"] == "application/zip" and reply.headers["content-disposition"].endswith('-pi.zip"')
    archive = zipfile.ZipFile(io.BytesIO(reply.content))
    names = archive.namelist()
    folder = names[0].split("/")[0]
    assert all(name.startswith(folder + "/") for name in names)
    assert archive.getinfo(f"{folder}/run.py").external_attr >> 16 == 0o755
    archive.extractall(tmp_path)
    return tmp_path / folder, {name.split("/", 1)[1] for name in names}


def run(folder, *args):
    done = subprocess.run([sys.executable, "-I", "run.py", *map(str, args)], cwd=folder, capture_output=True, text=True, timeout=120)
    return done.returncode, done.stdout, done.stderr


def answers(folder, photo):
    code, out, err = run(folder, "--json", "--photo", photo)
    assert code == 0, err
    return json.loads(out.strip().splitlines()[-1])["answers"]
