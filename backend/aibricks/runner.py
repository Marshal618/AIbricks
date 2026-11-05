"""Runs a generated project's smoke test in a separate, time-limited process.

Only code produced by the generator ever runs here. User text reaches it only as
validated string literals, never as code.
"""
from __future__ import annotations

import asyncio
import json
import os
import resource
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

TIME_LIMIT_SECONDS = 180
CPU_LIMIT_SECONDS = 400
OUTPUT_LINE_LIMIT = 4096
_slots = asyncio.Semaphore(2)


def _limit_process() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _friendly_failure(stderr: str) -> str:
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1][:OUTPUT_LINE_LIMIT] if lines else "The test run stopped unexpectedly."


async def run_smoke_test(files: dict[str, str]) -> AsyncIterator[dict]:
    """Streams events from the smoke test, ending with exactly one {"event": "result"}."""
    async with _slots:
        with tempfile.TemporaryDirectory(prefix="aibricks-") as folder:
            for name, contents in files.items():
                (Path(folder) / name).write_text(contents, encoding="utf-8")
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-c",
                "import runpy,sys; sys.path.insert(0, '.'); runpy.run_path('smoke_test.py', run_name='__main__')",
                cwd=folder,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "2", "HOME": folder},
                preexec_fn=_limit_process,
            )
            errors: list[bytes] = []

            async def drain_errors() -> None:
                async for line in process.stderr:
                    errors.append(line)
                    del errors[:-50]

            drain = asyncio.create_task(drain_errors())
            loss_passed = False
            failed_check: dict | None = None
            deadline = asyncio.get_running_loop().time() + TIME_LIMIT_SECONDS
            timed_out = False
            try:
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), max(remaining, 0.01))
                    except asyncio.TimeoutError:
                        timed_out = True
                        break
                    if not line:
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") == "check":
                        if event.get("check") == "loss":
                            loss_passed = bool(event.get("passed"))
                        if not event.get("passed"):
                            failed_check = event
                    yield event
            finally:
                if process.returncode is None and timed_out:
                    process.kill()
                elif process.returncode is None:
                    # Reader left early (client disconnected) or stdout closed.
                    try:
                        await asyncio.wait_for(process.wait(), 5)
                    except asyncio.TimeoutError:
                        process.kill()
                await process.wait()
                await drain

            if timed_out:
                yield {"event": "result", "passed": False, "detail": f"The test run took longer than {TIME_LIMIT_SECONDS} seconds. Try smaller images, fewer steps per sample or fewer bricks."}
            elif process.returncode == 0 and loss_passed:
                yield {"event": "result", "passed": True}
            elif failed_check and failed_check.get("check") == "loss":
                yield {"event": "result", "passed": False, "detail": "The model runs, but the loss did not go down on the sample data. Try a simpler build, or lower Dropout."}
            elif failed_check:
                yield {"event": "result", "passed": False, "detail": failed_check.get("detail", "A check failed.")}
            else:
                yield {"event": "result", "passed": False, "detail": _friendly_failure(b"".join(errors).decode("utf-8", "replace"))}
