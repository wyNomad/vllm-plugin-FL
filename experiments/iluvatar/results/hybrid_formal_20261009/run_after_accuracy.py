#!/usr/bin/env python3
"""Gate long-context checks and the formal benchmark on Math500 accuracy."""

import fcntl
import json
import re
import subprocess
import time
from pathlib import Path


BASE = Path("/workspace/wyNomad/results")
ACCURACY = BASE / "hybrid_accuracy_fixed_20261009"
CORRECTNESS = BASE / "hybrid_correctness_20261009"
FORMAL = BASE / "hybrid_formal_20261009"
NAME = "wynomad-minicpm-s2-gpu0"


def read_exit(directory):
    marker = directory / "runner.exit"
    return marker.read_text().strip() if marker.exists() else None


def require_free_container():
    result = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"name=^{NAME}$"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"personal container still running: {NAME}")


def check_long_response(mode):
    directory = CORRECTNESS / mode
    if read_exit(directory) != "0":
        raise RuntimeError(f"{mode} runner did not succeed")
    payload = json.loads((directory / "response.json").read_text())
    if payload.get("error"):
        raise RuntimeError(f"{mode} API error: {payload['error']}")
    content = payload["choices"][0]["message"]["content"]
    input_tokens = payload["usage"]["prompt_tokens"]
    if input_tokens < 16000:
        raise RuntimeError(f"{mode} prompt too short: {input_tokens}")
    if not re.search(r"(?<!\d)4(?!\d)|\bfour\b", content, re.I):
        raise RuntimeError(f"{mode} answer does not contain 4: {content[:500]}")
    print(mode, "input_tokens", input_tokens, "answer", repr(content[:500]), flush=True)


def run_step(command, log_name, timeout):
    print("START", command, flush=True)
    with (FORMAL / log_name).open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    print("END", command, "exit", result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(f"step failed with exit {result.returncode}: {command}")


def main():
    FORMAL.mkdir(parents=True, exist_ok=True)
    with (FORMAL / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (FORMAL / "pipeline.error").unlink(missing_ok=True)
        deadline = time.monotonic() + 3 * 3600
        while read_exit(ACCURACY) is None:
            if time.monotonic() > deadline:
                raise TimeoutError("Math500 did not finish within three hours")
            time.sleep(15)
        if not (ACCURACY / "accuracy_succeeded.ok").exists() or not (ACCURACY / "evaluation_succeeded.ok").exists():
            raise RuntimeError("Math500 accuracy or evaluation did not pass")
        require_free_container()
        if read_exit(ACCURACY) != "0":
            print("Math500 runner had a cleanup timeout; container is now absent", flush=True)
        print((ACCURACY / "accuracy_succeeded.ok").read_text().strip(), flush=True)

        for mode in ("triton_long", "hybrid_long"):
            directory = CORRECTNESS / mode
            if read_exit(directory) != "0":
                run_step(["python3", str(CORRECTNESS / "probe.py"), mode],
                         f"{mode}.log", 1800)
            check_long_response(mode)
            require_free_container()

        if read_exit(FORMAL) is None:
            run_step(["python3", str(FORMAL / "run_and_cleanup.py")],
                     "formal_supervisor.log", 11 * 3600)
        if read_exit(FORMAL) != "0" or not (FORMAL / "benchmark_succeeded.ok").exists():
            raise RuntimeError("formal throughput benchmark failed")
        require_free_container()
        (FORMAL / "pipeline_succeeded.ok").write_text(
            "Math500, 16k Triton/hybrid checks, and official throughput all passed\n"
        )
        print("PIPELINE SUCCESS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PIPELINE ERROR", repr(exc), flush=True)
        (FORMAL / "pipeline.error").write_text(repr(exc) + "\n")
        raise
