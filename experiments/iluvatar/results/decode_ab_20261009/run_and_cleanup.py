#!/usr/bin/env python3
"""Sequential decode-heavy backend comparison with automatic cleanup."""

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/workspace/wyNomad/results/decode_ab_20261009")
NAME = "wynomad-minicpm-s2-gpu0"
COMPILATION_CONFIG = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_num_of_warmups": 0,
}


def wait_ready(service):
    probe = [
        "docker", "exec", NAME, "python3", "-c",
        "import urllib.request; urllib.request.urlopen("
        "'http://127.0.0.1:9031/health', timeout=3)",
    ]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if service.poll() is not None:
            raise RuntimeError(f"vLLM exited early: {service.returncode}")
        ready = subprocess.run(probe, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=8)
        if ready.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError("vLLM service did not become healthy in ten minutes")


def run_variant(variant):
    directory = ROOT / variant
    directory.mkdir(parents=True, exist_ok=True)
    started_container = False
    try:
        launch = subprocess.run(
            ["python3", "/workspace/wyNomad/run_tianshu_container.py",
             "--detach", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45,
        )
        print("CONTAINER", variant, launch.stdout, flush=True)
        if launch.returncode != 0:
            raise RuntimeError(f"container launcher exited {launch.returncode}")
        started_container = True
        serve = ["docker", "exec"]
        if variant == "flaggems":
            serve += ["-e", "VLLM_FL_USE_FLAGGEMS_ATTN=1"]
        serve += [
            NAME, "vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
            "--served-model-name", "minicpm", "--max-model-len", "131072",
            "--gpu-memory-utilization", "0.85",
            "--compilation-config", json.dumps(COMPILATION_CONFIG),
        ]
        print("SERVE", variant, serve, flush=True)
        with (directory / "serve.log").open("w") as log:
            service = subprocess.Popen(serve, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(service)
            print("SERVICE READY", variant, flush=True)
            with (directory / "probe.log").open("w") as probe_log:
                probe = subprocess.run(
                    ["docker", "exec", NAME, "python3",
                     "/workspace/benchmark_results/decode_ab_20261009/run_cases.py",
                     variant],
                    stdout=probe_log, stderr=subprocess.STDOUT, timeout=1800,
                )
            print("PROBE EXIT", variant, probe.returncode, flush=True)
            if probe.returncode != 0:
                raise RuntimeError(f"{variant} probes failed: {probe.returncode}")
    finally:
        if started_container:
            stop = subprocess.run(["docker", "stop", NAME], capture_output=True,
                                  text=True, timeout=45)
            (directory / "container_stop.log").write_text(
                stop.stdout + stop.stderr + f"exit_code={stop.returncode}\n"
            )
            time.sleep(8)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    code = 1
    try:
        for variant in ("triton", "flaggems"):
            print("START", variant, flush=True)
            run_variant(variant)
            print("DONE", variant, flush=True)
        (ROOT / "finished.ok").write_text("both attention backends completed\n")
        code = 0
    except Exception as exc:
        print("ERROR", repr(exc), flush=True)
    finally:
        (ROOT / "runner.exit").write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
