#!/usr/bin/env python3
"""Refresh the original Triton short-load reference on the same GPU 1."""

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/workspace/wyNomad/results/native2d_20261011")
NAME = "wynomad-minicpm-s2-gpu1"
VARIANTS = (
    ("triton_refresh", ()),
)
COMPILATION = {"cudagraph_mode": "FULL_DECODE_ONLY",
               "cudagraph_num_of_warmups": 0}


def container_exists():
    result = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^{NAME}$",
                             "--format", "{{.Names}}"], capture_output=True,
                            text=True, timeout=15, check=True)
    return NAME in result.stdout.splitlines()


def wait_ready(service):
    probe = ["docker", "exec", NAME, "python3", "-c",
             "import urllib.request; urllib.request.urlopen("
             "'http://127.0.0.1:9031/health', timeout=3)"]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if service.poll() is not None:
            raise RuntimeError(f"vLLM exited early: {service.returncode}")
        ready = subprocess.run(probe, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=8)
        if ready.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError("vLLM service did not become healthy")


def stop_own_container(directory):
    try:
        result = subprocess.run(["docker", "stop", "--time", "20", NAME],
                                capture_output=True, text=True, timeout=90)
        (directory / "container_stop.log").write_text(
            result.stdout + result.stderr + f"exit_code={result.returncode}\n")
    except subprocess.TimeoutExpired:
        (directory / "container_stop.log").write_text("docker stop timed out\n")
    for _ in range(45):
        if not container_exists():
            return
        time.sleep(2)
    kill = subprocess.run(["docker", "kill", NAME], capture_output=True,
                          text=True, timeout=30)
    (directory / "container_kill.log").write_text(
        kill.stdout + kill.stderr + f"exit_code={kill.returncode}\n")
    for _ in range(15):
        if not container_exists():
            return
        time.sleep(2)
    raise RuntimeError("our container still exists after stop; aborting sequence")


def run_variant(variant, environment):
    directory = ROOT / variant
    directory.mkdir(parents=True, exist_ok=True)
    if container_exists():
        raise RuntimeError(f"refusing to reuse existing {NAME}")
    launch = subprocess.run(["python3", "/workspace/wyNomad/run_tianshu_container.py",
                             "--detach", "1"], capture_output=True,
                            text=True, timeout=60)
    (directory / "container_launch.log").write_text(
        launch.stdout + launch.stderr + f"exit_code={launch.returncode}\n")
    if launch.returncode != 0:
        raise RuntimeError(f"container launch failed: {variant}")
    try:
        serve = ["docker", "exec"]
        for item in environment:
            serve.extend(["-e", item])
        serve.extend([NAME, "vllm", "serve", "/workspace/MiniCPM5-2B",
                      "--port", "9031", "--served-model-name", "minicpm",
                      "--max-model-len", "131072", "--gpu-memory-utilization",
                      "0.85", "--compilation-config", json.dumps(COMPILATION)])
        (directory / "serve_command.json").write_text(json.dumps(serve, indent=2) + "\n")
        with (directory / "serve.log").open("w") as log:
            service = subprocess.Popen(serve, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(service)
            print("SERVICE READY", variant, flush=True)
            benchmark = ["docker", "exec", NAME, "python3",
                         "/workspace/benchmark_results/native2d_20261011/bench_service_ab.py",
                         variant]
            with (directory / "benchmark.log").open("w") as bench_log:
                result = subprocess.run(benchmark, stdout=bench_log,
                                        stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode != 0:
                raise RuntimeError(f"benchmark failed: {variant}; see benchmark.log")
    finally:
        stop_own_container(directory)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    code = 1
    try:
        for variant, environment in VARIANTS:
            print("START", variant, flush=True)
            run_variant(variant, environment)
            print("DONE", variant, flush=True)
        (ROOT / "baseline_short_finished.ok").write_text("reference passed\n")
        code = 0
    except Exception as exc:
        print("ERROR", repr(exc), flush=True)
    finally:
        (ROOT / "baseline_short.exit").write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
