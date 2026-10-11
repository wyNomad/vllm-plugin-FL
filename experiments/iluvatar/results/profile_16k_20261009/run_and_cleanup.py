#!/usr/bin/env python3
"""Launch personal profiling container and always clean it up."""

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/workspace/wyNomad/results/profile_16k_20261009")
NAME = "wynomad-minicpm-s2-gpu0"
PROFILE_CONFIG = {
    "profiler": "torch",
    "torch_profiler_dir": "/workspace/benchmark_results/profile_16k_20261009/traces",
    "torch_profiler_with_stack": False,
    "torch_profiler_record_shapes": False,
    "torch_profiler_with_memory": False,
    "torch_profiler_use_gzip": True,
    "torch_profiler_dump_cuda_time_total": True,
    "active_iterations": 64,
}
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
    deadline = time.monotonic() + 360
    while time.monotonic() < deadline:
        if service.poll() is not None:
            raise RuntimeError(f"vLLM exited early: {service.returncode}")
        ready = subprocess.run(probe, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=8)
        if ready.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError("vLLM service did not become healthy in six minutes")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "traces").mkdir(exist_ok=True)
    started_container = False
    exit_code = 1
    try:
        launch = subprocess.run(
            ["python3", "/workspace/wyNomad/run_tianshu_container.py",
             "--detach", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=45,
        )
        print("CONTAINER", launch.stdout, flush=True)
        if launch.returncode != 0:
            raise RuntimeError(f"container launcher exited {launch.returncode}")
        started_container = True
        serve = [
            "docker", "exec", NAME, "vllm", "serve",
            "/workspace/MiniCPM5-2B", "--port", "9031",
            "--served-model-name", "minicpm", "--max-model-len", "131072",
            "--gpu-memory-utilization", "0.85",
            "--compilation-config", json.dumps(COMPILATION_CONFIG),
            "--profiler-config", json.dumps(PROFILE_CONFIG),
        ]
        print("SERVE", serve, flush=True)
        with (ROOT / "serve.log").open("w") as log:
            service = subprocess.Popen(serve, stdout=log,
                                       stderr=subprocess.STDOUT)
            wait_ready(service)
            print("SERVICE READY", flush=True)
            with (ROOT / "probe.log").open("w") as probe_log:
                probe = subprocess.run(
                    ["docker", "exec", NAME, "python3",
                     "/workspace/benchmark_results/profile_16k_20261009/run_probes.py"],
                    stdout=probe_log, stderr=subprocess.STDOUT, timeout=900,
                )
            print("PROBE EXIT", probe.returncode, flush=True)
            if probe.returncode != 0:
                raise RuntimeError(f"profile probe exited {probe.returncode}")
        exit_code = 0
    except Exception as exc:
        print("ERROR", repr(exc), flush=True)
    finally:
        if started_container:
            stop = subprocess.run(["docker", "stop", NAME], capture_output=True,
                                  text=True, timeout=45)
            (ROOT / "container_stop.log").write_text(
                stop.stdout + stop.stderr + f"exit_code={stop.returncode}\n"
            )
        (ROOT / "runner.exit").write_text(f"{exit_code}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
