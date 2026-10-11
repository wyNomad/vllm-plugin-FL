#!/usr/bin/env python3
"""Compare one deterministic Math500-style request across attention backends."""

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path("/workspace/wyNomad/results/hybrid_correctness_20261009")
NAME = "wynomad-minicpm-s2-gpu0"
PROMPT = (
    "What is the distance, in units, between the points $(2, -6)$ and "
    "$(-4, 3)$? Express your answer in simplest radical form.\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def run(mode: str) -> int:
    directory = ROOT / mode
    directory.mkdir(parents=True, exist_ok=True)
    started = False
    status = 1
    try:
        launch = subprocess.run(
            ["python3", "/workspace/wyNomad/run_tianshu_container.py", "--detach", "0"],
            capture_output=True, text=True, timeout=50,
        )
        print("CONTAINER", launch.stdout, launch.stderr, flush=True)
        if launch.returncode:
            raise RuntimeError(f"container launch failed: {launch.returncode}")
        started = True
        serve = ["docker", "exec"]
        env = {
            "flaggems": "VLLM_FL_USE_FLAGGEMS_ATTN=1",
            "flaggems_fixed": "VLLM_FL_USE_FLAGGEMS_ATTN=1",
            "hybrid": "VLLM_FL_USE_HYBRID_ATTN=1",
            "hybrid_fixed": "VLLM_FL_USE_HYBRID_ATTN=1",
            "hybrid_long": "VLLM_FL_USE_HYBRID_ATTN=1",
            "hybrid_force_triton": "VLLM_FL_USE_HYBRID_ATTN=1",
            "hybrid_triton_write": "VLLM_FL_USE_HYBRID_ATTN=1",
            "hybrid_all_triton": "VLLM_FL_USE_HYBRID_ATTN=1",
            "hybrid_trace_alltriton": "VLLM_FL_USE_HYBRID_ATTN=1",
        }.get(mode)
        if env:
            serve.extend(["-e", env])
        if mode in ("hybrid_force_triton", "hybrid_all_triton", "hybrid_trace_alltriton"):
            serve.extend(["-e", "VLLM_FL_HYBRID_FORCE_TRITON=1"])
        if mode in ("hybrid_triton_write", "hybrid_all_triton", "hybrid_trace_alltriton"):
            serve.extend(["-e", "VLLM_FL_HYBRID_TRITON_WRITE=1"])
        if mode == "hybrid_trace_alltriton":
            serve.extend(["-e", "VLLM_FL_HYBRID_TRACE=1"])
        serve.extend([
            NAME, "vllm", "serve", "/workspace/MiniCPM5-2B",
            "--port", "9031", "--served-model-name", "minicpm",
            "--max-model-len", "131072", "--gpu-memory-utilization", "0.85",
            "--compilation-config",
            '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_num_of_warmups":0}',
        ])
        with (directory / "serve.log").open("w") as serve_log:
            service = subprocess.Popen(serve, stdout=serve_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if service.poll() is not None:
                    raise RuntimeError(f"service exited early: {service.returncode}")
                health = subprocess.run(
                    ["docker", "exec", NAME, "python3", "-c",
                     "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9031/health',timeout=3)"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=8, check=False,
                )
                if health.returncode == 0:
                    break
                time.sleep(4)
            else:
                raise TimeoutError("health probe timed out")
            print("SERVICE READY", flush=True)
            client_name = "long_client.py" if mode.endswith("_long") else "client.py"
            result = subprocess.run(
                ["docker", "exec", NAME, "python3",
                 "/workspace/benchmark_results/hybrid_correctness_20261009/" + client_name],
                capture_output=True, text=True, timeout=300,
            )
            (directory / "response.json").write_text(result.stdout)
            (directory / "client.stderr").write_text(result.stderr)
            print("CLIENT", result.returncode, result.stdout[:1200], flush=True)
            if result.returncode:
                raise RuntimeError(f"client failed: {result.returncode}")
            response = json.loads(result.stdout)
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            status = 0
    except Exception as exc:
        print("ERROR", repr(exc), flush=True)
    finally:
        if started:
            stop = subprocess.run(["docker", "stop", NAME], capture_output=True,
                                  text=True, timeout=120, check=False)
            (directory / "container_stop.log").write_text(
                stop.stdout + stop.stderr + f"exit_code={stop.returncode}\n"
            )
        (directory / "runner.exit").write_text(f"{status}\n")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["triton", "triton_long", "flaggems", "flaggems_fixed", "hybrid", "hybrid_fixed", "hybrid_long", "hybrid_force_triton",
                 "hybrid_triton_write", "hybrid_all_triton", "hybrid_trace_alltriton"],
    )
    raise SystemExit(run(parser.parse_args().mode))
