#!/usr/bin/env python3
"""Run the competition Math500 Level 3 eval with hybrid attention."""

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/workspace/wyNomad/results/hybrid_accuracy_20261009")
NAME = "wynomad-minicpm-s2-gpu0"
WORK_DIR = "/workspace/evalscope-datasets/level3/hybrid_accuracy_20261009"
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
        result = subprocess.run(
            probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=8, check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError("vLLM service did not become healthy")


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    started_container = False
    code = 1
    try:
        launch = subprocess.run(
            ["python3", "/workspace/wyNomad/run_tianshu_container.py",
             "--detach", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=45, check=False,
        )
        print("CONTAINER", launch.stdout, flush=True)
        if launch.returncode != 0:
            raise RuntimeError(f"container launch exited {launch.returncode}")
        started_container = True

        serve = [
            "docker", "exec", "-e", "VLLM_FL_USE_HYBRID_ATTN=1", NAME,
            "vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
            "--served-model-name", "minicpm", "--max-model-len", "131072",
            "--gpu-memory-utilization", "0.85",
            "--compilation-config", json.dumps(COMPILATION_CONFIG),
        ]
        print("SERVE", serve, flush=True)
        with (ROOT / "serve.log").open("w") as serve_log:
            service = subprocess.Popen(
                serve, stdout=serve_log, stderr=subprocess.STDOUT,
            )
            wait_ready(service)
            print("SERVICE READY", flush=True)

            dataset_args = {
                "math_500": {
                    "dataset_id": "/workspace/evalscope-datasets/math_500",
                    "subset_list": ["Level 3"],
                }
            }
            generation_config = {
                "max_tokens": 32768,
                "top_p": 0.95,
                "temperature": 1.0,
            }
            evaluation = [
                "docker", "exec", NAME, "evalscope", "eval",
                "--model", "minicpm", "--datasets", "math_500",
                "--dataset-args", json.dumps(dataset_args),
                "--eval-type", "openai_api",
                "--api-url", "http://127.0.0.1:9031/v1/chat/completions",
                "--api-key", "EMPTY", "--eval-batch-size", "8",
                "--generation-config", json.dumps(generation_config),
                "--timeout", "3600", "--seed", "42", "--ignore-errors",
                "--work-dir", WORK_DIR,
            ]
            (ROOT / "evaluation_command.json").write_text(
                json.dumps(evaluation, indent=2) + "\n"
            )
            print("EVALUATION START", flush=True)
            with (ROOT / "eval.log").open("w") as eval_log:
                completed = subprocess.run(
                    evaluation, stdout=eval_log, stderr=subprocess.STDOUT,
                    timeout=7200, check=False,
                )
            print("EVALUATION EXIT", completed.returncode, flush=True)
            if completed.returncode != 0:
                raise RuntimeError("EvalScope failed")
            (ROOT / "evaluation_succeeded.ok").write_text("evalscope exited 0\n")
        code = 0
    except Exception as exc:
        print("ERROR", repr(exc), flush=True)
    finally:
        if started_container:
            try:
                stop = subprocess.run(
                    ["docker", "stop", NAME], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, timeout=120, check=False,
                )
                (ROOT / "container_stop.log").write_text(
                    stop.stdout + f"exit_code={stop.returncode}\n"
                )
            except subprocess.TimeoutExpired:
                (ROOT / "container_stop.log").write_text(
                    "docker stop timed out after 120s; inspect container state\n"
                )
                code = 1
        (ROOT / "runner.exit").write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(run())
