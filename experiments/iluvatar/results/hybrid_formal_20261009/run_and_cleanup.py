#!/usr/bin/env python3
"""Run the unmodified competition throughput script for hybrid attention."""

import csv
import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/workspace/wyNomad/results/hybrid_formal_20261009")
NAME = "wynomad-minicpm-s2-gpu0"
CASES = [[4096, 1024, 64, 256], [16384, 1024, 64, 128]]
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


def validate_results():
    output = ROOT / "benchmark_results"
    raw = list(output.glob("raw_runs_*.csv"))
    summaries = list(output.glob("summary_*.csv"))
    if len(raw) != 1 or len(summaries) != 1:
        raise RuntimeError("expected one raw CSV and one summary CSV")
    with raw[0].open(newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    with summaries[0].open(newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    if len(raw_rows) != 8 or len(summary_rows) != 2:
        raise RuntimeError(
            f"incomplete official benchmark: {len(raw_rows)} raw, "
            f"{len(summary_rows)} summary rows"
        )
    for row in raw_rows:
        if row["Run Status"] != "SUCCESS" or int(float(row["Successful Requests"])) != int(row["Num Prompts"]):
            raise RuntimeError("official benchmark includes a failed run")
    return raw[0], summaries[0]


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

            benchmark = [
                "docker", "exec", "-w",
                "/workspace/benchmark_results/hybrid_formal_20261009",
                NAME, "python3",
                "/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py",
                "--model", "/workspace/MiniCPM5-2B",
                "--served-model-name", "minicpm", "--port", "9031",
                "--test-cases", json.dumps(CASES),
            ]
            (ROOT / "benchmark_command.json").write_text(
                json.dumps(benchmark, indent=2) + "\n"
            )
            print("BENCHMARK START", flush=True)
            with (ROOT / "bench.log").open("w") as bench_log:
                result = subprocess.run(
                    benchmark, stdout=bench_log, stderr=subprocess.STDOUT,
                    timeout=36000, check=False,
                )
            print("BENCHMARK EXIT", result.returncode, flush=True)
            if result.returncode != 0:
                raise RuntimeError("competition benchmark script failed")
            raw, summary = validate_results()
            print("RESULTS", raw, summary, flush=True)
            (ROOT / "benchmark_succeeded.ok").write_text(
                "all eight runs succeeded; two score rows saved\n"
            )
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
