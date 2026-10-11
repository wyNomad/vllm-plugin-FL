#!/usr/bin/env python3
"""Check native-layout hybrid correctness, then run the same short A/B loads."""

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/workspace/wyNomad/results/hybrid_native_20261011")
NAME = "wynomad-minicpm-s2-gpu1"
DIAGNOSTICS = "/workspace/benchmark_results/hybrid_diagnostics_20261010"
COMPILATION = {"cudagraph_mode": "FULL_DECODE_ONLY",
               "cudagraph_num_of_warmups": 0}


def exists():
    proc = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^{NAME}$",
                           "--format", "{{.Names}}"], capture_output=True,
                          text=True, timeout=15, check=True)
    return NAME in proc.stdout.splitlines()


def ready(service):
    probe = ["docker", "exec", NAME, "python3", "-c",
             "import urllib.request; urllib.request.urlopen("
             "'http://127.0.0.1:9031/health', timeout=3)"]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if service.poll() is not None:
            raise RuntimeError(f"vLLM exited early: {service.returncode}")
        proc = subprocess.run(probe, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=8)
        if proc.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError("service did not become healthy")


def stop():
    try:
        proc = subprocess.run(["docker", "stop", "--time", "20", NAME],
                              capture_output=True, text=True, timeout=90)
        (ROOT / "container_stop.log").write_text(
            proc.stdout + proc.stderr + f"exit_code={proc.returncode}\n")
    except subprocess.TimeoutExpired:
        (ROOT / "container_stop.log").write_text("docker stop timed out\n")
    for _ in range(15):
        if not exists():
            return
        time.sleep(2)
    raise RuntimeError("our container still exists after stop")


def run_correctness(name, client):
    command = ["docker", "exec", NAME, "python3", client]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=360)
    (ROOT / f"{name}.stdout").write_text(proc.stdout)
    (ROOT / f"{name}.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed: exit {proc.returncode}")
    response = json.loads(proc.stdout)
    text = response["choices"][0]["message"]["content"]
    if name == "short_math":
        if "117" not in text or "13" not in text:
            raise RuntimeError("short math response lacks expected calculation")
    else:
        tokens = response["usage"]["prompt_tokens"]
        if tokens < 16000 or not text.strip().endswith("4"):
            raise RuntimeError("16k response has wrong length or answer")
    print("CORRECTNESS", name, response["usage"], flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    code = 1
    started = False
    try:
        if exists():
            raise RuntimeError(f"refusing to reuse existing {NAME}")
        source_tree = subprocess.run(
            ["git", "-C", "/workspace/wyNomad/vllm-plugin-FL",
             "rev-parse", "HEAD^{tree}"], capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip()
        (ROOT / "source_tree.txt").write_text(source_tree + "\n")
        launch = subprocess.run(
            ["python3", "/workspace/wyNomad/run_tianshu_container.py",
             "--detach", "1"], capture_output=True, text=True, timeout=60,
        )
        (ROOT / "container_launch.log").write_text(
            launch.stdout + launch.stderr + f"exit_code={launch.returncode}\n")
        if launch.returncode != 0:
            raise RuntimeError("container launch failed")
        started = True
        serve = ["docker", "exec", "-e", "VLLM_FL_USE_HYBRID_ATTN=1",
                 "-e", "VLLM_FL_HYBRID_NATIVE_TRITON_LAYOUT=1",
                 "-e", "VLLM_FL_HYBRID_TRITON_KV_UPDATE=1", NAME,
                 "vllm", "serve", "/workspace/MiniCPM5-2B", "--port", "9031",
                 "--served-model-name", "minicpm", "--max-model-len", "131072",
                 "--gpu-memory-utilization", "0.85", "--compilation-config",
                 json.dumps(COMPILATION)]
        (ROOT / "serve_command.json").write_text(json.dumps(serve, indent=2) + "\n")
        with (ROOT / "serve.log").open("w") as log:
            service = subprocess.Popen(serve, stdout=log, stderr=subprocess.STDOUT)
            ready(service)
            print("SERVICE READY", flush=True)
            run_correctness("short_math",
                            "/workspace/benchmark_results/hybrid_correctness_20261009/client.py")
            run_correctness("long_16k_math",
                            "/workspace/benchmark_results/hybrid_correctness_20261009/long_client.py")
            (ROOT / "correctness_passed.ok").write_text(
                "short and 16k deterministic requests passed\n")
            benchmark = ["docker", "exec", NAME, "python3",
                         DIAGNOSTICS + "/bench_service_ab.py", "hybrid_native_tritonk"]
            with (ROOT / "benchmark.log").open("w") as bench_log:
                result = subprocess.run(benchmark, stdout=bench_log,
                                        stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode != 0:
                raise RuntimeError(f"service A/B failed: exit {result.returncode}")
            (ROOT / "benchmark_passed.ok").write_text("short A/B loads passed\n")
        code = 0
    except Exception as exc:
        print("ERROR", repr(exc), flush=True)
    finally:
        if started:
            try:
                stop()
            except Exception as exc:
                print("CLEANUP ERROR", repr(exc), flush=True)
                code = 1
        (ROOT / "runner.exit").write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
