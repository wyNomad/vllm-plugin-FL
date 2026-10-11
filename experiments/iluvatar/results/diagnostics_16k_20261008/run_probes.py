#!/usr/bin/env python3
"""Non-scoring 16k probes with the competition server configuration unchanged."""

import csv
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


ROOT = Path("/workspace/benchmark_results/diagnostics_16k_20261008")
METRICS_URL = "http://127.0.0.1:9031/metrics"
PHASE_METRICS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "inference": "vllm:request_inference_time_seconds",
}
GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
}
CASES = [
    ("c1_warmup", 1, 1, 1),
    ("c1_repeat", 1, 1, 1),
    ("c16", 16, 16, 1),
    ("c32", 32, 32, 1),
    ("c64", 64, 64, 1),
]
SAMPLE_FIELDS = ["case", "timestamp", *GAUGES]


def metric_value(body, name):
    pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.M)
    values = [float(m.group(1)) for m in pattern.finditer(body)]
    return sum(values) if values else None


def scrape():
    with urlopen(METRICS_URL, timeout=5) as response:
        body = response.read().decode()
    result = {key: metric_value(body, name) for key, name in GAUGES.items()}
    for key, name in PHASE_METRICS.items():
        result[key + "_sum"] = metric_value(body, name + "_sum")
        result[key + "_count"] = metric_value(body, name + "_count")
    return result


def benchmark_command(concurrency, requests, output_tokens):
    return [
        "vllm", "bench", "serve", "--backend", "vllm", "--model", "minicpm",
        "--tokenizer", "/workspace/MiniCPM5-2B", "--endpoint", "/v1/completions",
        "--host", "localhost", "--port", "9031", "--dataset-name", "random",
        "--ignore-eos", "--random-input-len", "16384",
        "--random-output-len", str(output_tokens),
        "--max-concurrency", str(concurrency), "--num-prompts", str(requests),
    ]


def delta(before, after, name):
    a, b = before.get(name), after.get(name)
    return None if a is None or b is None else b - a


def run_case(name, concurrency, requests, output_tokens, sample_writer):
    before = scrape()
    start = time.monotonic()
    command = benchmark_command(concurrency, requests, output_tokens)
    with (ROOT / f"{name}.log").open("w") as logfile:
        logfile.write("COMMAND: " + " ".join(command) + "\n")
        logfile.flush()
        process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                sample = scrape()
                sample_writer.writerow({"case": name, "timestamp": datetime.now(timezone.utc).isoformat(), **sample})
            except Exception as exc:
                logfile.write(f"METRICS SCRAPE ERROR: {exc}\n")
                logfile.flush()
            time.sleep(10)
            if time.monotonic() - start > 3600:
                process.terminate()
                raise TimeoutError(f"{name} exceeded one hour")
    after = scrape()
    phase = {}
    for key in PHASE_METRICS:
        count = delta(before, after, key + "_count")
        total = delta(before, after, key + "_sum")
        phase[key + "_count"] = count
        phase[key + "_mean_s"] = total / count if count and total is not None else None
    return {
        "case": name,
        "concurrency": concurrency,
        "requests": requests,
        "output_tokens": output_tokens,
        "exit_code": process.returncode,
        "duration_s": round(time.monotonic() - start, 2),
        "preemptions": delta(before, after, "preemptions"),
        "prompt_tokens": delta(before, after, "prompt_tokens"),
        **phase,
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    with (ROOT / "metrics_samples.csv").open("w", newline="") as sample_file:
        sample_writer = csv.DictWriter(sample_file, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        sample_writer.writeheader()
        for case in CASES:
            print("START", case, flush=True)
            result = run_case(*case, sample_writer)
            results.append(result)
            (ROOT / "phase_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
            sample_file.flush()
            print("DONE", json.dumps(result, ensure_ascii=False), flush=True)
            if result["exit_code"] != 0:
                raise SystemExit(result["exit_code"])
    (ROOT / "finished.ok").write_text("all diagnostic cases completed\n")


if __name__ == "__main__":
    main()
