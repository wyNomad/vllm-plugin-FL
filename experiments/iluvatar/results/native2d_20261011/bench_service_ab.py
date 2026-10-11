#!/usr/bin/env python3
"""Run repeatable small service loads against one isolated attention variant."""

import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


ROOT = Path("/workspace/benchmark_results/native2d_20261011")
CASES = [
    ("warmup", 4096, 32, 1, 701),
    ("decode_4k", 4096, 256, 64, 702),
    ("long_16k", 16384, 128, 32, 703),
]
METRICS = {
    "cached_tokens": "vllm:prompt_tokens_cached_total",
    "preemptions": "vllm:num_preemptions_total",
}


def scrape():
    with urlopen("http://127.0.0.1:9031/metrics", timeout=10) as response:
        body = response.read().decode()
    result = {}
    for label, key in METRICS.items():
        values = re.findall(r"^" + re.escape(key) +
                            r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", body, re.M)
        result[label] = sum(map(float, values)) if values else None
    return result


def extract(body, label):
    match = re.search(r"^" + re.escape(label) + r":\s+([0-9.]+)", body, re.M)
    return float(match.group(1)) if match else None


def run_case(variant, spec):
    name, input_len, output_len, count, seed = spec
    command = [
        "vllm", "bench", "serve", "--backend", "vllm", "--model", "minicpm",
        "--tokenizer", "/workspace/MiniCPM5-2B", "--endpoint", "/v1/completions",
        "--host", "localhost", "--port", "9031", "--dataset-name", "random",
        "--ignore-eos", "--temperature", "0", "--random-input-len", str(input_len),
        "--random-output-len", str(output_len), "--max-concurrency", str(count),
        "--num-prompts", str(count), "--seed", str(seed), "--disable-tqdm",
    ]
    before = scrape()
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=1800, check=False)
    directory = ROOT / variant
    (directory / f"{name}.log").write_text(
        "COMMAND: " + " ".join(command) + "\n" + result.stdout + result.stderr
    )
    after = scrape()
    row = {
        "variant": variant, "case": name, "input_tokens": input_len,
        "output_tokens": output_len, "concurrency": count, "seed": seed,
        "exit_code": result.returncode,
        "successful_requests": extract(result.stdout, "Successful requests"),
        "failed_requests": extract(result.stdout, "Failed requests"),
        "total_tokens_per_s": extract(result.stdout, "Total token throughput (tok/s)"),
        "output_tokens_per_s": extract(result.stdout, "Output token throughput (tok/s)"),
        "mean_ttft_ms": extract(result.stdout, "Mean TTFT (ms)"),
        "mean_tpot_ms": extract(result.stdout, "Mean TPOT (ms)"),
        "cached_tokens_delta": None if before["cached_tokens"] is None or
            after["cached_tokens"] is None else
            after["cached_tokens"] - before["cached_tokens"],
        "preemptions_delta": None if before["preemptions"] is None or
            after["preemptions"] is None else
            after["preemptions"] - before["preemptions"],
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if (result.returncode != 0 or row["successful_requests"] != count
            or row["failed_requests"] != 0):
        raise RuntimeError(f"benchmark failed: {variant}/{name}")
    return row


def main():
    variant = sys.argv[1]
    directory = ROOT / variant
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in CASES:
        rows.append(run_case(variant, spec))
        (directory / "summary.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )
    (directory / "finished.ok").write_text("all requests succeeded\n")


if __name__ == "__main__":
    main()
