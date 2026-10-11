#!/usr/bin/env python3
"""Run identical decode-heavy probes against one attention backend."""

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.request import urlopen


ROOT = Path("/workspace/benchmark_results/decode_ab_20261009")
CASES = [
    ("warmup", 512, 16, 1, 511),
    ("decode", 512, 1024, 8, 512),
]
METRICS = {
    "cached_tokens": "vllm:prompt_tokens_cached_total",
    "preemptions": "vllm:num_preemptions_total",
}


def scrape():
    with urlopen("http://127.0.0.1:9031/metrics", timeout=10) as response:
        text = response.read().decode()
    values = {}
    for key, name in METRICS.items():
        matches = re.findall(r"^" + re.escape(name) +
                             r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", text, re.M)
        values[key] = sum(float(x) for x in matches) if matches else None
    return values


def bench_command(input_tokens, output_tokens, count, seed):
    return [
        "vllm", "bench", "serve", "--backend", "vllm", "--model", "minicpm",
        "--tokenizer", "/workspace/MiniCPM5-2B", "--endpoint", "/v1/completions",
        "--host", "localhost", "--port", "9031", "--dataset-name", "random",
        "--ignore-eos", "--random-input-len", str(input_tokens),
        "--random-output-len", str(output_tokens), "--max-concurrency", str(count),
        "--num-prompts", str(count), "--seed", str(seed), "--disable-tqdm",
    ]


def metric(text, label):
    match = re.search(r"^" + re.escape(label) + r":\s+([0-9.]+)", text, re.M)
    return float(match.group(1)) if match else None


def run_case(variant, name, input_tokens, output_tokens, count, seed):
    directory = ROOT / variant
    directory.mkdir(parents=True, exist_ok=True)
    command = bench_command(input_tokens, output_tokens, count, seed)
    before = scrape()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=900, check=False)
    (directory / f"{name}.log").write_text(
        "COMMAND: " + " ".join(command) + "\n" + result.stdout
    )
    after = scrape()
    output = {
        "variant": variant, "case": name, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "concurrency": count,
        "requests": count, "seed": seed, "exit_code": result.returncode,
        "successful_requests": metric(result.stdout, "Successful requests"),
        "failed_requests": metric(result.stdout, "Failed requests"),
        "total_tokens_per_s": metric(result.stdout, "Total token throughput (tok/s)"),
        "mean_ttft_ms": metric(result.stdout, "Mean TTFT (ms)"),
        "cached_tokens_delta": None if before["cached_tokens"] is None or
            after["cached_tokens"] is None else
            after["cached_tokens"] - before["cached_tokens"],
        "preemptions_delta": None if before["preemptions"] is None or
            after["preemptions"] is None else
            after["preemptions"] - before["preemptions"],
    }
    print("CASE", json.dumps(output, ensure_ascii=False), flush=True)
    if result.returncode != 0 or output["successful_requests"] != count or \
            output["failed_requests"] != 0:
        raise RuntimeError(f"{variant}/{name} failed; inspect {name}.log")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("triton", "flaggems"))
    variant = parser.parse_args().variant
    summaries = []
    for case in CASES:
        summaries.append(run_case(variant, *case))
        (ROOT / variant / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
        )
    (ROOT / variant / "finished.ok").write_text("all cases succeeded\n")


if __name__ == "__main__":
    main()
