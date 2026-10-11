#!/usr/bin/env python3
"""Summarize the archived baseline's 10-second vLLM engine log samples."""

import csv
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, median


RESULTS = Path(__file__).resolve().parents[1] / "baseline_20261007"
LOG = RESULTS / "logs/serve_baseline_20261007_231136.log"
RAW = RESULTS / "raw_runs_20261007_232244.csv"
PATTERN = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*Engine 000: "
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%"
)


def parse_samples():
    samples = []
    for line in LOG.read_text(errors="replace").splitlines():
        match = PATTERN.search(line)
        if match:
            samples.append(
                (
                    datetime.strptime("2026-" + match[1], "%Y-%m-%d %H:%M:%S"),
                    *map(float, match.groups()[1:]),
                )
            )
    return samples


def split_runs(samples):
    """Close a run after two consecutive idle metric samples."""
    runs, current, idle = [], [], 0
    for sample in samples:
        if sample[3] + sample[4] > 0:
            current.append(sample)
            idle = 0
        elif current:
            idle += 1
            if idle >= 2:
                runs.append(current)
                current, idle = [], 0
    if current:
        runs.append(current)
    return runs


def main():
    runs = split_runs(parse_samples())
    with RAW.open(newline="") as source:
        benchmarks = list(csv.DictReader(source))
    if len(runs) != len(benchmarks):
        raise SystemExit(f"日志分段 {len(runs)} 段，CSV 为 {len(benchmarks)} 轮；请人工核对")

    print("run,case,start,end,samples,mean_running,median_running,mean_waiting,median_waiting,mean_kv_pct,median_kv_pct,pct_waiting,pct_kv_ge_90,pct_kv_ge_95,max_running,max_waiting,max_kv_pct,mean_prompt_tok_s,mean_generation_tok_s,total_tok_s,mean_ttft_ms,mean_tpot_ms")
    for index, (samples, bench) in enumerate(zip(runs, benchmarks), 1):
        n = len(samples)
        avg = lambda column: mean(s[column] for s in samples)
        med = lambda column: median(s[column] for s in samples)
        pct = lambda predicate: 100 * sum(predicate(s) for s in samples) / n
        print(
            f"{index},{bench['Prefill']},{samples[0][0]},{samples[-1][0]},{n},"
            f"{avg(3):.1f},{med(3):.1f},{avg(4):.1f},{med(4):.1f},"
            f"{avg(5):.1f},{med(5):.1f},{pct(lambda s: s[4] > 0):.1f},"
            f"{pct(lambda s: s[5] >= 90):.1f},{pct(lambda s: s[5] >= 95):.1f},"
            f"{max(s[3] for s in samples):.0f},{max(s[4] for s in samples):.0f},"
            f"{max(s[5] for s in samples):.1f},{avg(1):.1f},{avg(2):.1f},"
            f"{bench['Total tok/s']},{bench['Mean TTFT (ms)']},{bench['Mean TPOT (ms)']}"
        )


if __name__ == "__main__":
    main()
