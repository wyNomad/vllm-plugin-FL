#!/usr/bin/env python3
"""Summarize captured GPU kernel durations without double-counting CPU op rows."""

import gzip
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = ("prefill_16k", "prefill_and_decode_16k")


def summarize(events):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for event in events:
        totals[event["name"]] += event["dur"]
        counts[event["name"]] += 1
    total_us = sum(totals.values())
    return {
        "kernel_calls": len(events),
        "gpu_kernel_time_ms": round(total_us / 1000, 3),
        "top_kernels": [
            {"name": name, "calls": counts[name],
             "time_ms": round(duration / 1000, 3),
             "share_percent": round(duration / total_us * 100, 2)}
            for name, duration in sorted(totals.items(), key=lambda x: -x[1])[:15]
        ],
    }


def main():
    output = {}
    for case in CASES:
        traces = list((ROOT / case).glob("rank0.*.pt.trace.json.gz"))
        if len(traces) != 1:
            raise RuntimeError(f"Expected exactly one worker trace for {case}")
        with gzip.open(traces[0], "rt") as file:
            events = json.load(file)["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
        launches = [e for e in events if e.get("name") == "cudaGraphLaunch"]
        case_summary = {"trace": traces[0].name,
                        "cuda_graph_launches": len(launches),
                        "all": summarize(kernels)}
        if launches:
            boundary = min(e["ts"] for e in launches)
            case_summary["before_first_graph_launch"] = summarize(
                [e for e in kernels if e["ts"] < boundary]
            )
            case_summary["from_first_graph_launch"] = summarize(
                [e for e in kernels if e["ts"] >= boundary]
            )
        output[case] = case_summary
    (ROOT / "kernel_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    for case, summary in output.items():
        print(case, summary["all"]["gpu_kernel_time_ms"], "ms")
        print(*summary["all"]["top_kernels"][:3], sep="\n")


if __name__ == "__main__":
    main()
