#!/usr/bin/env python3
"""Collect bounded, non-scoring vLLM operator traces inside our container."""

import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path("/workspace/benchmark_results/profile_16k_20261009")
CASES = [
    ("prefill_16k", 16384, 1, 301),
    ("prefill_and_decode_16k", 16384, 16, 302),
]


def bench_command(input_tokens, output_tokens, seed):
    return [
        "vllm", "bench", "serve", "--backend", "vllm", "--model", "minicpm",
        "--tokenizer", "/workspace/MiniCPM5-2B", "--endpoint", "/v1/completions",
        "--host", "localhost", "--port", "9031", "--dataset-name", "random",
        "--ignore-eos", "--random-input-len", str(input_tokens),
        "--random-output-len", str(output_tokens), "--max-concurrency", "1",
        "--num-prompts", "1", "--seed", str(seed), "--disable-tqdm",
        "--profile",
    ]


def run_case(name, input_tokens, output_tokens, seed):
    log = ROOT / (name + ".log")
    command = bench_command(input_tokens, output_tokens, seed)
    started = time.monotonic()
    print("START", name, flush=True)
    with log.open("w") as file:
        file.write("COMMAND: " + " ".join(command) + "\n")
        file.flush()
        result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT,
                                timeout=300, check=False)
    case_dir = ROOT / name
    case_dir.mkdir(exist_ok=True)
    for path in (ROOT / "traces").iterdir():
        if path.is_file():
            shutil.move(str(path), str(case_dir / path.name))
    print("DONE", name, "exit", result.returncode,
          "seconds", round(time.monotonic() - started, 2), flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{name} benchmark exited {result.returncode}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        run_case(*case)
    (ROOT / "finished.ok").write_text("both profile cases completed\n")


if __name__ == "__main__":
    main()
