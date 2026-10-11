#!/usr/bin/env python3
"""Check whether the CoreX PyTorch build emits CUDA profiler events."""

from pathlib import Path

import torch


OUT = Path("/workspace/benchmark_results/profile_16k_20261009")
OUT.mkdir(parents=True, exist_ok=True)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), flush=True)
x = torch.ones((1024, 1024), device="cuda", dtype=torch.float16)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    for _ in range(3):
        y = x + 1
        torch.cuda.synchronize()
print("sum", y.sum().item(), flush=True)
summary = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
(OUT / "smoke_profiler_summary.txt").write_text(summary)
print(summary, flush=True)
