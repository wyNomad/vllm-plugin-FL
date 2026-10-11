#!/usr/bin/env python3
"""Compare KV-cache write kernels on the same Iluvatar device and tensors."""

import json
import time
from pathlib import Path

import torch
from flag_gems import reshape_and_cache_flash as flaggems_update
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash as triton_update,
)


ROOT = Path("/workspace/benchmark_results/hybrid_diagnostics_20261010")
BLOCKS = 1024
BLOCK_SIZE = 16
KV_HEADS = 2
HEAD_DIM = 128
DTYPE = torch.bfloat16
DEVICE = "cuda:0"


def measure(fn, iterations=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_begin = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return {
        "gpu_ms_per_call": begin.elapsed_time(end) / iterations,
        "wall_ms_per_call": (time.perf_counter() - wall_begin) * 1000 / iterations,
    }


def case(num_tokens):
    torch.manual_seed(42 + num_tokens)
    key = torch.randn(num_tokens, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    value = torch.randn_like(key)
    slots = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    scale = torch.ones((), device=DEVICE, dtype=torch.float32)
    gems_cache = torch.zeros(
        (2, BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM), device=DEVICE, dtype=DTYPE
    )
    view_cache = torch.zeros_like(gems_cache)
    native_cache = torch.zeros(
        (BLOCKS, 2, BLOCK_SIZE, KV_HEADS, HEAD_DIM), device=DEVICE, dtype=DTYPE
    )

    def gems():
        k_cache, v_cache = gems_cache.unbind(0)
        flaggems_update(key, value, k_cache, v_cache, slots, "auto", scale, scale)

    def triton_view():
        k_cache, v_cache = view_cache.transpose(0, 1).unbind(1)
        triton_update(key, value, k_cache, v_cache, slots, "auto", scale, scale)

    def triton_native():
        k_cache, v_cache = native_cache.unbind(1)
        triton_update(key, value, k_cache, v_cache, slots, "auto", scale, scale)

    timings = {}
    for name, fn in (
        ("flaggems_layout_write", gems),
        ("triton_on_flaggems_layout", triton_view),
        ("triton_native_layout_write", triton_native),
    ):
        timings[name] = measure(fn)
        print(num_tokens, name, timings[name], flush=True)

    if not torch.equal(gems_cache, view_cache):
        raise RuntimeError("FlagGems and Triton view writes differ")
    if not torch.equal(gems_cache, native_cache.transpose(0, 1)):
        raise RuntimeError("FlagGems and Triton native writes differ")
    return {"num_tokens": num_tokens, "timings": timings, "identical_cache": True}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    result = {
        "device": torch.cuda.get_device_name(0),
        "dtype": str(DTYPE),
        "blocks": BLOCKS,
        "block_size": BLOCK_SIZE,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "cases": [case(64), case(2048)],
    }
    (ROOT / "kv_kernel_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SAVED", ROOT / "kv_kernel_results.json", flush=True)


if __name__ == "__main__":
    main()
