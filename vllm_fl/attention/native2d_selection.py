# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in Native 2D selection without changing the dispatch order."""

import os


def native2d_backend_path(vendor: str) -> str | None:
    if vendor != "iluvatar" or os.environ.get("VLLM_FL_ILUVATAR_NATIVE2D") != "1":
        return None
    if any(
        os.environ.get(name, "0").lower() in ("1", "true", "yes")
        for name in ("VLLM_FL_USE_HYBRID_ATTN", "VLLM_FL_USE_FLAGGEMS_ATTN")
    ):
        raise ValueError("Select Native 2D or the older attention experiments, not both")
    return (
        "vllm_fl.dispatch.backends.vendor.iluvatar.impl."
        "native2d_attention.Native2DAttentionBackend"
    )
