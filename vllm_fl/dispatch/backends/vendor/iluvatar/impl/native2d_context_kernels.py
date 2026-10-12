# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""Pinned attention kernels with per-sequence GPU routing for graph replay.

Adapted from vLLM 0.24, original SHA256
bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0.
Only two added constexpr arguments and early-return guards differ. A captured
short-context graph can therefore replay with long or mixed contexts without
reading GPU lengths back to the host or changing the attention formula.
"""
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_unified_attention import (
    _cast_kv_tile, _load_q_td, _load_kv_tile_td, _store_output_td, float8_info,
)
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    find_seq_idx,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)
from vllm.v1.kv_cache_interface import KVQuantMode


@triton.jit
def kernel_unified_attention(
    # Output destination for the 2D path.  In 3D mode per-segment partials
    # go to the ``segm_*`` tensors (see bottom of signature) and
    # ``output_ptr`` is unused (callers may pass any non-null pointer).
    output_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # Scalars
    scale,
    q_scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_CAUSAL: tl.constexpr,  # bool
    USE_PER_SEQ_CAUSAL: tl.constexpr,  # bool
    per_seq_causal_ptr,  # [num_seqs] bool, or None
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
    # per-segment partials, finalized by ``reduce_segments``.
    IS_3D: tl.constexpr,
    # Parameters below default to None so Triton can skip materialising them
    # on call sites where the corresponding constexpr branch is dead.
    # Credit: @quinnlp identified this as a perf regression source in
    # intel/intel-xpu-backend-for-triton#6758 (review comment r3204641104).
    # Per-segment outputs: used in 3D mode; unused in 2D (IS_3D=False).
    segm_output_ptr=None,
    segm_max_ptr=None,
    segm_expsum_ptr=None,
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    stride_ks_blk: tl.int64 = None,
    stride_ks_slot: tl.int64 = None,
    stride_ks_head: tl.int64 = None,
    stride_vs_blk: tl.int64 = None,
    stride_vs_slot: tl.int64 = None,
    stride_vs_head: tl.int64 = None,
    # KV cache quantization mode handled inside this kernel via constexpr
    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
    # FP8_PER_TOKEN_HEAD (3).
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ROUTE_THRESHOLD: tl.constexpr = 8192,
    ROUTE_LONG: tl.constexpr = False,
    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
    # chunked masking (used by Gemma3 block-local layers); takes precedence
    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    # Tensor-descriptor load/store for HW 2D block reads on Intel Xe2/Xe3.
    # ``USE_TD`` gates KV tile loads; ``USE_TD_QO`` separately gates Q/output
    # (see ``unified_attention`` wrapper for the gating rules).
    USE_TD: tl.constexpr = False,
    USE_TD_QO: tl.constexpr = False,
    Q_IS_FP8: tl.constexpr = False,
):
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2
    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    # Device-side decision is re-evaluated on every CUDA Graph replay.
    if (seq_len >= ROUTE_THRESHOLD) != ROUTE_LONG:
        return

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    # Number of valid query rows in this block (used by TD descriptor
    # shapes, but always computed so the variable stays in scope).
    q_block_local_len = tl.minimum(
        BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    if USE_TD_QO:
        Q = _load_q_td(
            query_ptr,
            q_block_local_len,
            query_stride_0,
            query_stride_1,
            cur_batch_in_all_start_index,
            q_block_local_idx,
            kv_head_idx,
            num_queries_per_kv,
            BLOCK_Q,
            BLOCK_M,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    score_scale = scale
    value_scale = 1.0
    if USE_FP8_Q_DESCALE:
        score_scale = scale * tl.load(q_scale) * tl.load(k_scale)
        value_scale = tl.load(v_scale)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        USE_CAUSAL,
        USE_PER_SEQ_CAUSAL,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )

    # iterate through tiles (now limited to the sliding window range)
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        if USE_TD:
            # All TILE_SIZE slots within a single KV tile map to one
            # physical block (guaranteed by ``BLOCK_SIZE % TILE_SIZE == 0``
            # from the static_assert above), so load the block index as
            # a scalar instead of a broadcast reduction.
            offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE
            physical_block_scalar = tl.load(
                block_tables_ptr + block_table_offset + (j * TILE_SIZE) // BLOCK_SIZE
            ).to(tl.int64)
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = _load_kv_tile_td(
                key_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            ).T
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = _load_kv_tile_td(
                value_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            USE_CAUSAL,
            USE_PER_SEQ_CAUSAL,
            per_seq_causal_ptr,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (score_scale * k_token_head_scales[None, :])
        else:
            S += score_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            dist = context_len + qpos_lo - seq_offset[:, None]
            if USE_PER_SEQ_CAUSAL:
                is_causal_seq = tl.load(per_seq_causal_ptr + seq_idx)
                sw_mask_v = tl.where(
                    is_causal_seq,
                    dist < SLIDING_WINDOW,
                    (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW),
                )
            elif USE_CAUSAL:
                sw_mask_v = dist < SLIDING_WINDOW
            else:
                sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
            V = tl.where(sw_mask_v, V, 0.0)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        if USE_FP8_Q_DESCALE:
            acc *= value_scale
        # Store per-segment partials; finalized by ``reduce_segments``.
        if USE_TD_QO:
            # 3D target: segm_output[token, head, segm_idx, :].  Advance
            # the base to the correct (token-start, head-start, segm)
            # slice; strides step between tokens / heads of the flattened
            # (T, H, SEGS, PAD) layout.
            segm_base = (
                segm_output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q).to(
                    tl.int64
                )
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + (kv_head_idx * num_queries_per_kv)
                * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
            )
            _store_output_td(
                segm_base,
                acc,
                q_block_local_len,
                num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            segm_output_offset = (
                query_offset_0[:, None].to(tl.int64)
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
                + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
            )
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8_Q_DESCALE:
            acc *= value_scale
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        if USE_TD_QO:
            # 2D target: flat output[token, head, :].  Strides come
            # straight from the caller (``output_stride_0`` per token,
            # ``output_stride_1`` per head).
            output_base = (
                output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q)
                * output_stride_0
                + (kv_head_idx * num_queries_per_kv) * output_stride_1
            )
            _store_output_td(
                output_base,
                acc,
                q_block_local_len,
                output_stride_0,
                output_stride_1,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            output_offset = (
                query_offset_0[:, None] * output_stride_0
                + query_offset_1[:, None] * output_stride_1
                + offs_d[None, :]
            )
            tl.store(
                output_ptr + output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ROUTE_THRESHOLD: tl.constexpr = 8192,
    ROUTE_LONG: tl.constexpr = False,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if (seq_len >= ROUTE_THRESHOLD) != ROUTE_LONG:
        return

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)

