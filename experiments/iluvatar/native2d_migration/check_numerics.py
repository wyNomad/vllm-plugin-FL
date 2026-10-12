#!/usr/bin/env python3
"""GPU checks against the unchanged legacy launch policy and dense PyTorch."""

import json
import os
from pathlib import Path

import torch

import legacy_reference_ops
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.native2d_attention import (
    Native2DAttentionBackend,
)
from vllm_fl.dispatch.backends.vendor.iluvatar.impl import native2d_ops


def run(dtype, sequences, query_len, context_len, head_size=128, gqa=8):
    device = 'cuda'
    kv_heads, block_size, segments = 2, 16, 16
    heads = kv_heads * gqa
    blocks_per_sequence = (context_len + block_size - 1) // block_size
    cache = torch.randn(sequences * blocks_per_sequence, 2, block_size,
                        kv_heads, head_size, device=device, dtype=dtype) * 0.1
    key, value = cache.unbind(1)
    query = torch.randn(sequences * query_len, heads, head_size,
                        device=device, dtype=dtype) * 0.1
    table = torch.arange(sequences * blocks_per_sequence, device=device,
                         dtype=torch.int32).reshape(sequences, blocks_per_sequence)
    starts = torch.arange(sequences + 1, device=device, dtype=torch.int32) * query_len
    lengths = torch.full((sequences,), context_len, device=device, dtype=torch.int32)
    intermediate = torch.empty(sequences, heads, segments, head_size,
                               device=device, dtype=torch.float32)
    maxima = torch.empty(sequences, heads, segments, device=device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    outputs = [torch.empty_like(query), torch.empty_like(query)]
    params = dict(q=query, k=key, v=value, cu_seqlens_q=starts,
                  max_seqlen_q=query_len, seqused_k=lengths, max_seqlen_k=context_len,
                  softmax_scale=head_size**-0.5, causal=True, window_size=(-1, -1),
                  block_table=table, softcap=0, q_descale=None, k_descale=None, v_descale=None,
                  seq_threshold_3D=64, num_par_softmax_segments=segments,
                  softmax_segm_output=intermediate, softmax_segm_max=maxima,
                  softmax_segm_expsum=sums)
    legacy_reference_ops.unified_attention(out=outputs[0], **params)
    native2d_ops.unified_attention(out=outputs[1], **params)
    torch.cuda.synchronize()
    if not torch.equal(outputs[0], outputs[1]):
        raise RuntimeError('migration changed outputs versus the exact first-version policy')
    if not torch.isfinite(outputs[1]).all():
        raise RuntimeError('non-finite attention output')
    # Independently check the first sequence. The query positions are the final
    # query_len tokens of its context; this exercises causal masking and GQA.
    k_dense = key[:blocks_per_sequence].reshape(-1, kv_heads, head_size)[:context_len]
    v_dense = value[:blocks_per_sequence].reshape(-1, kv_heads, head_size)[:context_len]
    k_dense = k_dense.repeat_interleave(gqa, dim=1).transpose(0, 1).float()
    v_dense = v_dense.repeat_interleave(gqa, dim=1).transpose(0, 1).float()
    q_dense = query[:query_len].transpose(0, 1).float()
    score = torch.matmul(q_dense, k_dense.transpose(1, 2)) * head_size**-0.5
    causal = torch.arange(context_len, device=device)[None, :] <= (
        context_len - query_len + torch.arange(query_len, device=device)[:, None])
    score.masked_fill_(~causal, float('-inf'))
    reference = torch.matmul(score.softmax(-1), v_dense).transpose(0, 1)
    actual = outputs[1][:query_len].float()
    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-2)
    return {'dtype': str(dtype), 'sequences': sequences, 'query_len': query_len,
            'context_len': context_len, 'head_size': head_size, 'gqa': gqa,
            'legacy_bitwise_equal': True,
            'dense_max_abs_error': (actual - reference).abs().max().item()}


def main():
    os.environ['VLLM_FL_NATIVE2D_PROBE'] = '1'
    os.environ['VLLM_FL_ILUVATAR_NATIVE2D'] = '1'
    torch.manual_seed(824)
    cases = [(1, 16, 128), (8, 1, 256), (16, 1, 4096),
             (32, 1, 16384), (64, 1, 16384)]
    rows = [run(dtype, *shape) for dtype in (torch.float16, torch.bfloat16) for shape in cases]
    rows.append(run(torch.bfloat16, 8, 1, 256, head_size=64, gqa=4))
    assert Native2DAttentionBackend.forward_includes_kv_cache_update is False
    Path('numerics.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
