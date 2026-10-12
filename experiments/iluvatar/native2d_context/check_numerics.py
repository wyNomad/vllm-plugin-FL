#!/usr/bin/env python3
"""Check device routing, graph replay, dense attention, and kernel latency."""
import gc
import json
import os
from pathlib import Path

import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention as baseline
from vllm_fl.dispatch.backends.vendor.iluvatar.impl.native2d_ops import unified_attention


def build(sequences, capacity, dtype):
    heads, kv_heads, head_size, block = 16, 2, 128, 16
    blocks = (capacity + block - 1) // block
    cache = torch.randn(sequences * blocks, 2, block, kv_heads, head_size,
                        device='cuda', dtype=dtype) * .1
    key, value = cache.unbind(1)
    query = torch.randn(sequences, heads, head_size, device='cuda', dtype=dtype) * .1
    lengths = torch.ones(sequences, device='cuda', dtype=torch.int32)
    starts = torch.arange(sequences + 1, device='cuda', dtype=torch.int32)
    table = torch.arange(sequences * blocks, device='cuda', dtype=torch.int32).reshape(sequences, blocks)
    arguments = dict(q=query, k=key, v=value, cu_seqlens_q=starts, max_seqlen_q=1,
                     seqused_k=lengths, max_seqlen_k=capacity, softmax_scale=head_size ** -.5,
                     causal=True, window_size=(-1, -1), block_table=table, softcap=0,
                     q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=64,
                     num_par_softmax_segments=16,
                     softmax_segm_output=torch.empty(64, heads, 16, head_size, device='cuda'),
                     softmax_segm_max=torch.empty(64, heads, 16, device='cuda'),
                     softmax_segm_expsum=torch.empty(64, heads, 16, device='cuda'))
    return arguments


def capture(policy, arguments):
    os.environ['VLLM_FL_ILUVATAR_NATIVE2D_POLICY'] = policy
    output = torch.empty_like(arguments['q'])
    fn = baseline if policy == 'baseline' else unified_attention
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(out=output, **arguments)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(out=output, **arguments)
    return graph, output


def dense(arguments, output, assert_accuracy=True):
    length = int(arguments['seqused_k'][0].item())
    blocks = arguments['block_table'][0]
    key = arguments['k'][blocks].reshape(-1, 2, 128)[:length].repeat_interleave(8, 1).transpose(0, 1).float()
    value = arguments['v'][blocks].reshape(-1, 2, 128)[:length].repeat_interleave(8, 1).transpose(0, 1).float()
    query = arguments['q'][:1].transpose(0, 1).float()
    reference = ((query @ key.transpose(1, 2) * 128 ** -.5).softmax(-1) @ value).transpose(0, 1)
    if assert_accuracy:
        torch.testing.assert_close(output[:1].float(), reference, atol=2e-3, rtol=2e-2)
    return (output[:1].float() - reference).abs().max().item()


def latency(graph, iterations=100):
    for _ in range(10):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def check(sequences, dtype):
    arguments = build(sequences, 16384, dtype)
    graphs = {policy: capture(policy, arguments) for policy in ('baseline', 'native', 'context', 'prefill')}
    rows = []
    for length in (1, 4096, 8191, 8192, 16384, 'mixed'):
        lengths = arguments['seqused_k']
        if length == 'mixed':
            lengths.copy_(torch.tensor(([4096, 8191, 8192, 16384] * ((sequences + 3) // 4))[:sequences],
                                      device='cuda', dtype=torch.int32))
        else:
            lengths.fill_(length)
        for graph, output in graphs.values():
            output.fill_(float('nan'))
            graph.replay()
        torch.cuda.synchronize()
        expected = torch.where((lengths >= 8192)[:, None, None], graphs['native'][1], graphs['baseline'][1])
        actual = graphs['context'][1]
        assert torch.isfinite(actual).all(), 'routing skipped a live output'
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(graphs['prefill'][1], graphs['baseline'][1], atol=2e-3, rtol=2e-2)
        # A dummy length=1 discrepancy appeared above the 3D batch threshold.
        # Verify whether an unmodified eager upstream call reproduces it
        # separately; all real 4k/16k/boundary/mixed contexts must pass dense.
        capture_dummy = sequences > 64 and length == 1
        row = dict(sequences=sequences, dtype=str(dtype), context=length,
                   replay_max_abs_error=(actual - expected).abs().max().item(),
                   dense_max_abs_error=dense(arguments, actual, not capture_dummy),
                   dense_checked=not capture_dummy)
        if capture_dummy:
            eager = torch.empty_like(actual)
            baseline(out=eager, **arguments)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, eager, atol=2e-3, rtol=2e-2)
            row['upstream_eager_dense_max_abs_error'] = dense(arguments, eager, False)
            row['note'] = 'capture-only dummy length; upstream eager reproduces discrepancy'
        if length in (4096, 16384):
            row['graph_latency_us'] = {policy: latency(pair[0]) for policy, pair in graphs.items()}
        rows.append(row)
        print(json.dumps(row), flush=True)
    del graphs, arguments
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    os.environ['VLLM_FL_ILUVATAR_NATIVE2D'] = '1'
    torch.manual_seed(824)
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for sequences in (16, 64, 96):
            rows.extend(check(sequences, dtype))
    Path('numerics.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('CONTEXT ROUTING GPU NUMERICS AND GRAPH REPLAY PASSED', flush=True)


if __name__ == '__main__':
    main()
