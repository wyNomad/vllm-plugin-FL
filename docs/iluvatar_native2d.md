# Iluvatar Native 2D attention

The backend is opt-in and currently targets the pinned BI-V150 / vLLM 0.24
image used for the MiniCPM5-2B experiment. It does not edit installed vLLM files.

```bash
export VLLM_PLUGINS=fl
export VLLM_FL_ILUVATAR_NATIVE2D=1
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

The existing dispatch order remains in place. Both FlagGems and the Iluvatar
vendor backend can return the plugin class when this flag is enabled on
Iluvatar. With the flag unset, their original selections remain unchanged.
Do not combine it with the older FlagGems/hybrid attention experiment flags.
Metadata, graph support, KV layout and the independent KV update are inherited
from vLLM. The decoder forward and launch policy are owned by the plugin.
Non-quantized KV passes no descale; quantized paths retain their original logic.
The backend checks both pinned engine source hashes and fails on incompatible
or previously patched sources. The validated image digest is recorded in the
experiment manifests; this version is not a generic vLLM-version adapter.

## What changes

The attention equation, causal mask and model precision are unchanged.
vLLM's same Triton JIT kernel supports two grids: a 2D grid over query blocks
and KV heads, and a 3D grid that additionally partitions the KV context into
softmax segments. The latter writes partial results and invokes a reducer.

For unquantized head_size=128, GQA=8, non-TD inputs, the first-version policy
uses 2D for prefill/mixed batches or at least 16 sequences. It sets BLOCK_M=128
(BLOCK_Q=16), KV TILE_SIZE=16, 8 warps and 2 stages. Other shapes retain the
original launch policy. This matches the first experimental patch; it does
not yet apply the proposed context-length threshold that may address the
confirmed 4k regression.

Prefill already used 2D in the baseline: its improvement comes from tiling and
launch geometry, not merely renaming a backend. Decode can avoid segment
buffers and the reduction launch. The best trade-off depends on shape and
context length; 2D is not universally faster. It does not free allocated KV
cache memory or use a new quantization method.

## Validation

CPU selection tests cover opt-in, unchanged defaults, other hardware and
conflicting experiment flags. GPU numerical checks compare the plugin launch
policy with the first-version oracle, plus an independent dense PyTorch
reference, including 2D, retained 3D, GQA and long contexts. Service validation
uses fresh personal containers for legacy-patch and plugin candidates with
the same official service arguments, saves commands and raw metrics, checks
deterministic short/16k answers, and hashes the installed vLLM Python sources
before and after the plugin stage. Short workloads establish migration
equivalence; they do not replace the official four-round/Math500 acceptance.

### Verified migration (2026-10-12)

Execution commit e86fd22 passed 4 CPU selection checks, function AST equivalence,
11 GPU cases with bitwise equality to the first-version oracle and a dense
reference (max absolute error 1.30e-4), and identical short/16,379-token answers.
All 5,426 installed vLLM Python source files remained byte-identical. Each
variant completed 97 benchmark requests with full token counts, no cache hits
or preemptions, and container cleanup. The plugin differed from the legacy
patch by -0.53% on the short 4k case and +0.04% on short 16k. See the archived
migration report. Full Math500 and official four-round acceptance were not
repeated; the known formal 4k regression is not fixed by this migration.
