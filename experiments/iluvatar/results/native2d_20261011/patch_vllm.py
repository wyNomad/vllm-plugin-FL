#!/usr/bin/env python3
"""Patch the pinned vLLM 0.24 files inside a disposable personal container.

This is a PDF-inspired probe, not a reproduction of the unavailable patch.
The only active kernel change is gated by VLLM_FL_NATIVE2D_PROBE=1.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path


BACKEND = Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/backends/triton_attn.py')
KERNEL = Path('/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py')
EXPECTED = {
    BACKEND: '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
    KERNEL: 'bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0',
}


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected one match, found {count}')
    return source.replace(old, new, 1)


def patch_backend(source):
    return replace_once(
        source,
        '            k_descale = layer._k_scale.expand(descale_shape)\n'
        '            v_descale = layer._v_scale.expand(descale_shape)\n',
        '            # Non-quantized KV does not have a descale. Keep quantized paths intact.\n'
        '            if self._kv_quant_mode == KVQuantMode.NONE:\n'
        '                k_descale = None\n'
        '                v_descale = None\n'
        '            else:\n'
        '                k_descale = layer._k_scale.expand(descale_shape)\n'
        '                v_descale = layer._v_scale.expand(descale_shape)\n',
        'BF16/auto descale',
    )


def patch_kernel(source):
    source = replace_once(source, 'from typing import Any\n', 'import os\nfrom typing import Any\n', 'os import')
    source = replace_once(
        source,
        'float8_info = torch.finfo(current_platform.fp8_dtype())\n',
        'float8_info = torch.finfo(current_platform.fp8_dtype())\n'
        '_native2d_probe_hits = 0\n',
        'route counter',
    )
    source = replace_once(
        source,
        '    if tuned_large_head:\n'
        '        BLOCK_M = 32\n'
        '        BLOCK_Q = BLOCK_M // num_queries_per_kv\n'
        '        launch_num_warps = 8\n'
        '        launch_num_stages = 2\n',
        '    if tuned_large_head:\n'
        '        BLOCK_M = 32\n'
        '        BLOCK_Q = BLOCK_M // num_queries_per_kv\n'
        '        launch_num_warps = 8\n'
        '        launch_num_stages = 2\n'
        '\n'
        '    # BI-V150/MiniCPM5-2B probe. Keep small decode batches on native 3D.\n'
        '    native2d_probe = (\n'
        '        os.environ.get("VLLM_FL_NATIVE2D_PROBE") == "1"\n'
        '        and kv_quant_mode == KVQuantMode.NONE\n'
        '        and head_size == 128\n'
        '        and num_queries_per_kv == 8\n'
        '        and not use_td\n'
        '        and (max_seqlen_q > 1 or num_seqs >= 16)\n'
        '    )\n'
        '    if native2d_probe:\n'
        '        BLOCK_M = 128\n'
        '        BLOCK_Q = BLOCK_M // num_queries_per_kv\n'
        '        launch_num_warps = 8\n'
        '        launch_num_stages = 2\n',
        'Native 2D launch shape',
    )
    source = replace_once(
        source,
        '    if tuned_large_head:\n'
        '        TILE_SIZE_PREFILL = 128\n',
        '    if tuned_large_head:\n'
        '        TILE_SIZE_PREFILL = 128\n'
        '    if native2d_probe:\n'
        '        TILE_SIZE_PREFILL = 16\n'
        '        TILE_SIZE_DECODE = 16\n',
        'KV tile size',
    )
    source = replace_once(
        source,
        '        or is_batch_invariant\n'
        '    )\n',
        '        or is_batch_invariant\n'
        '    )\n'
        '    if native2d_probe:\n'
        '        use_3d = False\n'
        '        global _native2d_probe_hits\n'
        '        _native2d_probe_hits += 1\n'
        '        if _native2d_probe_hits in (1, 10, 100, 1000):\n'
        '            logger.info(\n'
        '                "NATIVE2D_PROBE hit=%s seqs=%s max_q=%s block_m=%s tile=%s",\n'
        '                _native2d_probe_hits, num_seqs, max_seqlen_q,\n'
        '                BLOCK_M, TILE_SIZE_PREFILL,\n'
        '            )\n',
        'route selection and evidence',
    )
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--backend', type=Path, default=BACKEND)
    parser.add_argument('--kernel', type=Path, default=KERNEL)
    parser.add_argument('--evidence', type=Path, default=Path('/workspace/benchmark_results/native2d_20261011/source_patch'))
    args = parser.parse_args()
    files = [(args.backend, BACKEND, patch_backend), (args.kernel, KERNEL, patch_kernel)]
    changes = []
    for path, identity, transform in files:
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        if digest != EXPECTED[identity]:
            raise RuntimeError(f'{path}: unexpected SHA-256 {digest}')
        updated = transform(original.decode())
        compile(updated, str(path), 'exec')
        changes.append((path, original, updated.encode()))
    if args.dry_run:
        print('dry run passed: exact source hashes and patch anchors match')
        return
    args.evidence.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path, original, updated in changes:
        (args.evidence / (path.name + '.original')).write_bytes(original)
        (args.evidence / (path.name + '.diff')).write_text(''.join(difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            updated.decode().splitlines(keepends=True),
            fromfile=path.name + '.original', tofile=path.name + '.probe',
        )))
        path.write_bytes(updated)
        manifest.append({'path': str(path), 'original_sha256': hashlib.sha256(original).hexdigest(),
                         'probe_sha256': hashlib.sha256(updated).hexdigest()})
    (args.evidence / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
