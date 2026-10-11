#!/usr/bin/env python3
"""Keep the Native 2D prefill benefit but gate decode by KV context length.

This is a separate follow-up candidate. It must run in a fresh personal
container and must never be applied on top of the first candidate.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

import patch_vllm as base


def patch_kernel_long_context(source):
    updated = base.patch_kernel(source)
    updated = base.replace_once(
        updated,
        'os.environ.get("VLLM_FL_NATIVE2D_PROBE") == "1"',
        'os.environ.get("VLLM_FL_NATIVE2D_LONG_CONTEXT") == "1"',
        'distinct experiment switch',
    )
    updated = base.replace_once(
        updated,
        '(max_seqlen_q > 1 or num_seqs >= 16)',
        '(max_seqlen_q > 1 or (num_seqs >= 16 and max_seqlen_k >= 8192))',
        'keep 4k decode on original route',
    )
    updated = base.replace_once(
        updated,
        'NATIVE2D_PROBE hit=%s',
        'NATIVE2D_LONG_CONTEXT hit=%s',
        'distinct route evidence',
    )
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--backend', type=Path, default=base.BACKEND)
    parser.add_argument('--kernel', type=Path, default=base.KERNEL)
    parser.add_argument('--evidence', type=Path, default=Path(
        '/workspace/benchmark_results/native2d_20261011/long_context/source_patch'))
    args = parser.parse_args()
    files = [
        (args.backend, base.BACKEND, base.patch_backend),
        (args.kernel, base.KERNEL, patch_kernel_long_context),
    ]
    changes = []
    for path, identity, transform in files:
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        if digest != base.EXPECTED[identity]:
            raise RuntimeError(f'{path}: unexpected SHA-256 {digest}')
        updated = transform(original.decode())
        compile(updated, str(path), 'exec')
        changes.append((path, original, updated.encode()))
    if args.dry_run:
        print('long-context dry run passed: exact source hashes and anchors match')
        return
    args.evidence.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path, original, updated in changes:
        (args.evidence / (path.name + '.original')).write_bytes(original)
        (args.evidence / (path.name + '.diff')).write_text(''.join(
            difflib.unified_diff(
                original.decode().splitlines(keepends=True),
                updated.decode().splitlines(keepends=True),
                fromfile=path.name + '.original', tofile=path.name + '.long-context',
            )
        ))
        path.write_bytes(updated)
        manifest.append({
            'path': str(path),
            'original_sha256': hashlib.sha256(original).hexdigest(),
            'long_context_sha256': hashlib.sha256(updated).hexdigest(),
        })
    (args.evidence / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
