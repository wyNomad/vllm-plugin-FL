#!/usr/bin/env python3
"""Representative 1024-output 4k screening, with 16k prefill preservation."""
import importlib.util
import json
import sys
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'results/native2d_20261011/bench_service_ab.py'
spec = importlib.util.spec_from_file_location('seeded_benchmark', source)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.ROOT = Path(sys.argv[1])
variant = sys.argv[2]
rows = []
cases = [('warmup', 4096, 32, 1, 2701),
         ('decode_4k_1024', 4096, 1024, 64, 2702),
         ('long_16k_128', 16384, 128, 32, 2703)]
for case in cases:
    row = bench.run_case(variant, case)
    assert row['cached_tokens_delta'] == 0 and row['preemptions_delta'] == 0
    rows.append(row)
    (bench.ROOT / variant / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
(bench.ROOT / variant / 'benchmark_succeeded.ok').write_text('all seeded requests passed\n')
