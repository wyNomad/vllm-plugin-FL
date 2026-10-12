#!/usr/bin/env python3
"""Same seeded small service loads as the first-version Native 2D probe."""

import importlib.util
import json
import sys
from pathlib import Path


source = Path(__file__).resolve().parents[1] / 'results/native2d_20261011/bench_service_ab.py'
spec = importlib.util.spec_from_file_location('original_probe_benchmark', source)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.ROOT = Path(sys.argv[1])
variant = sys.argv[2]
rows = []
for case in bench.CASES:
    rows.append(bench.run_case(variant, case))
    (bench.ROOT / variant / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
(bench.ROOT / variant / 'benchmark_succeeded.ok').write_text('all seeded requests passed\n')
