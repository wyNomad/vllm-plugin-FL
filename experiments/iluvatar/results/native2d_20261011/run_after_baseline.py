#!/usr/bin/env python3
"""Wait for the same-GPU reference, then run gated accuracy and formal checks."""

import subprocess
import time
from pathlib import Path


ROOT = Path('/workspace/wyNomad/results/native2d_20261011')


def main():
    deadline = time.monotonic() + 4 * 3600
    reference_exit = ROOT / 'baseline_short.exit'
    while not reference_exit.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('same-GPU baseline did not finish within four hours')
        time.sleep(15)
    if reference_exit.read_text().strip() != '0':
        raise RuntimeError('same-GPU baseline supervisor failed')
    command = ['python3', '-u', str(ROOT / 'run_validation.py')]
    with (ROOT / 'validation_supervisor.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                timeout=16 * 3600)
    if result.returncode:
        raise RuntimeError(f'validation failed: {result.returncode}')
    (ROOT / 'pipeline_succeeded.ok').write_text(
        'same-GPU short A/B, Math500, formal throughput passed\n'
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (ROOT / 'pipeline_error.txt').write_text(repr(error) + '\n')
        raise
