#!/usr/bin/env python3
"""Official baseline/candidate acceptance after the representative screen."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('screen_utilities', REPO / 'experiments/iluvatar/native2d_context/run_screen.py')
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)
util = screen.util
PERSONAL = Path('/workspace/wyNomad')
CASES = [[4096, 1024, 64, 256], [16384, 1024, 64, 128]]


def formal(directory):
    command = ['docker', 'exec', '-w', '/workspace/benchmark_results/' + directory.parent.name + '/' + directory.name,
               util.common.NAME, 'python3', '/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py',
               '--model', '/workspace/MiniCPM5-2B', '--served-model-name', 'minicpm', '--port', '9031',
               '--test-cases', json.dumps(CASES)]
    util.record(directory, 'benchmark_command.json', command)
    with (directory / 'bench.log').open('w') as out:
        subprocess.run(command, check=True, stdout=out, stderr=subprocess.STDOUT, timeout=36000)
    raw_files = list((directory / 'benchmark_results').glob('raw_runs_*.csv'))
    summary_files = list((directory / 'benchmark_results').glob('summary_*.csv'))
    assert len(raw_files) == len(summary_files) == 1
    with raw_files[0].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with summary_files[0].open(newline='') as stream:
        summaries = list(csv.DictReader(stream))
    assert len(rows) == 8 and len(summaries) == 2
    scores = []
    for case in CASES:
        selected = [row for row in rows if int(row['Prefill']) == case[0]]
        assert len(selected) == 4
        for row in selected:
            assert row['Run Status'] == 'SUCCESS'
            assert int(float(row['Successful Requests'])) == int(row['Num Prompts']) == case[3]
            assert int(float(row['Total Input Tokens'])) == case[0] * case[3]
            assert int(float(row['Total Output Tokens'])) == case[1] * case[3]
        counted = selected[1:]
        score = dict(input_len=case[0], output_len=case[1], concurrency=case[2], count=case[3],
                     throughput=statistics.mean(float(row['Total tok/s']) for row in counted),
                     ttft_ms=statistics.mean(float(row['Mean TTFT (ms)']) for row in counted),
                     tpot_ms=statistics.mean(float(row['Mean TPOT (ms)']) for row in counted))
        summary = next(row for row in summaries if int(row['Prefill']) == case[0])
        assert abs(score['throughput'] - float(summary['Total tok/s'])) <= .011
        scores.append(score)
    util.record(directory, 'validated_scores.json', scores)
    print('FORMAL', directory.name, json.dumps(scores), flush=True)


def accuracy(directory):
    util.common.REPORT_ROOT = PERSONAL / 'evalscope-results' / directory.parent.name
    util.common.WORK_DIR = '/workspace/evalscope-datasets/level3/' + directory.parent.name
    util.common.accuracy(directory)


def comparison(root):
    base = json.loads((root / 'baseline_formal/validated_scores.json').read_text())
    candidate = json.loads((root / 'prefill_formal/validated_scores.json').read_text())
    rows = []
    for b, c in zip(base, candidate):
        assert b['input_len'] == c['input_len']
        item = dict(input_len=b['input_len'], throughput_change_pct=100 * (c['throughput'] / b['throughput'] - 1),
                    ttft_change_pct=100 * (c['ttft_ms'] / b['ttft_ms'] - 1),
                    tpot_change_pct=100 * (c['tpot_ms'] / b['tpot_ms'] - 1))
        item['performance_gate_passed'] = item['throughput_change_pct'] >= -1 and item['ttft_change_pct'] <= 1
        rows.append(item)
    util.record(root, 'comparison.json', dict(baseline=base, prefill=candidate, changes=rows,
        performance_gate_passed=all(row['performance_gate_passed'] for row in rows)))
    print('OFFICIAL COMPARISON', json.dumps(rows), flush=True)
    assert all(row['performance_gate_passed'] for row in rows), 'competition performance gate failed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--run-id', default='native2d_prefill_acceptance_20261012')
    parser.add_argument('--screen-id', default='native2d_prefill_screen_20261012')
    args = parser.parse_args()
    if not 0 <= args.gpu <= 15:
        parser.error('invalid GPU')
    for name in (args.run_id, args.screen_id):
        if not name.startswith('native2d_prefill_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name):
            parser.error('invalid unique run/screen id')
    screened = PERSONAL / 'results' / args.screen_id
    assert (screened / 'run.exit').read_text().strip() == '0'
    prior = json.loads((screened / 'comparison.json').read_text())
    checks = [row for row in prior['changes'] if row['case'] != 'warmup']
    assert len(checks) == 2
    assert all(row['throughput_change_pct'] >= -1 and row['ttft_change_pct'] <= 1 for row in checks), 'screen failed; do not start formal'
    numeric = PERSONAL / 'results/native2d_context_20261012_numeric_v2'
    assert (numeric / 'run.exit').read_text().strip() == '0'
    root = PERSONAL / 'results' / args.run_id
    util.common.NAME = 'wynomad-minicpm-s2-gpu' + str(args.gpu)
    with (PERSONAL / ('native2d_gpu' + str(args.gpu) + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(exist_ok=False)
        commit = util.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
        util.record(root, 'manifest.json', dict(commit=commit, gpu=args.gpu, started_at=util.stamp(),
                     cases=CASES, rounds=4, discard_first=True, screened_from=args.screen_id))
        code = 1
        try:
            for name, policy, workload in [('accuracy', 'prefill', accuracy),
                                           ('baseline_formal', 'baseline', formal),
                                           ('prefill_formal', 'prefill', formal)]:
                (root / 'current_stage.txt').write_text(name + ': running\n')
                screen.stage(name, policy, args.gpu, root, commit, workload=workload)
            comparison(root)
            (root / 'current_stage.txt').write_text('complete: accuracy and competition gates passed\n')
            code = 0
        except Exception as error:
            (root / 'current_stage.txt').write_text('failed: ' + repr(error) + '\n')
            print('ERROR', repr(error), flush=True)
        finally:
            (root / 'run.exit').write_text(str(code) + '\n')
            (root / 'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(root)) + '\n'
                 for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.txt'))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
