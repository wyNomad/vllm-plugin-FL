#!/usr/bin/env python3
"""Refresh unpatched Triton formal scores on GPU 1 after the candidate exits."""

import csv
import json
import subprocess
import time

import run_validation as common


DIRECTORY = common.ROOT / 'matched_baseline_formal'


def wait_candidate():
    deadline = time.monotonic() + 6 * 3600
    marker = common.ROOT / 'validation.exit'
    while not marker.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('candidate did not finish within six hours')
        time.sleep(15)
    if marker.read_text().strip() != '0':
        raise RuntimeError('candidate supervision failed; inspect before baseline')
    if common.exists():
        raise RuntimeError('candidate personal container still exists')


def benchmark():
    command = [
        'docker', 'exec', '-w',
        '/workspace/benchmark_results/native2d_20261011/matched_baseline_formal',
        common.NAME, 'python3',
        '/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py',
        '--model', '/workspace/MiniCPM5-2B', '--served-model-name', 'minicpm',
        '--port', '9031', '--test-cases', json.dumps(common.CASES),
    ]
    (DIRECTORY / 'benchmark_command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (DIRECTORY / 'bench.log').open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=36000)
    if result.returncode:
        raise RuntimeError(f'original benchmark exited {result.returncode}')
    output = DIRECTORY / 'benchmark_results'
    raws = list(output.glob('raw_runs_*.csv'))
    summaries = list(output.glob('summary_*.csv'))
    if len(raws) != 1 or len(summaries) != 1:
        raise RuntimeError('missing unique raw or summary CSV')
    rows = list(csv.DictReader(raws[0].open(newline='')))
    scores = list(csv.DictReader(summaries[0].open(newline='')))
    if len(rows) != 8 or len(scores) != 2:
        raise RuntimeError('incomplete formal baseline')
    for row in rows:
        if (row['Run Status'] != 'SUCCESS'
                or float(row['Successful Requests']) != float(row['Num Prompts'])):
            raise RuntimeError('formal baseline has failed requests')
    print('MATCHED BASELINE', [(r['Prefill'], r['Total tok/s']) for r in scores], flush=True)
    (DIRECTORY / 'benchmark_succeeded.ok').write_text('eight formal runs passed\n')


def main():
    started = False
    code = 1
    try:
        print('WAITING FOR CANDIDATE', flush=True)
        wait_candidate()
        DIRECTORY.mkdir(exist_ok=False)
        launch = subprocess.run(
            ['python3', '/workspace/wyNomad/run_tianshu_container.py', '--detach', '1'],
            capture_output=True, text=True, timeout=60,
        )
        (DIRECTORY / 'container_launch.log').write_text(
            launch.stdout + launch.stderr + f'exit_code={launch.returncode}\n')
        if launch.returncode:
            raise RuntimeError('personal idle-GPU container launch failed')
        started = True
        check = [
            'docker', 'exec', common.NAME, 'python3',
            '/workspace/benchmark_results/native2d_20261011/patch_vllm.py', '--dry-run',
        ]
        with (DIRECTORY / 'source_check.log').open('w') as stream:
            result = subprocess.run(check, stdout=stream, stderr=subprocess.STDOUT, timeout=30)
        if result.returncode:
            raise RuntimeError('baseline installed vLLM does not match original hashes')
        serve = [
            'docker', 'exec', common.NAME, 'vllm', 'serve', '/workspace/MiniCPM5-2B',
            '--port', '9031', '--served-model-name', 'minicpm',
            '--max-model-len', '131072', '--gpu-memory-utilization', '0.85',
            '--compilation-config', json.dumps(common.COMPILATION),
        ]
        (DIRECTORY / 'serve_command.json').write_text(json.dumps(serve, indent=2) + '\n')
        with (DIRECTORY / 'serve.log').open('w') as stream:
            service = subprocess.Popen(serve, stdout=stream, stderr=subprocess.STDOUT)
            common.wait_ready(service)
            print('SERVICE READY matched baseline', flush=True)
            benchmark()
        code = 0
    except Exception as error:
        print('ERROR', repr(error), flush=True)
    finally:
        if started:
            try:
                common.stop_personal_container(DIRECTORY)
            except Exception as error:
                print('CLEANUP ERROR', repr(error), flush=True)
                code = 1
        (common.ROOT / 'matched_baseline_formal.exit').write_text(f'{code}\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
