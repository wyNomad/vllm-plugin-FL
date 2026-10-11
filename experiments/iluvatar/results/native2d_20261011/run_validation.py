#!/usr/bin/env python3
"""Gate the PDF-inspired Native 2D candidate by Math500, then official throughput.

Run only after reviewing the short probe. Each stage gets a fresh disposable
personal GPU 1 container. Failure stops the sequence and cleans that container.
"""

import csv
import json
import subprocess
import time
from pathlib import Path


ROOT = Path('/workspace/wyNomad/results/native2d_20261011')
REPORT_ROOT = Path('/workspace/wyNomad/evalscope-results/native2d_accuracy_20261011')
WORK_DIR = '/workspace/evalscope-datasets/level3/native2d_accuracy_20261011'
NAME = 'wynomad-minicpm-s2-gpu1'
CASES = [[4096, 1024, 64, 256], [16384, 1024, 64, 128]]
COMPILATION = {'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_num_of_warmups': 0}


def exists():
    result = subprocess.run(['docker', 'ps', '-a', '--filter', f'name=^{NAME}$',
                             '--format', '{{.Names}}'], capture_output=True,
                            text=True, timeout=15, check=True)
    return NAME in result.stdout.splitlines()


def wait_ready(service):
    probe = ['docker', 'exec', NAME, 'python3', '-c',
             "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9031/health', timeout=3)"]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if service.poll() is not None:
            raise RuntimeError(f'vLLM exited early with {service.returncode}')
        status = subprocess.run(probe, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=8)
        if status.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError('vLLM service did not become healthy')


def stop_personal_container(directory):
    try:
        result = subprocess.run(['docker', 'stop', '--time', '20', NAME],
                                capture_output=True, text=True, timeout=90)
        (directory / 'container_stop.log').write_text(
            result.stdout + result.stderr + f'exit_code={result.returncode}\n')
    except subprocess.TimeoutExpired:
        (directory / 'container_stop.log').write_text('docker stop timed out\n')
    for _ in range(45):
        if not exists():
            return
        time.sleep(2)
    kill = subprocess.run(['docker', 'kill', NAME], capture_output=True,
                          text=True, timeout=30)
    (directory / 'container_kill.log').write_text(
        kill.stdout + kill.stderr + f'exit_code={kill.returncode}\n')
    for _ in range(15):
        if not exists():
            return
        time.sleep(2)
    raise RuntimeError('our personal container still exists')


def run_stage(name, workload):
    directory = ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    if exists():
        raise RuntimeError(f'refusing to reuse existing {NAME}')
    started = False
    try:
        launch = subprocess.run(['python3', '/workspace/wyNomad/run_tianshu_container.py',
                                 '--detach', '1'], capture_output=True, text=True,
                                timeout=60)
        (directory / 'container_launch.log').write_text(
            launch.stdout + launch.stderr + f'exit_code={launch.returncode}\n')
        if launch.returncode:
            raise RuntimeError(f'container launch failed for {name}')
        started = True
        patch = ['docker', 'exec', NAME, 'python3',
                 '/workspace/benchmark_results/native2d_20261011/patch_vllm.py',
                 '--evidence', f'/workspace/benchmark_results/native2d_20261011/{name}/source_patch']
        with (directory / 'patch.log').open('w') as stream:
            result = subprocess.run(patch, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=30)
        if result.returncode:
            raise RuntimeError(f'exact-source patch failed for {name}')
        serve = [
            'docker', 'exec', '-e', 'VLLM_FL_NATIVE2D_PROBE=1', NAME,
            'vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
            '--served-model-name', 'minicpm', '--max-model-len', '131072',
            '--gpu-memory-utilization', '0.85', '--compilation-config',
            json.dumps(COMPILATION),
        ]
        (directory / 'serve_command.json').write_text(json.dumps(serve, indent=2) + '\n')
        with (directory / 'serve.log').open('w') as stream:
            service = subprocess.Popen(serve, stdout=stream, stderr=subprocess.STDOUT)
            wait_ready(service)
            print('SERVICE READY', name, flush=True)
            workload(directory)
        (directory / 'stage_succeeded.ok').write_text(f'{name} passed\n')
    finally:
        if started:
            stop_personal_container(directory)


def accuracy(directory):
    dataset_args = {'math_500': {
        'dataset_id': '/workspace/evalscope-datasets/math_500',
        'subset_list': ['Level 3'],
    }}
    generation = {'max_tokens': 32768, 'top_p': 0.95, 'temperature': 1.0}
    command = [
        'docker', 'exec', NAME, 'evalscope', 'eval', '--model', 'minicpm',
        '--datasets', 'math_500', '--dataset-args', json.dumps(dataset_args),
        '--eval-type', 'openai_api', '--api-url',
        'http://127.0.0.1:9031/v1/chat/completions', '--api-key', 'EMPTY',
        '--eval-batch-size', '8', '--generation-config', json.dumps(generation),
        '--timeout', '3600', '--seed', '42', '--ignore-errors',
        '--work-dir', WORK_DIR,
    ]
    (directory / 'evaluation_command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (directory / 'eval.log').open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=7200)
    if result.returncode:
        raise RuntimeError(f'EvalScope exited {result.returncode}')
    reports = list(REPORT_ROOT.glob('*/reports/minicpm/math_500.json'))
    if len(reports) != 1:
        raise RuntimeError(f'expected one Math500 report, got {len(reports)}')
    report = json.loads(reports[0].read_text())
    summary = report['execution_summary']
    if (summary['requested'] != 105 or summary['succeeded'] != 105
            or summary['errored'] != 0 or summary['incomplete']):
        raise RuntimeError(f'Math500 incomplete: {summary}')
    score = next(metric['score'] for metric in report['metrics']
                 if metric['identity']['name'] == 'accuracy')
    print('ACCURACY', score, reports[0], flush=True)
    if score < 0.95:
        raise RuntimeError(f'Math500 below gate: {score}')
    (directory / 'accuracy_succeeded.ok').write_text(f'accuracy={score}\n')


def formal(directory):
    command = [
        'docker', 'exec', '-w',
        '/workspace/benchmark_results/native2d_20261011/formal',
        NAME, 'python3',
        '/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py',
        '--model', '/workspace/MiniCPM5-2B', '--served-model-name', 'minicpm',
        '--port', '9031', '--test-cases', json.dumps(CASES),
    ]
    (directory / 'benchmark_command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (directory / 'bench.log').open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=36000)
    if result.returncode:
        raise RuntimeError(f'official throughput script exited {result.returncode}')
    output = directory / 'benchmark_results'
    raws = list(output.glob('raw_runs_*.csv'))
    summaries = list(output.glob('summary_*.csv'))
    if len(raws) != 1 or len(summaries) != 1:
        raise RuntimeError('missing unique raw or summary CSV')
    with raws[0].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with summaries[0].open(newline='') as stream:
        scores = list(csv.DictReader(stream))
    if len(rows) != 8 or len(scores) != 2:
        raise RuntimeError('official throughput has incomplete rows')
    for row in rows:
        if (row['Run Status'] != 'SUCCESS'
                or int(float(row['Successful Requests'])) != int(row['Num Prompts'])):
            raise RuntimeError('official throughput includes failed requests')
    print('FORMAL RESULTS', [(row['Prefill'], row['Total tok/s']) for row in scores], flush=True)


def short_ab_gate():
    if not (ROOT / 'baseline_short_finished.ok').exists():
        raise RuntimeError('same-GPU Triton reference is incomplete')
    reference = json.loads((ROOT / 'triton_refresh/summary.json').read_text())
    candidate = json.loads((ROOT / 'native2d/summary.json').read_text())
    by_name = {row['case']: row for row in reference}
    optimized = {row['case']: row for row in candidate}
    thresholds = {'decode_4k': 1.05, 'long_16k': 1.10}
    ratios = {}
    for name, minimum in thresholds.items():
        base = by_name[name]
        trial = optimized[name]
        for row in (base, trial):
            if (row['exit_code'] != 0 or row['failed_requests'] != 0
                    or row['successful_requests'] != row['concurrency']):
                raise RuntimeError(f'incomplete short load: {name}, {row}')
        ratios[name] = trial['total_tokens_per_s'] / base['total_tokens_per_s']
        if ratios[name] < minimum:
            raise RuntimeError(f'short A/B gate failed: {name} ratio={ratios[name]:.3f}')
    (ROOT / 'short_ab_gate.json').write_text(json.dumps(ratios, indent=2) + '\n')
    print('SHORT A/B GATE', ratios, flush=True)


def main():
    code = 1
    try:
        if (ROOT / 'runner.exit').read_text().strip() != '0':
            if not ((ROOT / 'benchmark_passed.ok').exists()
                    and (ROOT / 'cleanup_verified.ok').exists()):
                raise RuntimeError('short probe failed or cleanup was not verified')
        if not (ROOT / 'correctness_passed.ok').exists():
            raise RuntimeError('short probe correctness gate missing')
        short_ab_gate()
        run_stage('accuracy', accuracy)
        run_stage('formal', formal)
        (ROOT / 'validation_succeeded.ok').write_text('accuracy and eight formal runs passed\n')
        code = 0
    except Exception as exc:
        print('ERROR', repr(exc), flush=True)
    finally:
        (ROOT / 'validation.exit').write_text(f'{code}\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
