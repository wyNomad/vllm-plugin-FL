#!/usr/bin/env python3
"""Run the gated Native 2D service probe in our already-patched GPU 1 container."""

import json
import subprocess
import time
from pathlib import Path


ROOT = Path('/workspace/wyNomad/results/native2d_20261011')
NAME = 'wynomad-minicpm-s2-gpu1'
COMPILATION = {'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_num_of_warmups': 0}


def container_exists():
    result = subprocess.run(
        ['docker', 'ps', '-a', '--filter', f'name=^{NAME}$', '--format', '{{.Names}}'],
        capture_output=True, text=True, timeout=15, check=True,
    )
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


def correctness(name, client):
    result = subprocess.run(['docker', 'exec', NAME, 'python3', client],
                            capture_output=True, text=True, timeout=420)
    (ROOT / f'{name}.stdout').write_text(result.stdout)
    (ROOT / f'{name}.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'{name} client exited {result.returncode}')
    response = json.loads(result.stdout)
    answer = response['choices'][0]['message']['content']
    if name == 'short_math':
        if '117' not in answer or '13' not in answer:
            raise RuntimeError('short math answer mismatch')
    else:
        if response['usage']['prompt_tokens'] < 16000 or not answer.strip().endswith('4'):
            raise RuntimeError('16k math answer mismatch')
    print('CORRECTNESS', name, response['usage'], flush=True)


def stop_personal_container():
    try:
        result = subprocess.run(['docker', 'stop', '--time', '20', NAME],
                                capture_output=True, text=True, timeout=90)
        (ROOT / 'container_stop.log').write_text(
            result.stdout + result.stderr + f'exit_code={result.returncode}\n')
    except subprocess.TimeoutExpired:
        (ROOT / 'container_stop.log').write_text('docker stop timed out\n')
    for _ in range(15):
        if not container_exists():
            return
        time.sleep(2)
    raise RuntimeError('personal container still exists after stop')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    code = 1
    try:
        if not container_exists():
            raise RuntimeError(f'expected patched personal container {NAME}')
        command = [
            'docker', 'exec', '-e', 'VLLM_FL_NATIVE2D_PROBE=1', NAME,
            'vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
            '--served-model-name', 'minicpm', '--max-model-len', '131072',
            '--gpu-memory-utilization', '0.85', '--compilation-config',
            json.dumps(COMPILATION),
        ]
        (ROOT / 'serve_command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (ROOT / 'serve.log').open('w') as stream:
            service = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            wait_ready(service)
            print('SERVICE READY', flush=True)
            correctness('short_math', '/workspace/benchmark_results/hybrid_correctness_20261009/client.py')
            correctness('long_16k_math', '/workspace/benchmark_results/hybrid_correctness_20261009/long_client.py')
            (ROOT / 'correctness_passed.ok').write_text('two deterministic requests passed\n')
            bench = ['docker', 'exec', NAME, 'python3',
                     '/workspace/benchmark_results/native2d_20261011/bench_service_ab.py',
                     'native2d']
            with (ROOT / 'benchmark.log').open('w') as bench_stream:
                result = subprocess.run(bench, stdout=bench_stream,
                                        stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode:
                raise RuntimeError(f'short service benchmark failed: {result.returncode}')
            (ROOT / 'benchmark_passed.ok').write_text('short service A/B loads passed\n')
            print('BENCHMARK COMPLETE', flush=True)
        code = 0
    except Exception as exc:
        print('ERROR', repr(exc), flush=True)
    finally:
        if container_exists():
            try:
                stop_personal_container()
            except Exception as exc:
                print('CLEANUP ERROR', repr(exc), flush=True)
                code = 1
        (ROOT / 'runner.exit').write_text(f'{code}\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
