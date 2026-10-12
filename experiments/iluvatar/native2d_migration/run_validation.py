#!/usr/bin/env python3
"""Verify first-version migration in fresh, sequential personal containers."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / 'experiments/iluvatar'
PERSONAL = Path('/workspace/wyNomad')
ROOT = PERSONAL / 'results/native2d_plugin_20261012'
CONTAINER_ROOT = '/workspace/benchmark_results/native2d_plugin_20261012'
CONTAINER_CODE = '/workspace/vllm-plugin-FL/experiments/iluvatar/native2d_migration'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


common = module('cleanup_and_health', ARCHIVE / 'results/native2d_20261011/run_validation.py')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def run(args, timeout=30):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def record(directory, name, value):
    (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def installed_sources():
    code = (
        "import hashlib,importlib.util,json;from pathlib import Path;"
        "r=Path(importlib.util.find_spec('vllm').origin).parent;"
        "print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() "
        "for p in sorted(r.rglob('*.py'))}))"
    )
    return json.loads(run(['docker', 'exec', common.NAME, 'python3', '-c', code]))


def correctness(directory):
    for name, client in [('short_math', 'client.py'), ('long_16k_math', 'long_client.py')]:
        path = '/workspace/vllm-plugin-FL/experiments/iluvatar/results/hybrid_correctness_20261009/' + client
        completed = subprocess.run(['docker', 'exec', common.NAME, 'python3', path],
                                   capture_output=True, text=True, timeout=420)
        (directory / (name + '.stdout')).write_text(completed.stdout)
        (directory / (name + '.stderr')).write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError('deterministic request failed: ' + name)
        response = json.loads(completed.stdout)
        answer = response['choices'][0]['message']['content']
        if name == 'short_math':
            assert '117' in answer and '13' in answer, 'short math output mismatch'
        else:
            assert response['usage']['prompt_tokens'] >= 16000 and answer.strip().endswith('4'), '16k output mismatch'
        record(directory, name + '_check.json', {'passed': True, 'usage': response['usage']})
        print('CORRECTNESS', directory.name, name, response['usage'], flush=True)


def stage(name, commit):
    directory = ROOT / name
    directory.mkdir(exist_ok=False)
    assert run(['git', '-C', str(REPO), 'rev-parse', 'HEAD']) == commit
    assert not run(['git', '-C', str(REPO), 'status', '--porcelain'])
    if common.exists():
        raise RuntimeError('personal GPU 1 container already exists')
    started = False
    identity = None
    try:
        launch = ['python3', str(ARCHIVE / 'environment/run_tianshu_container.py'), '--detach', '1']
        with (directory / 'container_launch.log').open('w') as stream:
            subprocess.run(launch, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
        started = True
        identity = run(['docker', 'inspect', '--format', '{{.Id}}', common.NAME])
        image_id = run(['docker', 'inspect', '--format', '{{.Image}}', identity])
        digests = json.loads(run(['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', image_id]))
        assert any(value.endswith('@sha256:cc8ed9969977a06996c86c483851567f212ab9761bb9147e85082b4a34ce88b8') for value in digests)
        record(directory, 'environment.json', {'plugin_commit': commit, 'container_id': identity,
               'image_id': image_id, 'repo_digests': digests, 'gpu': 1,
               'compilation': {'cudagraph_mode': 'FULL_DECODE_ONLY'}, 'started_at': stamp()})
        before = installed_sources()
        record(directory, 'vllm_sources_before.json', before)
        if name == 'legacy_reference':
            patch = ['docker', 'exec', common.NAME, 'python3',
                     '/workspace/vllm-plugin-FL/experiments/iluvatar/results/native2d_20261011/patch_vllm.py',
                     '--evidence', CONTAINER_ROOT + '/' + name + '/source_patch']
            with (directory / 'patch.log').open('w') as stream:
                subprocess.run(patch, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=30)
            env = ['-e', 'VLLM_FL_NATIVE2D_PROBE=1']
        else:
            env = ['-e', 'VLLM_FL_ILUVATAR_NATIVE2D=1']
            numeric = ['docker', 'exec', *env, '-w', CONTAINER_ROOT + '/' + name, common.NAME,
                       'python3', CONTAINER_CODE + '/check_numerics.py']
            with (directory / 'numerics.log').open('w') as stream:
                subprocess.run(numeric, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
            print('GPU NUMERICS PASSED', stamp(), flush=True)
        serve = ['docker', 'exec', *env, common.NAME, 'vllm', 'serve', '/workspace/MiniCPM5-2B',
                 '--port', '9031', '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
                 '--max-model-len', '131072', '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}']
        record(directory, 'serve_command.json', serve)
        with (directory / 'serve.log').open('w') as stream:
            service = subprocess.Popen(serve, stdout=stream, stderr=subprocess.STDOUT)
            common.wait_ready(service)
            print('SERVICE READY', name, stamp(), flush=True)
            correctness(directory)
            (ROOT / 'current_stage.txt').write_text(name + ' short benchmarks running\n')
            command = ['docker', 'exec', common.NAME, 'python3', CONTAINER_CODE + '/service_checks.py', CONTAINER_ROOT, name]
            record(directory, 'benchmark_command.json', command)
            with (directory / 'bench.log').open('w') as bench_stream:
                subprocess.run(command, stdout=bench_stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
        after = installed_sources()
        record(directory, 'vllm_sources_after.json', after)
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        record(directory, 'vllm_source_changes.json', changed)
        if name == 'plugin':
            assert not changed, 'plugin validation modified installed vLLM sources'
            text = (directory / 'serve.log').read_text()
            assert 'Native 2D plugin backend loaded' in text and 'NATIVE2D_PLUGIN hit=' in text, 'plugin route not actually invoked'
        else:
            assert changed == ['v1/attention/backends/triton_attn.py', 'v1/attention/ops/triton_unified_attention.py']
    finally:
        if started and common.exists():
            assert run(['docker', 'inspect', '--format', '{{.Id}}', common.NAME]) == identity
            common.stop_personal_container(directory)
    (directory / 'stage_succeeded.ok').write_text(stamp() + '\n')


def comparison():
    previous = json.loads((ROOT / 'legacy_reference/summary.json').read_text())
    current = json.loads((ROOT / 'plugin/summary.json').read_text())
    old = {row['case']: row for row in previous}
    changes = []
    lines = ['# Native 2D 插件迁移验证', '',
             '同 GPU 1、新一次性容器、官网原服务命令；比较旧补丁版与插件版，未改变第一版路由。',
             'GPU 数值对照、短题与16k确定性题通过；插件版安装的vLLM Python源文件保持原样。', '',
             '| 负载 | 旧补丁 tok/s | 插件 tok/s | 变化 |', '| --- | ---: | ---: | ---: |']
    for row in current:
        base = old[row['case']]
        for item in (base, row):
            assert item['exit_code'] == 0 and item['failed_requests'] == 0
            assert item['successful_requests'] == item['concurrency']
            assert item['cached_tokens_delta'] == 0 and item['preemptions_delta'] == 0
        change = (row['total_tokens_per_s'] / base['total_tokens_per_s'] - 1) * 100
        changes.append({'case': row['case'], 'change_pct': change})
        lines.append(f"| {row['case']} | {base['total_tokens_per_s']:.2f} | {row['total_tokens_per_s']:.2f} | {change:+.2f}% |")
    lines += ['', '短版各一次只能用于迁移验证，不是正式成绩或重复性置信区间。',
              '4k正式回退修复与Math500/原版四轮正式验收仍待后续；本轮没有启用8192阈值方案。', '']
    record(ROOT, 'comparison.json', {'legacy_reference': previous, 'plugin': current, 'changes': changes})
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    if any(row['change_pct'] < -5 for row in changes if row['case'] != 'warmup'):
        raise RuntimeError('migration performance lost over 5%; inspect before acceptance')


def main():
    global ROOT, CONTAINER_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='native2d_plugin_20261012')
    args = parser.parse_args()
    if not args.run_id.startswith('native2d_plugin_') or any(
        char not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for char in args.run_id
    ):
        parser.error('use a unique simple native2d_plugin_* name')
    ROOT = PERSONAL / 'results' / args.run_id
    CONTAINER_ROOT = '/workspace/benchmark_results/' + args.run_id
    with (PERSONAL / 'native2d_gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ROOT.mkdir(exist_ok=False)
        commit = run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
        record(ROOT, 'run_manifest.json', {'plugin_commit': commit, 'started_at': stamp(),
               'stages': ['plugin', 'legacy_reference'], 'gpu': 1, 'new_routing_policy': False,
               'official_serve_compilation': {'cudagraph_mode': 'FULL_DECODE_ONLY'}})
        code = 1
        try:
            for name in ('plugin', 'legacy_reference'):
                (ROOT / 'current_stage.txt').write_text(name + ' starting\n')
                stage(name, commit)
            comparison()
            (ROOT / 'current_stage.txt').write_text('complete: migration verification passed\n')
            print('MIGRATION VERIFICATION COMPLETE', stamp(), flush=True)
            code = 0
        except Exception as error:
            (ROOT / 'current_stage.txt').write_text('failed: ' + repr(error) + '\n')
            print('ERROR', repr(error), flush=True)
        finally:
            (ROOT / 'run.exit').write_text(str(code) + '\n')
            paths = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
            (ROOT / 'SHA256SUMS.txt').write_text(''.join(
                hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(ROOT)) + '\n' for p in paths))
        return code


if __name__ == '__main__':
    raise SystemExit(main())
