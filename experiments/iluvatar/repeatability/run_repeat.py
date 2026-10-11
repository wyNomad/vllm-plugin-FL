#!/usr/bin/env python3
"""Repeat the unchanged Native 2D formal comparison, baseline first.

Run from the committed personal server checkout. A fresh result directory and
an exclusive personal GPU lock prevent accidental reuse. No accuracy rerun is
needed because this repeats the exact already validated candidate.
"""

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / 'experiments/iluvatar'
PREVIOUS = ARCHIVE / 'results/native2d_20261011'
PERSONAL = Path('/workspace/wyNomad')
EXPECTED_IMAGE = 'sha256:cc8ed9969977a06996c86c483851567f212ab9761bb9147e85082b4a34ce88b8'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


common = module('validated_native2d_runner', PREVIOUS / 'run_validation.py')
analysis = module('validated_formal_analysis', PREVIOUS / 'analyze_formal.py')
launcher = module('personal_launcher', ARCHIVE / 'environment/run_tianshu_container.py')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, timeout=30):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


def git(*args):
    return command(['git', '-C', str(REPO), *args])


def record(directory, name, value):
    (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def telemetry(directory, stop):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = launcher.DRIVER + '/lib64:' + env.get('LD_LIBRARY_PATH', '')
    with (directory / 'gpu_telemetry.log').open('w') as stream:
        while not stop.is_set():
            stream.write('\n' + stamp() + '\n')
            try:
                status = subprocess.run(['/usr/local/corex/bin/ixsmi'], env=env,
                                        capture_output=True, text=True, timeout=15)
                stream.write(status.stdout + status.stderr)
            except subprocess.TimeoutExpired:
                stream.write('telemetry timed out\n')
            stream.flush()
            stop.wait(60)


def validate(directory):
    scores = analysis.load(directory / 'benchmark_results')
    if set(scores) != {4096, 16384}:
        raise ValueError('unexpected test cases')
    raw_path = next((directory / 'benchmark_results').glob('raw_runs_*.csv'))
    with raw_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if (int(row['Decode']) != 1024 or int(row['Conc']) != 64
                    or int(row['Num Prompts']) != {4096: 256, 16384: 128}[int(row['Prefill'])]):
                raise ValueError('formal workload changed')
    return scores


def run_stage(root, stage, commit):
    directory = root / stage
    directory.mkdir(exist_ok=False)
    if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain'):
        raise RuntimeError('checkout changed after experiment launch')
    if common.exists():
        raise RuntimeError('personal GPU 1 container already exists; refuse to reuse')
    used, total = launcher.gpu_memory_used(1)
    record(directory, 'gpu_before.json', {'at': stamp(), 'gpu': 1, 'used_mib': used, 'total_mib': total})
    if used > launcher.IDLE_LIMIT_MIB:
        raise RuntimeError('GPU 1 is no longer idle; do not touch another workload')
    started = False
    container_id = None
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(directory, stop), daemon=True)
    try:
        launch = ['python3', str(ARCHIVE / 'environment/run_tianshu_container.py'), '--detach', '1']
        with (directory / 'container_launch.log').open('w') as stream:
            launched = subprocess.run(launch, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        if launched.returncode:
            raise RuntimeError('personal container launch failed')
        started = True
        container_id = command(['docker', 'inspect', '--format', '{{.Id}}', common.NAME])
        image_id = command(['docker', 'inspect', '--format', '{{.Image}}', container_id])
        digests = json.loads(command(['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', image_id]))
        if not any(digest.endswith('@' + EXPECTED_IMAGE) for digest in (digests or [])):
            raise RuntimeError('fixed image digest changed')
        record(directory, 'environment.json', {
            'started_at': stamp(), 'plugin_commit': commit, 'plugin_tree': git('rev-parse', 'HEAD^{tree}'),
            'flaggems_commit': command(['git', '-C', str(PERSONAL / 'FlagGems'), 'rev-parse', 'HEAD']),
            'flaggems_status': command(['git', '-C', str(PERSONAL / 'FlagGems'), 'status', '--porcelain']),
            'image_id': image_id, 'repo_digests': digests, 'container_id': container_id,
            'gpu': 1, 'cases': common.CASES, 'compilation': common.COMPILATION,
            'patch_sha256': sha(PREVIOUS / 'patch_vllm.py'),
            'benchmark_sha256': sha(REPO / 'benchmarks/benchmark_throughput_serve.py'),
            'launcher_sha256': sha(ARCHIVE / 'environment/run_tianshu_container.py'),
        })
        patch_path = '/workspace/vllm-plugin-FL/experiments/iluvatar/results/native2d_20261011/patch_vllm.py'
        patch = ['docker', 'exec', common.NAME, 'python3', patch_path]
        if stage == 'baseline':
            patch += ['--dry-run']
        else:
            patch += ['--evidence', f'/workspace/benchmark_results/{root.name}/{stage}/source_patch']
        with (directory / 'source_check.log').open('w') as stream:
            subprocess.run(patch, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=30)
        if stage == 'candidate':
            actual = json.loads((directory / 'source_patch/manifest.json').read_text())
            expected = json.loads((PREVIOUS / 'formal/source_patch/manifest.json').read_text())
            if actual != expected:
                raise RuntimeError('candidate source differs from the previously validated version')
        serve = ['docker', 'exec']
        if stage == 'candidate':
            serve += ['-e', 'VLLM_FL_NATIVE2D_PROBE=1']
        serve += [common.NAME, 'vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
                  '--served-model-name', 'minicpm', '--max-model-len', '131072',
                  '--gpu-memory-utilization', '0.85', '--compilation-config', json.dumps(common.COMPILATION)]
        record(directory, 'serve_command.json', serve)
        monitor.start()
        with (directory / 'serve.log').open('w') as stream:
            service = subprocess.Popen(serve, stdout=stream, stderr=subprocess.STDOUT)
            common.wait_ready(service)
            print('SERVICE READY', stage, stamp(), flush=True)
            (root / 'current_stage.txt').write_text(stage + ' formal benchmark running\n')
            bench = ['docker', 'exec', '-w', f'/workspace/benchmark_results/{root.name}/{stage}',
                     common.NAME, 'python3', '/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py',
                     '--model', '/workspace/MiniCPM5-2B', '--served-model-name', 'minicpm',
                     '--port', '9031', '--test-cases', json.dumps(common.CASES)]
            record(directory, 'benchmark_command.json', bench)
            with (directory / 'bench.log').open('w') as bench_stream:
                subprocess.run(bench, stdout=bench_stream, stderr=subprocess.STDOUT, check=True, timeout=36000)
        result = validate(directory)
        record(directory, 'validated_scores.json', result)
        print('FORMAL PASSED', stage, result, flush=True)
    finally:
        if started:
            if common.exists():
                current = command(['docker', 'inspect', '--format', '{{.Id}}', common.NAME])
                if current != container_id:
                    raise RuntimeError('container identity changed; refuse to stop it')
                common.stop_personal_container(directory)
            if common.exists():
                raise RuntimeError('personal container cleanup incomplete')
        stop.set()
        if monitor.ident is not None:
            monitor.join(timeout=20)
    (directory / 'stage_succeeded.ok').write_text(stamp() + '\n')


def compare(root):
    previous = {
        'baseline': analysis.load(PREVIOUS / 'matched_baseline_formal/benchmark_results'),
        'candidate': analysis.load(PREVIOUS / 'formal/benchmark_results'),
    }
    repeat = {name: validate(root / name) for name in ('baseline', 'candidate')}
    changes = {}
    lines = ['# Native 2D 重复性实验', '',
             '上一轮候选→基线，本轮基线→候选；同 GPU 1、同镜像、同第一版补丁及原版正式脚本。',
             '每组 4k/16k 各四轮、舍弃首轮；每组 1536/1536 请求成功且 token 总数完整。', '',
             '| 场景 | 上轮基线 | 本轮基线 | 基线跨次变化 | 上轮候选 | 本轮候选 | 候选跨次变化 | 上轮提升 | 本轮提升 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for size in (4096, 16384):
        old_b, old_c = previous['baseline'][size]['mean'], previous['candidate'][size]['mean']
        new_b, new_c = repeat['baseline'][size]['mean'], repeat['candidate'][size]['mean']
        changes[size] = {'baseline_change_pct': (new_b / old_b - 1) * 100,
                         'candidate_change_pct': (new_c / old_c - 1) * 100,
                         'previous_gain_pct': (old_c / old_b - 1) * 100,
                         'repeat_gain_pct': (new_c / new_b - 1) * 100}
        row = changes[size]
        lines.append(f"| {size} | {old_b:.2f} | {new_b:.2f} | {row['baseline_change_pct']:+.2f}% "
                     f"| {old_c:.2f} | {new_c:.2f} | {row['candidate_change_pct']:+.2f}% "
                     f"| {row['previous_gain_pct']:+.2f}% | {row['repeat_gain_pct']:+.2f}% |")
    lines += ['', '| 方案 | 场景 | 本轮计分三轮 tok/s | CV | 极差/均值 |',
              '| --- | --- | --- | ---: | ---: |']
    for name, scores in repeat.items():
        for size, row in sorted(scores.items()):
            values = '/'.join(f'{v:.2f}' for v in row['measured'])
            lines.append(f"| {name} | {size} | {values} | {row['cv']:.3f}% | {row['range']:.3f}% |")
    lines += ['', '两次独立运行及反向执行顺序用于检验可复现性；仍不能把三个计分轮的 CV 当作跨次置信区间。',
              '4k 与 16k 分别验收，不用 16k 提升掩盖 4k 回退。此次未改变候选，不重跑已通过的 Math500。', '']
    record(root, 'comparison.json', {'previous': previous, 'repeat': repeat, 'changes': changes})
    (root / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.startswith('native2d_repeat_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in args.run_id):
        parser.error('use a simple unique native2d_repeat_* name')
    root = PERSONAL / 'results' / args.run_id
    with (PERSONAL / 'native2d_gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(exist_ok=False)
        shutil.copyfile(__file__, root / 'run_repeat.py')
        commit = git('rev-parse', 'HEAD')
        code = 1
        try:
            if git('status', '--porcelain'):
                raise RuntimeError('personal checkout is dirty')
            record(root, 'run_manifest.json', {'started_at': stamp(), 'plugin_commit': commit,
                   'gpu': 1, 'order': ['baseline', 'candidate'], 'cases': common.CASES,
                   'runs_per_case': 4, 'discard_first': 1, 'new_candidate': False,
                   'runner_sha256': sha(Path(__file__)), 'expected_image_digest': EXPECTED_IMAGE})
            for stage in ('baseline', 'candidate'):
                (root / 'current_stage.txt').write_text(stage + ' starting\n')
                run_stage(root, stage, commit)
            compare(root)
            code = 0
            (root / 'current_stage.txt').write_text('complete: inspect REPORT.md\n')
            print('REPEATABILITY COMPLETE', stamp(), flush=True)
        except Exception as error:
            (root / 'current_stage.txt').write_text('failed: ' + repr(error) + '\n')
            print('ERROR', repr(error), flush=True)
        finally:
            (root / 'run.exit').write_text(str(code) + '\n')
            hashes = {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*'))
                      if path.is_file() and path.name not in {'supervisor.log', 'SHA256SUMS.txt'}}
            (root / 'SHA256SUMS.txt').write_text(''.join(value + '  ' + name + '\n' for name, value in hashes.items()))
        return code


if __name__ == '__main__':
    raise SystemExit(main())
