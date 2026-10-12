#!/usr/bin/env python3
"""Sequential same-card baseline/prefill-only screen in disposable containers."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location

REPO = Path(__file__).resolve().parents[3]
PERSONAL = Path('/workspace/wyNomad')
spec = spec_from_file_location('migration_utilities', REPO / 'experiments/iluvatar/native2d_migration/run_validation.py')
util = module_from_spec(spec)
spec.loader.exec_module(util)


def stage(name, policy, gpu, root, commit):
    directory = root / name
    directory.mkdir(exist_ok=False)
    assert util.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD']) == commit
    assert not util.run(['git', '-C', str(REPO), 'status', '--porcelain'])
    if util.common.exists():
        raise RuntimeError('personal container already exists; refusing to reuse it')
    identity = None
    service = None
    try:
        with (directory / 'container_launch.log').open('w') as stream:
            subprocess.run(['python3', str(REPO / 'experiments/iluvatar/environment/run_tianshu_container.py'),
                            '--detach', str(gpu)], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
        identity = util.run(['docker', 'inspect', '--format', '{{.Id}}', util.common.NAME])
        image = util.run(['docker', 'inspect', '--format', '{{.Image}}', identity])
        assert image == 'sha256:f0a3e064baf7ff721e5da81aafa68c471e1f17225556d651862e6e30cadaac96'
        util.record(directory, 'environment.json', dict(commit=commit, gpu=gpu, container_id=identity,
                    image_id=image, policy=policy, started_at=util.stamp()))
        before = util.installed_sources()
        env = ['-e', 'VLLM_FL_ILUVATAR_NATIVE2D=' + ('0' if policy == 'baseline' else '1'),
               '-e', 'VLLM_FL_ILUVATAR_NATIVE2D_POLICY=' + ('native' if policy == 'baseline' else policy)]
        serve = ['docker', 'exec', *env, util.common.NAME, 'vllm', 'serve', '/workspace/MiniCPM5-2B',
                 '--port', '9031', '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
                 '--max-model-len', '131072', '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}']
        util.record(directory, 'serve_command.json', serve)
        with (directory / 'serve.log').open('w') as stream:
            service = subprocess.Popen(serve, stdout=stream, stderr=subprocess.STDOUT)
            util.common.wait_ready(service)
            print('SERVICE READY', name, util.stamp(), flush=True)
            util.correctness(directory)
            (root / 'current_stage.txt').write_text(name + ': seeded throughput running\n')
            command = ['docker', 'exec', util.common.NAME, 'python3',
                       '/workspace/vllm-plugin-FL/experiments/iluvatar/native2d_context/service_checks.py',
                       '/workspace/benchmark_results/' + root.name, name]
            util.record(directory, 'benchmark_command.json', command)
            with (directory / 'bench.log').open('w') as out:
                subprocess.run(command, check=True, stdout=out, stderr=subprocess.STDOUT, timeout=7200)
        after = util.installed_sources()
        assert before == after, 'installed vLLM sources changed'
        util.record(directory, 'source_audit.json', dict(file_count=len(before), changed=[], sha256=before))
        if policy != 'baseline':
            log = (directory / 'serve.log').read_text()
            assert 'Native 2D plugin backend loaded' in log and 'NATIVE2D_PLUGIN hit=' in log
            assert 'policy=' + policy in log
    finally:
        if identity and util.common.exists():
            assert util.run(['docker', 'inspect', '--format', '{{.Id}}', util.common.NAME]) == identity
            util.common.stop_personal_container(directory)
        if service:
            try:
                service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                service.terminate()
        if identity:
            assert not util.common.exists(), 'cleanup incomplete'
    (directory / 'stage_succeeded.ok').write_text(util.stamp() + '\n')


def compare(root):
    rows = {name: json.loads((root / name / 'summary.json').read_text()) for name in ('baseline', 'prefill')}
    changes = []
    for base, trial in zip(rows['baseline'], rows['prefill']):
        assert base['case'] == trial['case']
        change = dict(case=base['case'], throughput_change_pct=100 * (trial['total_tokens_per_s'] / base['total_tokens_per_s'] - 1),
                      ttft_change_pct=100 * (trial['mean_ttft_ms'] / base['mean_ttft_ms'] - 1),
                      tpot_change_pct=100 * (trial['mean_tpot_ms'] / base['mean_tpot_ms'] - 1))
        changes.append(change)
    util.record(root, 'comparison.json', dict(results=rows, changes=changes,
        formal_score=False, note='4k output length matches competition; 16k is still a short screening case'))
    print('SCREEN COMPARISON', json.dumps(changes), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--run-id', default='native2d_prefill_screen_20261012')
    args = parser.parse_args()
    if not args.run_id.startswith('native2d_prefill_screen_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in args.run_id):
        parser.error('invalid unique run id')
    if not 0 <= args.gpu <= 15:
        parser.error('invalid GPU')
    util.common.NAME = 'wynomad-minicpm-s2-gpu' + str(args.gpu)
    root = PERSONAL / 'results' / args.run_id
    with (PERSONAL / ('native2d_gpu' + str(args.gpu) + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(exist_ok=False)
        commit = util.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
        util.record(root, 'manifest.json', dict(commit=commit, gpu=args.gpu, policies=['baseline', 'prefill'],
                     started_at=util.stamp(), service_compilation=dict(cudagraph_mode='FULL_DECODE_ONLY')))
        code = 1
        try:
            for name in ('baseline', 'prefill'):
                (root / 'current_stage.txt').write_text(name + ': starting\n')
                stage(name, name, args.gpu, root, commit)
            compare(root)
            (root / 'current_stage.txt').write_text('complete: screening only, formal acceptance pending\n')
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
