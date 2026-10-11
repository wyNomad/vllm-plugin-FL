#!/usr/bin/env python3
"""Validate full official CSVs and compare measured rounds after warmup."""

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD_BASELINE = {4096: 1977.17, 16384: 918.04}


def load(directory):
    raw_paths = list(directory.glob('raw_runs_*.csv'))
    summary_paths = list(directory.glob('summary_*.csv'))
    if len(raw_paths) != 1 or len(summary_paths) != 1:
        raise ValueError(f'{directory}: expected unique raw and summary CSVs')
    raw = list(csv.DictReader(raw_paths[0].open(newline='')))
    summary = list(csv.DictReader(summary_paths[0].open(newline='')))
    if len(raw) != 8 or len(summary) != 2:
        raise ValueError(f'{directory}: incomplete eight-round benchmark')
    by_prefill = {}
    for row in raw:
        prefill, decode = int(row['Prefill']), int(row['Decode'])
        prompts = int(row['Num Prompts'])
        if (row['Run Status'] != 'SUCCESS'
                or float(row['Successful Requests']) != prompts
                or float(row['Total Input Tokens']) != prefill * prompts
                or float(row['Total Output Tokens']) != decode * prompts):
            raise ValueError(f'{directory}: failed or incomplete token counts')
        by_prefill.setdefault(prefill, []).append(float(row['Total tok/s']))
    output = {}
    for row in summary:
        prefill = int(row['Prefill'])
        values = by_prefill[prefill]
        if len(values) != 4:
            raise ValueError('expected four runs per workload')
        measured = values[1:]
        mean = statistics.mean(measured)
        if abs(mean - float(row['Total tok/s'])) > 0.02:
            raise ValueError('summary does not match the last three rounds')
        output[prefill] = {
            'warmup': values[0], 'measured': measured,
            'mean': mean, 'cv': statistics.stdev(measured) / mean * 100,
            'range': (max(measured) - min(measured)) / mean * 100,
            'ttft': float(row['Mean TTFT (ms)']),
            'tpot': float(row['Mean TPOT (ms)']),
        }
    return output


def main():
    candidate = load(ROOT / 'formal/benchmark_results')
    matched_path = ROOT / 'matched_baseline_formal/benchmark_results'
    matched = load(matched_path) if matched_path.exists() else None
    lines = [
        '# Native 2D 正式结果分析', '',
        '八轮均成功；输入、输出 token 总数完整；汇总与各场景后三轮均值一致。', '',
        '| 场景 | 预热轮 | 计分三轮 tok/s | 平均 tok/s | 样本 CV | 极差/均值 | 对旧基线 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for prefill, row in sorted(candidate.items()):
        values = '/'.join(f'{v:.2f}' for v in row['measured'])
        change = (row['mean'] / OLD_BASELINE[prefill] - 1) * 100
        lines.append(
            f"| {prefill} | {row['warmup']:.2f} | {values} | {row['mean']:.2f} "
            f"| {row['cv']:.3f}% | {row['range']:.3f}% | {change:+.2f}% |")
    lines += ['', '旧正式基线与本候选跨日期、跨 GPU。三个计分轮只能描述本次运行内部的波动，不能作为跨次运行误差或置信区间。', '']
    if matched:
        lines += ['| 场景 | 同卡原生 Triton | Native 2D | 变化 |',
                  '| --- | ---: | ---: | ---: |']
        for prefill, row in sorted(candidate.items()):
            reference = matched[prefill]['mean']
            change = (row['mean'] / reference - 1) * 100
            lines.append(f"| {prefill} | {reference:.2f} | {row['mean']:.2f} | {change:+.2f}% |")
        lines += ['', '同卡对照按候选→基线顺序串行运行，仍需考虑时间漂移；未做随机交错重复。', '']
    else:
        lines += ['同卡原生 Triton 正式基线尚待完成。', '']
    (ROOT / 'formal_analysis.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
