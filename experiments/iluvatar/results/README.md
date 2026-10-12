# FlagOS S2 赛道二：实验结果索引

最新的 [Native 2D attention 实验](native2d_20261011/README.md)及同卡原生 Triton 正式基线均已完成。候选 Math500 **0.981**、105/105 成功；4k/16k 候选 **1810.35/1312.52 tok/s**，同卡基线 **1877.25/864.36 tok/s**，本轮变化 **−3.56%/+51.85%**。两边八轮请求与 token 数完整、汇总复算通过，同卡基线整体退出码 0，个人容器已清理。完整原件、SHA-256 核对与分析见 [正式分析](native2d_20261011/formal_analysis.md)。候选和基线按顺序各运行一次，仍有时间漂移的局限。下一版长上下文解码路由补丁已准备，尚未上机。

16k 长输入的第一步离线日志画像见 [analysis_16k_20261008/README.md](analysis_16k_20261008/README.md)。

后续不计分的短输出阶段诊断见 [diagnostics_16k_20261008_v2/README.md](diagnostics_16k_20261008_v2/README.md)。首版尝试因随机输入复用影响前缀缓存，已标记为无效并单独保留说明。

16k／1024 输出的同负载阶段诊断已完成并归档，结果见 [diagnostics_16k_full_20261008/README.md](diagnostics_16k_full_20261008/README.md)。

16k 长输入的第一轮 GPU 算子画像见 [profile_16k_20261009/README.md](profile_16k_20261009/README.md)：预填充 trace 中约 92.84% 的 GPU 算子时间用于 attention；此画像不参与计分。

attention 后端的非计分 A/B 见 [attention_ab_20261009/README.md](attention_ab_20261009/README.md)：FlagGems 在单请求 16k 预填充快约 2.22 倍，但 8 并发混合负载总吞吐下降约 16.5%，因此暂不直接切换默认后端。

后续的[解码隔离对照](decode_ab_20261009/README.md)证实 FlagGems 的长输出 TPOT 较高；[阶段路由第一版](hybrid_probe_20261009/README.md)只恢复了部分混合吞吐。[第二版](hybrid_mixed_v2_20261009/README.md)将含解码的混合批次交给 Triton，在非计分小负载中保留单请求预填充约 2.22 倍的优势，并将混合吞吐从默认的 818.38 提至 866.16 tok/s。但[首次 Math500 精度复测](hybrid_accuracy_20261009/README.md)只得到 0.0095，输出普遍乱码；该轮性能数字不能代表有效优化。[单题定位及16k对照](hybrid_correctness_20261009/README.md)发现自定义后端遗漏 KV cache 独立写入声明，修复后 FlagGems 与第二版混合路由恢复正常回答，16,379 token 单请求也给出正确答案。[修复版完整 Math500](hybrid_accuracy_fixed_20261009/README.md) 得到 **0.9714**，105/105 请求成功，达到赛题 0.95 门槛。但[正式吞吐复测](hybrid_formal_20261009/README.md)的 4k/16k Total tok/s 分别为 **1880.12/866.21**，比原 Triton 基线低 **4.91%/5.65%**；当前候选精度通过但性能不通过，不切换默认后端。

[混合方案退化隔离诊断](hybrid_diagnostics_20261010/README.md)已完成五组服务级短版 A/B。短版 16k 混合方案较原生 Triton 高 1.31%，与正式 16k 低 5.65% 的方向相反；单独换 KV 写入核函数无一致收益。原始汇总、运行与容器清理日志已同步，五份汇总及完成标记均与服务器原件校验一致。下一轮尚未启动，两个布局/路由实验分支均未上机验证。

这是一轮**未修改参赛源码**的 MiniCPM5-2B 基线。它用于后续优化前后比较，不代表性能优化成绩。测试使用 `vllm-plugin-FL` 的 `flagos-2026-s2` 提交 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，以及 FlagGems v5.3.5 提交 `a7620cc191a0b42e040194622c5758b22a7a25dc`。基础镜像为 `harbor.baai.ac.cn/plugin/iluvatar-corex4.5.0-flagtree0.6.0-triton3.6.0-cxnone-vllm_fl0.24.0@sha256:cc8ed9969977a06996c86c483851567f212ab9761bb9147e85082b4a34ce88b8`。

## 正式吞吐基线

2026-10-07 晚至 2026-10-08 03:00，在物理 GPU 0 上按赛题固定的 `vllm serve` 参数运行。原版 `benchmarks/benchmark_throughput_serve.py` 每组运行四轮、舍弃首轮；八轮请求均成功，脚本退出码 0。

| 场景 `[输入,输出,并发,请求数]` | 计分轮次平均 Total tok/s | 平均 TTFT |
| --- | ---: | ---: |
| `[4096,1024,64,256]` | 1977.17 | 11616.10 ms |
| `[16384,1024,64,128]` | 918.04 | 598124.12 ms |

- 本地原始轮次和汇总：`baseline_20261007/raw_runs_20261007_232244.csv`、`baseline_20261007/summary_20261007_232244.csv`。
- 本地服务、基准与自动清理日志：`baseline_20261007/logs/`。
- 服务器原始路径：`/workspace/wyNomad/results/` 下同名 CSV，`/workspace/wyNomad/logs/serve_baseline_20261007_231136.log`、`bench_baseline_20261007_2323.log`、`.exit` 与 `stop_after_baseline.log`。

## 正式精度基线

2026-10-08 16:41–17:14，在空闲的物理 GPU 3 上使用赛题 EvalScope 命令测试 `/workspace/evalscope-datasets/math_500` 的 `Level 3` 子集。报告准确率 **0.981（98.1%）**，高于赛题门槛 0.95；执行摘要显示请求 105、成功 105、错误 0、无未完成项，脚本退出码 0。评测使用 `temperature=1.0`、`top_p=0.95`、`max_tokens=32768`、批量 8，与赛题参数一致。

- 本地完整配置、逐题预测、评审及报告：`eval_math500_20261008/`。可打开 `eval_math500_20261008/reports/report.html`，机器可读得分在 `reports/minicpm/math_500.json`。
- 本地服务、评测与自动清理日志：`eval_math500_20261008/runtime_logs/`。
- 服务器原始路径：`/workspace/wyNomad/evalscope-results/20261008_164114/` 和 `/workspace/wyNomad/logs/` 下的 `serve_accuracy_20261008_1637.log`、`eval_math500_level3_20261008_1641.log`、`.exit` 与 `stop_after_eval.log`。

## 连通性测试

`smoke_20261007_231959/` 是 `[128,16,1,1]` 的小规模请求测试，只用于验证服务和基准脚本连通；**不参与正式成绩**。服务器副本位于 `/workspace/wyNomad/results/smoke_20261007_231959/`。

`SHA256SUMS.txt` 记录本地归档文件校验值。正式吞吐与精度的八个核心结果文件、短输出诊断的三个核心文件、同负载诊断的四个核心文件，以及最新五组服务级诊断汇总和完成标记，均已与服务器原件逐一核对，SHA-256 一致。相关个人一次性容器均已自动停止并删除。


2026-10-12 第三次优化正式验收完成：仅预填充Native2D插件方案同GPU3，4k 1980.92→2354.83 tok/s（+18.88%），16k 917.63→1891.09（+106.08%），Math500 Level3 0.9714。3072/3072请求成功，完整56份原件SHA校验通过，个人容器删除。完整原件、分析和波动限制见native2d_prefill_acceptance_20261012/；筛选结果单列native2d_prefill_screen_20261012/，不得混作正式计分。精选证据收入插件仓库experiments/iluvatar/results/下同名目录。
