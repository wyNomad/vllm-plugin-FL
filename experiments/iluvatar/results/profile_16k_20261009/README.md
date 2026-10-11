# 16k 长输入算子画像：第一轮

2026-10-09 在天数 BI-V150 物理 GPU 0 上，使用个人一次性容器、未修改的参赛源码和此前基线的服务参数，额外启用 PyTorch profiler，采集两组单请求、非计分数据：`[16384,1,1,1]` 和 `[16384,16,1,1]`。两个请求均成功，画像脚本退出码 0；容器已自动停止并删除。Profiler 会显著增加运行开销，因此这里的客户端吞吐、TTFT 不与正式成绩比较。

| 16k／1 输出 trace 中的 GPU 算子 | GPU 时间 | 占已记录 GPU 算子时间 |
| --- | ---: | ---: |
| `kernel_unified_attention` | 13.553 s，336 次 | 92.84% |
| `mm_kernel` | 0.900 s，1344 次 | 6.16% |
| 其他算子合计 | 约 0.145 s | 约 1.00% |

这里按 trace 的 `kernel` 事件求和，总计 14.597 秒；没有把 profiler 表中的 `vllm::unified_attention_with_output` 与其内部的 `kernel_unified_attention` 重复相加。结果由 `analyze_traces.py` 生成到 `kernel_summary.json`，原始 trace 和 profiler 算子表在 `prefill_16k/`。服务日志确认当前 attention 选择为 `default.flagos`，默认返回 vLLM 的 `TRITON_ATTN`。

第二组 `[16384,16,1,1]` 的 trace 记录了 15 次 `cudaGraphLaunch`，但聚合算子表仍几乎完全由 16k 预填充占据：`kernel_unified_attention` 为 13.559 秒、92.67%。16 token 的解码样本太短，且服务使用 `FULL_DECODE_ONLY` CUDA Graph；**本轮不能据此判断正式 64 并发、1024 输出场景中的解码热点**。

**当前最有依据的优化方向是 16k 预填充 attention 内核。**仓库中另有 FlagGems attention 实现，可先做相同输入和服务参数下的非计分 A/B，核实兼容性、数值一致性和端到端吞吐；这轮画像本身不证明该实现更快。后续仍需对正式长输出并发场景单独采样，检查 KV 压力和解码路径。任何候选源码改动均须按赛题原命令复测 4k、16k 吞吐及 Math500 精度。

目录内保留 `serve.log`、各组客户端日志、profiler 表和压缩 trace、运行与清理记录。服务器副本在 `/workspace/wyNomad/results/profile_16k_20261009/`；本地结果校验值见 `../SHA256SUMS.txt`。
