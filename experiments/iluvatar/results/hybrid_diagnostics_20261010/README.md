# 2026-10-10：混合注意力退化定位

正式测试中，修复 KV cache 写入后的混合方案虽然 Math500 达到 0.9714，但 4k / 16k 吞吐分别比原生 Triton 低 4.91% / 5.65%。本目录用于区分路由、缓存布局、KV 写入的成本，所有运行均为诊断，不计入比赛分数。

## 已核实事实

- 当前镜像中的 Triton 与自定义 FlagGems 注意力都把 KV 写入放在 `do_kv_cache_update` 中，每步只调用一次；之前怀疑的“双重写入”并不存在。
- 混合方案采用 FlagGems 的 `[K/V, blocks, slots, heads, dim]` 物理缓存布局；Triton 分支通过转置视图读写。原生 Triton 布局为 `[blocks, K/V, slots, heads, dim]`。
- vLLM 的分块预填充可以把长输入和解码放在同一批次。当前混合路由只在整批没有解码 token 时才选 FlagGems。单请求预填充性能不能代表正式混合负载的吞吐。参见 [vLLM 官方优化说明](https://docs.vllm.ai/en/stable/configuration/optimization/)。

## 已做的核函数诊断

`bench_kv_update.py` 在物理 GPU 1 的一次性容器中测试 64 和 2048 token 的 KV 写入，结果在 `kv_kernel_results.json`。三种写法的缓存内容逐元素相等。64-token 结果分别约 0.113 ms（FlagGems 布局与写入）、0.151 ms（FlagGems 布局与 Triton 写入）、0.070 ms（Triton 原生布局与写入）。2048-token 结果分别约 0.048、0.061、0.054 ms。仅测一轮且顺序固定，64-token 反而比 2048-token 慢，说明存在显著测量波动；这些值仅供制定后续 A/B，不能作为退化归因。

## 服务级 A/B

`run_service_ab.py` 在物理 GPU 1 串行启动并自动清理我们的 `wynomad-minicpm-s2-gpu1` 一次性容器；启动器会拒绝已被使用的卡。与正式测试相同的服务参数是 `--max-model-len 131072`、`--gpu-memory-utilization 0.85`、`FULL_DECODE_ONLY`，单次基准为固定种子的随机输入。`bench_service_ab.py` 测 4k/256/64 并发与 16k/128/32 并发，每种方案先做一条 4k 暖机请求。记录吞吐、TTFT、TPOT、缓存命中与抢占，全部请求必须成功。

诊断分支已推送到个人 fork 的 [`perf/iluvatar-attention-diagnostics`](https://github.com/wyNomad/vllm-plugin-FL/tree/perf/iluvatar-attention-diagnostics)。服务器个人仓库为同名分支、相同 Git tree；两个诊断开关默认关闭。

比较顺序：原生 Triton、原混合方案、强制 Triton 路由但保留 FlagGems KV 写入、强制 Triton 路由并改用 Triton KV 写入、原混合路由加 Triton KV 写入。五组已完成；两个诊断开关默认关闭。真正的优化还需要 Math500 ≥ 0.95 以及原版 4k/16k 正式脚本复验。

## 待验证的物理布局改法

已推送个人 fork 的实验分支 [`perf/iluvatar-native-kv-layout`](https://github.com/wyNomad/vllm-plugin-FL/tree/perf/iluvatar-native-kv-layout)（提交 `bbb7571`）加入默认关闭的 `VLLM_FL_HYBRID_NATIVE_TRITON_LAYOUT=1` 实验开关。开启时按 Triton 原生 `[blocks, K/V, slots, heads, dim]` 分配缓存，Triton 路由直接使用原生布局，FlagGems 路由通过转置视图访问。源码检查确认本机 FlagGems 的 `reshape_and_cache_flash` 使用 `key_cache.stride(0)`，`flash_api.mha_varlan_fwd` 也使用分页缓存的实际步长；这说明从接口上有可行性，但尚未在 GPU 上验证数值正确性或吞吐。当前本地工作区已切回与服务器一致的诊断分支，实验分支尚未同步到服务器个人源码，不能把它视为已有提速结果。

备选验证脚本 `run_native_probe.py` 会同时启用原生 Triton 物理布局与已有的 Triton KV 写入开关；这样写入和大多数解码都使用同一种原生布局，FlagGems 只在预填充路由时通过视图访问。脚本设计为先做两道确定性正确性检查，再运行相同的 4k/16k 短版吞吐对照。五组隔离实验已经结束，但按当前约定暂停下一轮，**该脚本尚未执行**。布局补丁仅放在服务器个人结果目录，尚未应用到服务器源码。

另一个已推送个人 fork 的实验分支 [`perf/iluvatar-mixed-prefill-threshold`](https://github.com/wyNomad/vllm-plugin-FL/tree/perf/iluvatar-mixed-prefill-threshold)（最新 `096db82`）加入默认关闭的 `VLLM_FL_HYBRID_PREFILL_MIN_CONTEXT`。本机 vLLM 0.24 的源码在该规格 GPU 的 OpenAI 服务给出默认每轮 token 预算 2048，服务启动命令没有显式覆盖；尚未从运行时配置日志独立确认实际值。因此当前查询块长度很可能无法区分 4k 与 16k 输入；该开关改按累计上下文长度选择 FlagGems，纯解码仍走 Triton。它用于测试长上下文预填充（包括混合批次）是否能取得净收益；阈值尚未选定，也没有上机验证。两个实验分支互不叠加，不能据此宣称精度或性能收益。

服务器原件位于 `/workspace/wyNomad/results/hybrid_diagnostics_20261010/`，后台运行记录为 `supervisor.log`、`runner.exit`、各变体目录下的 `summary.json` 与完整日志。物理 GPU 0 被其他作业占用，本轮不使用 GPU 0。

五组单轮结果如下，单位为 Total tok/s；均不计分：

| 变体 | 4k/256 输出/64 并发 | 16k/128 输出/32 并发 |
| --- | ---: | ---: |
| 原生 Triton | 1558.64 | 993.12 |
| 原混合路由、FlagGems KV 写入 | 1526.26 | 1006.17 |
| 强制 Triton 路由、FlagGems KV 写入 | 1548.71 | 991.22 |
| 强制 Triton 路由、Triton KV 写入 | 1523.04 | 992.01 |
| 原混合路由、Triton KV 写入 | 1552.83 | 1007.96 |

每组先运行一条 4k/32 输出暖机请求，再分别处理 64/32 条正式诊断请求。五组所有请求成功，失败、前缀缓存命中和抢占均为 0；各组 `container_stop.log` 均记录退出码 0。根目录 `runner.exit=0`、`finished.ok=all variants passed`，`supervisor.log` 从首组到末组均记录 `DONE`。服务器个人容器已经自动删除。本地五份 `summary.json`、根目录退出/完成标记的 SHA-256 已与服务器原件逐一核对一致。

## 结果解释与局限

- 与原生 Triton 比，原混合方案的短版 4k 低 2.08%，短版 16k 高 1.31%；正式 16k/1024 输出却低 5.65%。短版同时改变了输出长度、并发和请求数，不能把差异单独归因于某一项。正式 Triton/混合服务日志各有 799/849 个十秒采样点达到或超过 90% KV 占用，峰值均为 100%；本轮五组峰值最多 53.2%，没有任何采样达到 90%。短版缺少正式负载的高占用阶段，这是外推的主要限制之一。
- 强制 Triton 路由而保留 FlagGems 布局/写入时，短版 16k 从混合的 1006.17 降至 991.22 tok/s，TTFT 从 242665.91 升至 250599.29 ms；与 FlagGems 预填充在该短版 16k 有净收益相符。4k 则从混合的 1526.26 升至 1548.71 tok/s。由此看，固定的“纯预填充走 FlagGems”规则不能同时照顾两种短版负载；但单轮差值仍可能受时间漂移影响。
- 在同一 FlagGems 物理布局上把写入核函数换为 Triton，强制 Triton 路由下的 4k 下降 1.66%、16k 上升 0.08%；混合路由下的 4k 上升 1.74%、16k 上升 0.18%。方向不一致且幅度小，不能宣称 KV 写入替换已提速，也不能由此推断原生 Triton 物理布局的效果。
- 源码核对排除了“双重 KV 写入”解释。混合方案每批都建立 FlagGems 元数据，选 Triton 时还建立 Triton 元数据并给 KV cache 创建转置视图；这些可能带来开销，但现有 A/B 尚不能将其与物理布局、内核执行或调度差异分开量化。五组固定顺序、每组一轮，也没有新变体的 Math500 验证。

目前只保留两个待验证的源码假设：让混合路由使用原生 Triton 物理 KV 布局；或者按累计上下文长度决定混合批次是否走 FlagGems。两者在独立分支，均未上机或证明提速。下一轮启动前应先约定与正式负载一致的对照、重复次数和正确性门槛；当前**暂停新实验**。
