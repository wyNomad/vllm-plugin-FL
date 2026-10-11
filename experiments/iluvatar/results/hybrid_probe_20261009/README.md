# 阶段路由原型第一版（非计分）

本地 fork 的独立分支 `perf/iluvatar-hybrid-attention` 引入实验开关 `VLLM_FL_USE_HYBRID_ATTN=1`：FlagGems 维护 KV 缓存并执行预填充，纯单 token 解码使用 Triton attention，通过无拷贝的维度转置视图读取同一份 KV 缓存。当前实验只改变该开关，服务参数、模型、GPU 0 和测试输入与先前 [后端 A/B](../attention_ab_20261009/README.md)一致。服务器使用个人一次性容器，未修改共享仓库。

| 负载 | 第一版 Total tok/s | 对照 Triton | 对照 FlagGems |
| --- | ---: | ---: | ---: |
| `[16384,1,1,1]` 独立预填充 | 2465.05 | 1107.07 | 2461.41 |
| `[16384,128,8,8]` 混合 | 748.81 | 818.38 | 683.65 |

各组请求全部成功，缓存命中和抢占均为 0；监督脚本与容器停止退出码均为 0。混合组平均 TPOT 为 799.94 ms、中位 ITL 为 46.53 ms。第一版保留了 FlagGems 的单请求预填充收益，并改善了纯 FlagGems 的混合吞吐，但仍比 Triton 低约 8.5%。原因推断是混合了预填充和解码的批次仍整体进入 FlagGems；因此下一版改为只有纯预填充批次使用 FlagGems。这个单轮小负载不代替赛题正式成绩，也未完成 Math500 精度验证。

完整客户端、服务、运行与自动清理日志在本目录；服务器原件位于 `/workspace/wyNomad/results/hybrid_probe_20261009/`。
