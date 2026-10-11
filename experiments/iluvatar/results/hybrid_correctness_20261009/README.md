# 混合注意力数值错误定位（单题诊断，非计分）

使用同一张天数 BI-V150 GPU 0、同一镜像、模型和服务参数，每组都启动新的个人一次性容器，对同一道 Math500 Level 3 距离题发送 `temperature=0`、`seed=42`、最多 128 token 的请求。正确答案为 `3\sqrt{13}`。

| 后端或隔离配置 | 回答 | 作用 |
| --- | --- | --- |
| 默认 Triton | 正常，推导出 `3\sqrt{13}` | 排除 GPU 0、模型及镜像整体故障 |
| 原始 FlagGems | 乱码 | 复现 Math500 失败 |
| 原始混合后端，强制 Triton 前向 | 与原始 FlagGems 乱码逐字相同 | 排查前向路由 |
| 原始混合后端，强制 Triton KV 写入及前向 | 仍为相同乱码 | 排除单一 FlagGems 写入核函数的解释 |
| 修正 KV 写入声明后的 FlagGems | 正常，推导出 `3\sqrt{13}` | 验证根因修复 |
| 修正 KV 写入声明后的第二版混合路由 | 正常，推导出 `3\sqrt{13}` | 验证无临时诊断开关的候选源码 |

路由追踪证实自定义混合后端在 58 token 预填充时确实进入 Triton，但修复前没有 KV 写入调用。当前镜像的 vLLM 0.24 `AttentionBackend.forward_includes_kv_cache_update` 默认是 `True`，会跳过单独的 `do_kv_cache_update()`；而本仓库的 `AttentionFLImpl.forward()` 只读 cache，原本期待该方法先写入。因此自定义后端一直在用未正确写入的 cache。已在 `AttentionFLBackend` 将该属性声明为 `False`。修复后的 FlagGems 与第二版混合路由单题均恢复正常；完整 Math500 精度见 [修复版报告](../hybrid_accuracy_fixed_20261009/README.md)。

`flaggems/`、`triton/`、`hybrid_*`、`flaggems_fixed/` 保存响应与服务/清理日志。`hybrid_attention_diagnostic.py` 是定位时临时加路由、写入开关和日志的源码快照；正式实验分支已恢复无诊断开关的第二版混合路由。`probe.py` 中诊断模式只有配合该临时源码才会生效，不用于正式成绩。

后续 16k 单请求对照的第一次尝试无效：`long_client.py` 曾把 tokenizer 返回对象的字段数误当 token 数（日志显示 `input_tokens=2`），导致填充到 10,000 次、请求超时。原始证据已在服务器个人目录备份为 `triton_long_invalid_prompt/`，不计入 16k 对照。客户端现改为先渲染 chat template、再对实际文本编码计数，并在发请求前强制校验 16,000–16,384 token。

修正后，`triton_long/` 与 `hybrid_long/` 在同一 GPU 0、相同服务参数和同一条确定性请求下均实际接收 **16,379 token**，各自回答 `4`，请求成功且个人容器均正常清理。两组完整响应、客户端日志、服务日志和退出标记已同步本地。这验证了长输入功能与基本输出正确性；它不是正式吞吐分数，也不证明逐 token 数值完全一致。
