# 修复 KV cache 写入声明后的 Math500 Level 3 精度复测

本轮使用本地实验分支提交 `200bbc4`（服务器个人仓库对应提交 `89192e8`）及第二版混合路由。相对[失败的上一轮](../hybrid_accuracy_20261009/README.md)，唯一生产源码修复是在 `AttentionFLBackend` 声明 `forward_includes_kv_cache_update = False`，使 vLLM 0.24 调用后端已有的 `do_kv_cache_update()`。服务模型、启动参数、EvalScope 数据集及生成参数均与原基线保持一致，结果写入新目录，不覆盖失败证据。

`run_and_cleanup.py` 监督评测并在结束后检查：105/105 请求成功、0 错误、无未完成项，以及准确率至少 0.95。2026-10-09 23:24 的完整报告给出 **0.9714（97.14%）**，超过赛题 0.95 门槛；预测和评审各有 105 行。原始报告、配置、逐题记录、HTML 报告及日志已从服务器完整同步到 `20261009_225607/`，`report_bundle.tar.gz` 是同一报告目录的压缩副本。

`supervisor.log` 显示 EvalScope 退出码 0、精度检查通过。`runner.exit` 记录 1，仅因 `docker stop` 超过脚本的 120 秒等待；`container_stop.log` 保留原始超时证据。之后实际检查 `docker ps -a --filter name=wynomad-minicpm-s2-gpu0` 无结果，个人一次性容器已删除。后续门控脚本依据精度标记、评测标记及实时容器不存在继续，不改写这份历史退出码。相对默认 Triton 基线 0.981，本轮少约 0.0096；两次均高于门槛，不能仅凭两轮分数归因性能或数值差异。
