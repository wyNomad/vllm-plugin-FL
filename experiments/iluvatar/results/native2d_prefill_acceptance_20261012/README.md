# 第三次优化实验归档

本轮仅预填充 Native2D 插件优化在正式同卡验收中：4k +18.88%，16k +106.08%，Math500 Level3 0.9714；各侧八轮1536/1536请求成功，验收和监督退出码均0，个人容器删除。4k计分轮CV9.12%，需要独立复测；16k内部CV约0.02%，不代替跨次重复。

正式执行提交7b37daf，筛选执行197610d。REPORT.md为最终分析，formal_analysis.json保存完整计分轮、均值和波动；complete/为精选原始CSV、命令、环境与完成记录；accuracy_report/为原始精度报告；pipeline/为监督代码、日志及完成记录。installed_source_audit.json保存三个阶段5426份安装源码未改动的摘要及完整审计文件哈希。

SOURCE_SHA256SUMS.txt针对完整私有原件目录（56份），其中完整服务/基准日志、完整源码审计和回答输出未全部收入Git。原件在工作区results/native2d_prefill_acceptance_20261012/complete和服务器/workspace/wyNomad/results/native2d_prefill_acceptance_20261012。Git精选证据使用上级SHA256SUMS.txt校验。旧阶段快照与旧实验全部保留。

复现实验使用已经提交的native2d_context/run_acceptance.py和唯一新结果目录；默认实验目录已完成，禁止直接重跑。保持固定镜像、新一次性容器与同一服务参数，显式VLLM_FL_ILUVATAR_NATIVE2D=1和VLLM_FL_ILUVATAR_NATIVE2D_POLICY=prefill；不再编辑安装vLLM源码。本次仅归档，不启动下一次实验。
