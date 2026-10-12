# Native 2D 插件迁移验证

迁移第一版已测路由，未叠加8192上下文阈值、未修复4k回退。实现位于vllm_fl/dispatch/backends/vendor/iluvatar/impl/native2d_attention.py与native2d_ops.py；开关VLLM_FL_ILUVATAR_NATIVE2D=1，默认关闭。平台与原dispatch选择顺序保持不变，独立KV写入仍由原机制执行。

本地四项CPU选择检查通过；forward与launch函数AST和旧补丁一致（只更换开关/日志名称）。legacy_reference_ops.py只作为实验数值oracle，不用于插件运行时。

run_validation.py在个人GPU1新容器里串行验证插件版和旧补丁版，使用官网原服务命令，不再显式设置cudagraph_num_of_warmups=0。插件阶段先跑FP16/BF16、2D/3D、GQA/非目标形状共11组GPU数值测试；与旧oracle逐位一致，并与PyTorch dense参考比较。随后验证短题和16379 token确定性题，以及固定种子的warmup/4k-256输出/16k-128输出短版负载。

两阶段记录完整环境、原始日志、请求结果和vLLM所有.py源文件哈希。插件版要求源文件全部保持不变，服务日志必须证实插件backend和Native2D路径被调用。旧补丁控制容器只允许修改历史两份目标文件，结束后销毁；不触碰他人目录或容器。

服务器后台监督命令：python3 experiments/iluvatar/native2d_migration/run_validation.py --run-id native2d_plugin_20261012。必须通过脱离SSH的启动器运行，默认拒绝复用结果目录。完成看/workspace/wyNomad/results/native2d_plugin_20261012/run.exit、REPORT.md；完整日志归档个人目录。短版A/B不是比赛计分，后续仍须Math500与原版正式四轮验收。
