# Native 2D 插件迁移验收（10/12）

执行提交e86fd22，12:30:31整体退出码0。第一版forward/launch策略已迁入插件，未启用8192阈值。默认关闭，通过VLLM_FL_ILUVATAR_NATIVE2D=1启用，原dispatch/KV写入/metadata及回退机制保留。

四项CPU检查、函数AST等价核验、11组GPU数值测试、两侧逐字一致的短题/16379token长题、同GPU1短版性能A/B全部通过。每侧97/97计数请求成功，token完整，缓存命中和抢占均0。插件5426份vLLM.py源文件全部未变；实际backend和路由日志核实，个人容器已清理。

4k/256输出/64并发：旧补丁1923.70、插件1913.51 tok/s（-0.53%）；16k/128输出/32并发：旧补丁2596.10、插件2597.26（+0.04%）。两侧均使用官网原服务配置，不再显式配置cudagraph_num_of_warmups=0。短版各一次支持迁移没有明显性能损失，不代表比赛正式成绩。Math500和原版四轮仍待后续，原先4k正式回退尚未修复。

REPORT.md与comparison.json保存对照；installed_source_audit.json和completion_verified.json保存复算、源码完整性及验收摘要。两侧原始短测日志以.txt原字节归档。完整源文件哈希清单、服务/监督日志、响应和旧控制补丁原件保留在服务器个人结果目录及本地results/native2d_plugin_20261012/complete。SOURCE_SHA256SUMS.txt是完整原件目录的清单，不能在此精简目录直接执行；Git归档使用上层SHA256SUMS.txt。
