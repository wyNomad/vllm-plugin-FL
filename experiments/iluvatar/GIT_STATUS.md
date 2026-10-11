# Git 核验（2026-10-11）

本地诊断分支与个人 fork 的 `perf/iluvatar-attention-diagnostics` 均为 `03a1e546944ecb99e4b2579302a34298394b22c6`，通过实际远程查询确认。

服务器最初核验的诊断分支是 `11e8aecdd4d4490288555cbccf1fbcd45b460c18`，父提交为 `89192e8ba81713a1222f76f00f0b70bba863e937`，提交说明与本地相同，但历史不同。两端文件树均为 `71a77f13771cefb4bab714dea96485ace3ca844d`，所以源码内容一致，不能仅凭提交 SHA 不同断言有源码改动。服务器工作区干净。

已发起对齐操作：先核对个人 fork 的诊断提交与原文件树，再将服务器原有诊断分支保留为 `archive/server-attention-history-20261011`，切回 fork 对应的诊断提交。操作受 SSH 新通道间歇超时影响，完成状态仍待远程确认，不能视为已经对齐。

最新 Native 2D 及历次实验原来在仓库外，现在已归档到本地独立 `perf/iluvatar-native2d-experiments` 分支，基于本地诊断分支 `03a1e54`。此分支收录补丁脚本、环境配方、进度、吞吐 CSV、精度摘要、原件哈希及分析；没有改变插件默认后端实现。提交后尚未推送个人 fork，服务器也尚未部署这个新归档分支。

原分支均保留：

| 分支 | 本地提交 | 上机状态 |
| --- | --- | --- |
| `perf/iluvatar-hybrid-attention` | `200bbc4` | 已测，正式吞吐下降 |
| `perf/iluvatar-attention-diagnostics` | `03a1e54` | 已测，同文件树服务器历史待对齐 |
| `perf/iluvatar-native-kv-layout` | `bbb7571` | 本地准备，未上机、未推送 |
| `perf/iluvatar-mixed-prefill-threshold` | `096db82` | 本地准备，未上机、未推送 |

Native 2D 第一版已在个人一次性容器内验证，结果见 `results/native2d_20261011/formal_analysis.md`；下一版长上下文路由仅准备完成，未上机。
