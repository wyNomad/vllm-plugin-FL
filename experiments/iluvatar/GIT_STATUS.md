# 代码同步与进度保留（2026-10-11）

上一轮归档分支 `perf/iluvatar-native2d-experiments` 已同步到本地、个人 GitHub fork 与服务器个人仓库 `/workspace/wyNomad/vllm-plugin-FL`。首次同步核验点为 `c1fb81df6d48da4c0da053fb63bbae52af87e376`，文件树为 `62c6366f1404826bbf23bef03143482208322c44`，服务器工作区干净。核验记录见 `sync_20261011/verification.json`。本说明随后通过同一分支快进同步；最新提交以三端实际 `git rev-parse HEAD` 和远程查询为准，核验点并非永远固定的分支最新提交。

同步前后，服务器仓库外的 437 个实验、日志和评测文件逐一校验 SHA-256，全部保持原样；9 个 Native 2D 运行脚本与 Git 内归档版本一致。未重跑实验，未修改安装的 vLLM，也未启动新容器。

## 备份与历史

服务器原诊断提交为 `11e8aecdd4d4490288555cbccf1fbcd45b460c18`，本地诊断提交为 `03a1e546944ecb99e4b2579302a34298394b22c6`。两者文件树均为 `71a77f13771cefb4bab714dea96485ace3ca844d`，源码一致但提交历史不同。原服务器历史已在两端保留为 `archive/server-attention-history-20261011`；没有改写原诊断或混合方案分支。

- 服务器备份：`/workspace/wyNomad/backups/code-sync-20261011-c1fb81d/`，含完整 Git bundle、个人工作树与实验结果压缩包、同步前后核验记录。
- 上述服务器备份已下载到本地工作区 `backups/code-sync-20261011-c1fb81d/`，验证备份 SHA-256，且压缩包内 437 个受保护文件与同步前记录一致。
- 本地同步前备份：工作区 `backups/code-sync-20261011-c1fb81d-local/`，含所有本地分支及工作树、结果和进度文件。

完整日志、预测和评审文件仍在宿主机个人目录及本地原件目录。备份包与 Git 配置不上传 GitHub。本次统一的是当前开发分支；历史分支和未验证方案保留独立状态。

| 保留分支 | 本地提交 | 实验状态 |
| --- | --- | --- |
| `perf/iluvatar-hybrid-attention` | `200bbc4` | 已测，正式吞吐下降 |
| `perf/iluvatar-attention-diagnostics` | `03a1e54` | 已测，服务器原历史保留 |
| `perf/iluvatar-native-kv-layout` | `bbb7571` | 本地准备，未上机、未推送 |
| `perf/iluvatar-mixed-prefill-threshold` | `096db82` | 本地准备，未上机、未推送 |

## 优化成果与改动位置

Native 2D 第一版已验证：同卡 4k 吞吐 −3.56%、16k +51.85%，Math500 Level 3 精度 0.981。正式分析见 `results/native2d_20261011/formal_analysis.md`。下一版 `patch_vllm_long_context.py` 只完成 dry-run，未上机，不计为成果。

修改的两份源码属于容器内安装的 vLLM 0.24：`vllm/v1/attention/backends/triton_attn.py` 与 `vllm/v1/attention/ops/triton_unified_attention.py`。Git 保存 `patch_vllm.py`、原始源码哈希、补丁 diff 和结果，容器内改动由这些脚本生成。`run_validation.py` 的 `run_stage()` 已自动启动一次性容器、校验并应用补丁、设置实验开关、评测和清理。普通容器启动器仍启动原版基线；拉取分支不会自动启用优化。未构建新镜像。

## 后续工作约定

1. 本地修改源码或补丁脚本，完成检查后提交到实验分支并推送个人 fork。
2. 服务器只在个人仓库执行 fetch 和 `git merge --ff-only origin/perf/iluvatar-native2d-experiments`，检查分支、提交及工作区；若有未提交改动或分叉，先保存并分析，不强制覆盖。
3. 每次实验创建新结果目录，记录插件提交、镜像摘要、补丁哈希和空闲卡；由已提交脚本生成容器内优化，保留原始结果。旧实验目录不直接重跑。

## 10/12 重复性验收

执行提交574c1f6，基线→候选在同GPU1复测，06:24完成，退出码0。两组各1536/1536请求成功、token完整、CSV复算和两端原件校验通过，个人容器已删除。16k重复提升51.86%（上一轮51.85%），4k回退3.27%（上一轮3.56%）。完整原件和监督日志已下载，Git归档见results/native2d_repeat_20261011。本次仅归档复测证据，不启用下一版方案；归档提交继续按当前开发分支三端快进同步。

## 当前迁移开发分支

当前为perf/iluvatar-native2d-plugin，基于18eb55c归档分支，插件实现与验证脚本执行提交e86fd22。GPU及服务级验证已经完成，结果见results/native2d_plugin_20261012；4k短版-0.53%、16k+0.04%相对旧补丁。原实验分支与结果未改写，默认不启用新后端。完整结果、监督日志和5426份源文件哈希已下载私有目录；精简验收证据提交后按同分支快进同步三端。


## 4k 吞吐恢复开发分支（10/12）

当前分支 `perf/iluvatar-native2d-context-routing`，实验执行提交197610d已在本地/fork/服务器一致。GPU数值、图回放、微基准完成并归档；端到端同GPU3基线→仅预填充方案正在运行，服务参数使用官网原配置。新增正式验收驱动等待筛选，不把内核加速当比赛成绩。运行期间服务器保持冻结执行提交；后续归档/驱动提交在本轮screen结束后通过bundle快进同步，避免运行中HEAD检查失败。旧插件迁移分支与原结果保留。
