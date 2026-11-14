# ObjectNav 实验版本与数据名称

2026-11-14 对齐。此表为今后实验简称的依据；“当前版”“旧版”“提前版”不再单独作为数据标签。分支用于组织开发，**实际实验版本由方法配置、运行源码指纹和固定 episode 集共同确定**。

## 主要版本

下表分支均为 `codex/` 前缀。Q1 的归档分支从已提交的 `6829be9` 建立，其 160 个运行文件已与原前 50 回合源码快照逐字节核对。

| 简称 | 固定称呼 | 主要机制 | 对应分支 |
| --- | --- | --- | --- |
| **B0** | **原生基线** | 原 ApexNav 决策与 BLIP2 ITM；关闭新增 Qwen 复查；旧异步观测流程 | `codex/objectnav-sim-to-tracer-mini` |
| **Q1** | **原始复查版** | 所选低置信候选／STOP 条件触发 Qwen；同簇证据 crop；保留原追加观测和高置信 STOP 资格；旧异步流程 | `codex/objectnav-qwen-original-review` |
| **Q2** | **同步复查版** | 在 Q1 上加入 `step_sync`，同步本帧观测与规划；修正 400 步后不确定候选被提升为高置信的问题；保留条件追加观测 | `codex/objectnav-review-evidence-fix` |
| **Q3** | **单帧提前复查版（0.65）** | `linked_early`；单帧检测分数 ≥0.65 自动触发，发送同候选最佳可用 crop；不确定后自然行进取得新视角；关闭额外观测绕行和旧 STOP 复查 | `codex/objectnav-early-vlm-review` |
| **Q4** | **融合提前复查＋主动观测版** | `linked_fused`；原类别融合阈值触发，发送与地图候选绑定的最高单帧分数可用 crop；首次未通过且无其他已确认、可达目标时，规划安全观测点、执行补拍动作并二次复查 | `codex/objectnav-fused-vlm-active-observation` |
| **Q5** | **朝向后复查＋置信度加权版** | `linked_confidence`；沿用 Q4 融合触发和条件安全观测；二次调用等待到点朝向中心后的真实观测，仍用该对象全历史最高检测分数 crop；Qwen 返回类别置信度，400 步后未确认候选按 0.5 融合＋0.5 Qwen 排序，未调用者保持融合分数 | `codex/objectnav-facing-confidence-review` |

**用户当前要求的方法是 Q5，固定重跑 Q4 演示的第 10 和 885 两例；旧批次保持暂停。** Q3 的“单帧 0.65”和 Q4 的“原类别融合阈值”是两种不同触发机制。Q4 仍需要寻找观测点位；额外观测可能是转向、平移，也可能因无安全点位而未执行，须根据真实动作分别统计。

B0 的“原生”指基线算法行为，并不表示当前 ROS2 适配分支没有其他可选实验功能。原生参考提交为 `3828f38`；历史 B0 数据实际使用各自来源记录中的源码，并显式关闭新增复查。不能仅用分支 HEAD 代替数据来源。

## 阈值与图片的约定

| 类别 | 原类别融合阈值（Q4/Q5 沿用） |
| --- | ---: |
| bed | 0.65 |
| couch | 0.60 |
| chair | 0.50 |
| tv | 0.45 |
| toilet | 0.45 |
| potted plant | 0.40 |

地图原有类别、有效观测次数等资格检查仍保留。融合分数决定 Q4 首次复查资格；单帧检测分数决定输入哪张可用 crop，两者分开。图片归属使用 `(episode_token, candidate_id)` 及原 mask 的地图关联回执，不使用真值绑定。Q4 二次复查取追加观测产生的合格新视角中最佳图，不能反复发送旧最高分图。**Q5 有意改变此项：朝向后再复查，但输入仍是对象全历史最高检测分数图，允许复用旧图；不得混称 Q4 数据。**

## 已有数据归属

路径均相对仓库根目录；只新增别名，不移动、改写或混合原始结果。`gNNNN` 是固定 1000 清单的全局编号，单例 run 内部的 episode 1 不能当成全局编号。

| 数据简称 | 版本与集合 | 已核验结果／状态 | 本地原始目录 |
| --- | --- | --- | --- |
| `B0-paired50` | B0，同固定前 50 | 38/50，76.00% | `outputs/runs/qwen-paired50-20261110-baseline/` |
| `Q1-paired50` | Q1，同固定前 50 | 42/50，84.00% | `outputs/runs/qwen-paired50-20261110-qwen/` 的前 50；冻结前缀见 `experiment_notes/qwen-fullval-20261111/before-extension/` |
| `Q1-prefix197` | Q1，后续冻结的前 197 | 133/197，67.51%；198 未完成不计成绩 | `outputs/runs/qwen-paired50-20261110-qwen/` |
| `B0-paired13` | B0，全局 185–197 | 7/13；Q1 同集合 5/13 | `outputs/runs/baseline13-20261112-gNNNN/`；195 有独立 GPU 恢复目录，以该组配对清单为准 |
| **`Q2-full1000`** | Q2，完整 1000 | **757/1000，75.70%**；SPL 37.5471%、SoftSPL 39.8402%；整体失败分析暂缓 | `outputs/runs/qwen-sync-fullval-20261112/`；汇总见 `experiment_notes/qwen-sync-fullval-20261112/new-results.csv` |
| `Q2-demo73` | Q2，第 73 的独立录像重跑 | 72 步成功；49/65 步复查，16 个追加观测动作；不并入全量 | `outputs/runs/qwen-vlm-review-demo-20261114-g0073/` |
| `Q3-demo73` | Q3，第 73 的独立录像重跑 | 50 步成功；首次 18 步复查，0 追加观测动作 | `outputs/runs/qwen-early-review-20261114-g0073/` |
| **`Q3-first30`** | Q3，原清单前 30 | **29/30，96.67%**；Q2 同集合 28/30；Q3 没有额外绕行 | `outputs/runs/qwen-early30-20261114/` |
| **`Q4-fail30`** | Q4，Q2-full1000 的失败按全局编号取前 30 | **暂停：完成 16/30，恢复成功 2/16**；不是完整集成功率 | `outputs/runs/qwen-fused-fail30-20261114-offline/`；有效记录在 `experiment_notes/qwen-fused-fail30-20261114/init-offline/` |
| `Q4-tvplant4` | Q4，预选 111盆栽、122电视、174盆栽、237电视 | 0/4 完成；内存不足停止。首尝试 111 未完成、后一次导航前预检停止；均无 Qwen 调用 | `outputs/runs/qwen-fused-tv-plant-demo-20261114/` 和 `outputs/runs/qwen-fused-tv-plant-demo-20261114-after-cleanup/` |
| `Q4-demo2` | Q4，chair/tv/potted plant目的性小池 | 完成4例后停止；英文选例10 chair（1复核、0主动观测）与885 tv（5复核、6动作、4平移、导航成功但回答均不确定）；不推断总体SR | `outputs/runs/q4-two-demos-20261114-25g/`，观看版`presentation-en/`；见[记录](experiments/2026-11-14-q4-two-demos.zh-CN.md) |

| `Q5-demo2` | Q5，固定重跑第 10 chair 与 885 tv | 2/2 成功，197/77 步；2/3 次复核，20/39 观测动作；均在 400 步前结束，加权排序仅离线验证 | `outputs/runs/qwen-facing-confidence-20261114/`；见[记录](experiments/2026-11-14-qwen-facing-confidence.zh-CN.md) |

`Q4-fail30` 是旧失败子集，`Q3-first30` 是顺序前 30，不能直接比较两者百分比。Q1 前 50 对照曾在首次复查前出现轨迹分歧；它不是固定轨迹下只替换 VLM 的因果消融。Q2-full1000 也不能与 B0-paired50 直接作全量公平对照。

## 源码与分支保存状态

截至本次对齐，Q2、Q3、Q4 的实际运行实现仍包含未提交文件。下表诚实区分 Git 提交与本地实验快照；本次命名登记不会将文档提交冒充算法实现提交，也不会改变旧 provenance 的 `uncommitted=true`。

| 版本 | Git 状态 | 精确运行来源 |
| --- | --- | --- |
| B0 | 原工作分支停在 `e25740f`；原生参考 `3828f38` | B0-paired50 与 Q1-paired50 使用同一 160 文件快照，配置关闭复查；其他 B0 实验按各自 provenance |
| Q1 | 独立归档分支 `6829be9` 与 paired50 的 160 运行文件一致 | 前 50：`99f2fb75f4e38b9bdbd26f8656e6bad437099ee15b1ce20914068170af5c4b82`；后续还有运行修订，见下文 |
| Q2 | `review-evidence-fix` 分支仍在基底 `6829be9`；全量实现尚未提交 | 170 文件：`295b50a412301920968057b7dc731f1c38bcb593999bf2083277c7ebcdcfe96e` |
| Q3 | `early-vlm-review` 本地分支仍在基底 `6829be9`；实现尚未提交、分支尚未发布 | 172 文件：`5b561ad7d77359bf01fe40915e2be24add842d61616b8e195949a635edb5ab7b` |
| Q4 | `fused-vlm-active-observation` 的 `a7718ad` 为方案文档；本次登记也是文档改动，运行实现尚未提交 | 174 文件：`57c9b71c4a7424c648fbcbca93e677a7860b044e16a00d670ec3e0567e8651f2` |

| Q5 | 本次新分支提交完整可选实现（包含此前 Q2–Q4 未提交的必要运行基础）；旧分支和旧 provenance 不改写 | 179 文件：`8b7c78660f4efd79162dfb858c0f4a219e972412c5dc14680bcee509e5b8948e`；独立 facing-confidence overlay |

每组的 `provenance.json`、`source-snapshot/runtime-source.tar.gz`、`source-snapshot/files.json` 或 `resume-provenance/` 保存运行来源。复现还需该组 config、episode 清单、安装二进制／overlay 与模型权重指纹；单独 checkout 目前的 Q2/Q3/Q4 分支不等于恢复对应实验实现。后续完成各自归档时，应从精确快照提交并更新此表，不能把当前 Q4 工作区一起提交为 Q2 或 Q3。

### 运行修订与诊断数据

- Q1 的 161 文件 `522736612a799c61f6aeffaf515b7005324c44296242194dd3ac9c8438196517` 是全量扩展／失败视频修订；163 文件 `1c00c1e5be5cbbf67d4225b2e812a4b2cacb851b2bac91ac2a259401da396853` 是传输／有界恢复修订。保留 Q1 名称，另记来源修订，不将它们伪装成同一源码指纹。
- Q2 的算法父版为 `31ec9e08fde581dc1047200ba346ac6a4297f0bc67846f8de3d504fd86cc1d64`；全量版 `295b50…` 只增加固定单例调度兼容。网络、环境启动、离线审计修复单独记为运行修订，不因此提升 Q 编号。
- 早期同步诊断 `2f6c561c648f4d6e59d26da0d9f1e009b24ddb825842e2bdde09be37f83ee38d` 称 **`Q2-pre-sync15`**，还没有后来的 400 步修正，不能并入 Q2-full1000。
- 同步但关闭新增复查的诊断对照称 **`B0-sync-control15`**，属于 Q2 对照配置，不能与旧异步 B0 混称同一次实验。第一版诊断 Qwen 12/15、同步控制 13/15；后续 196 单例修正不回写此前 15 例成绩。更早消融继续使用原报告的 profile 名称，不能回标为 Q1–Q4 主版本。

## 日后汇报格式

统一写：**版本简称 + 集合简称 + 完成数／计划数 + 成功数 + source 指纹前 12 位**。例如：

- `Q2-full1000：1000/1000，757 成功，source=295b50a41230`。
- `Q3-first30：30/30，29 成功，source=5b561ad7d773`。
- `Q4-fail30：16/30，恢复成功 2/16，source=57c9b71c4a74，暂停`。

同版本不同集合也不自动合并。新批次在协议中记全局身份清单、seed、步数上限、profile、source／二进制／权重指纹、分支与 code_sha、输出路径。演示重跑与原成绩分开；网络未完成与完整导航失败分开；完整失败不选择性替换。当前 Q4-tvplant4 的恢复授权不恢复已暂停的 Q4-fail30 或全量失败分析。
