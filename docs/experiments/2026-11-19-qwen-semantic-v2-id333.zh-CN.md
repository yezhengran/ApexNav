# semantic-v2：剩余旧编号333例统一重测

用户2026-11-19明确授权：“现在把旧编号重新跑一遍，整合到一块。”本批只重跑最终混合1000中尚为`global_v1`的全部333个身份，使用最终semantic-v2和`image_order_v1`，保留其余667个新编号结果。尚未完成时不发布新的混合1000成绩。

## 范围与整合规则

| 身份集合 | 例数 | 原成功数 | 原SR |
| --- | ---: | ---: | ---: |
| chair | 122 | 106 | 86.8852% |
| potted plant | 100 | 52 | 52.0000% |
| toilet | 111 | 89 | 80.1802% |
| **本次全部重测** | **333** | **247** | **74.1742%** |
| 原样保留的新编号结果 | 667 | 525 | 78.7106% |

按全局编号升序运行固定清单，首例2、末例642；不是连续2–642窗口。局部episode1精确映射原HM3Dv2 val身份，CLI索引为全局编号减1。新333的每个首个完整结果无条件替换旧333，包括新失败；不按成功择优，不覆盖原结果。新混合成功数=`525+新333成功数`。保留667行、原772/1000及其证据不变。

运行清单：`experiment_notes/qwen-semantic-v2-id333-20261119/execution-plan.json`。旧编号明细已逐例核对原experiment、provenance和CSV；缺字段只在已验证的修复前源码/身份边界内认定旧编号。原混合结果SHA256为`b4fa39d82b268b3659571fbcfc18bf1cbfababe27adf3c1de5aa8bc316cde4a6`。

## 代码、配置与对照

- 开发基准：`Q8-C2-semantic-v2`。准备时HEAD `4528f10a44315a6479df7f4f8da0c5bdb119b2a1`；启动HEAD与完整源码清单由新`protocol.json`在协议文档提交后冻结，运行中不改HEAD/源码/配置。
- 使用移除方向模块后重建的完整六包：`outputs/build/semantic-v2-final-20261119/install`。不恢复已移除的方向模块或旧实验入口。
- HM3Dv2 val原身份、seed100、500步、step_sync、linked_choice、semantic_guard_v2、image_order_v1；crop-pose-fallback、crop-pose-single-view、choice-no-frontier开启。原Top5/环带/比较/STOP/plant0.60/400步后不绕行等保持。
- chair/plant/toilet的v2类别提示词与v1逐字相同。主要接口差异为比较请求使用图片序号1–N，严格映射回全局候选ID；历史333只有35例曾触发比较（56次调用），不保证重跑提高SR。
- 首先报告新333与旧333、Q2同身份配对；全部完成并独立验收后再报告整合1000及类别、场景、FP和配对增减。
- 保留667含487个v2和180个v1新编号结果，真实prompt/源码/HEAD按原记录保留。整合后统一编号协议，仍非全1000新跑、同源码或严格单变量消融；val曾参与开发，云端服务别名不保证不同日期权重相同。

## 控制、审计与费用

独立入口：`experiment_notes/qwen-semantic-v2-id333-20261119/launch.sh`；只以本目录`started.json`、`round-status.json`和实际进程判断是否启动，禁止重复入口。新输出`outputs/runs/qwen-semantic-v2-id333-20261119/gNNNN/`，日志`outputs/runtime/qwen-semantic-v2-id333-20261119-launcher.log`。只读进度：本目录`status.py`。

启动前核对1000原身份/CSV/源码快照及667保留结果，独立检查保留视频和审计证据；实际调度故障注入、无择优合并测试、首尾真实CLI+Hydra配置检查通过后冻结。冻结原结果、控制、源码、完整构建、权重、依赖与账本前缀；只允许既有三个ATSP临时输出变化。

单评测锁与started独占保护。首个完整第2例后由控制器调用独立进程进行唯一startup-validation，通过后继续调度；不另启重复守护。逐例严格审计身份、唯一完整attempt、末尾episode_end、Top5有序像素与mask来源、观测几何、比较池、最多2次完整类别复核、prompt哈希、ID映射和HTTP账本。离线角度检查沿用已验收的圆周角helper及原6e-5容限。完整失败保留steps+1帧并全解码，成功删除临时视频，API/partial/attempt证据保留。

本批共享账本起点 **8.536063/20元**，只计新批增量；全部历史条目与预留保持，不重复收费保留667。既有网络有界恢复不变：每逻辑至多3HTTP，2/5秒；只有严格匹配transport证据的未完成身份允许至多2次30/60秒重连。其他异常停止调查，不增加通用初始化、回答或本地HTTP重试；完整失败本批只计一次。

启动前空闲内存门槛沿用7GiB；用户此前已撤销加载后4GiB硬门槛，因此只记录加载后实际内存并持续监测。Codex桌面及代理保持打开。333完成先停模型，再独立复验新333及保留证据，写`independent-validation.json`后交付`hybrid-results.json`、`hybrid-comparison.json`、`hybrid-provenance.json`。所有历史入口继续归档，不扩大范围。
