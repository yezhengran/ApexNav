# Q8-C2-semantic-v1：前100回合评测

2026-11-16用户授权：“用这套提示词跑前100个episode。”仅固定HM3Dv2 val全局1–100全部新跑，沿用已完成432／446诊断的semantic_guard_v1及当前导航配置；不扩展101以后，不恢复任何旧入口，不导入旧成功或诊断成绩。每身份首个完整结果唯一纳入，完整失败不选择性重跑；只保留失败视频，沿用每小时跟进。

## 方法与假设

方法Q8-C2-semantic-v1，数据名Q8-C2-semantic-first100；分支`codex/objectnav-annular-top5`。实现提交`3f3dce98`，随后均为文档提交，启动HEAD记录在本批protocol。203运行源SHA256 `f5fa79fc92c610839d7265c08efd607936792d727dfb76049c0054dfa91108bd`，精确同两例诊断；沿用完整6包`outputs/build/no-frontier-single-20261116/install`的641不可变文件，不重新编译或更换模型。

- `linked_choice --qwen-prompt-policy semantic_guard_v1 --crop-pose-fallback --crop-pose-single-view --choice-no-frontier`，frontier-look关闭；seed100、500步、step_sync。
- 单物体／联合图提示词保守处理单座软包座椅的chair／couch歧义，以及TV／画作歧义；多候选仍强制选择一个，优先类别证据明确者。bed／plant／toilet文本和盆栽0.60判定保持。默认legacy兼容保留，本批显式启用新policy。
- 原融合资格或最佳crop检测分>0.7初审；严格检测Top5、单图初审、最多一次环带观测后联合1–5图，允许补拍图未入选；不可行时用最佳crop拍摄距离备用弧，可单点；400后不绕行。多确认／400后无确认／无前沿靠近阶段比较、候选池、原靠近／STOP／碰撞和有界网络恢复均保持。
- 检验这一整体版本在前100的成功率、路径效率与调用／观测效果。与原Q8和Q2的对照包含提示词以外的方法差异；没有同配置legacy的前100新评测，不能称纯提示词消融。前100是固定val子集，不代表完整1000或未见场景泛化。

## 同身份历史基准

| 前100固定身份 | 成功数／SR | SPL | SoftSPL |
| --- | ---: | ---: | ---: |
| Q2 | 78／78.00% | 40.6128% | 42.6519% |
| 原Q8 | 80／80.00% | 43.2433% | 44.5326% |
| Q8-C2-semantic-v1 | 待完成 | 待完成 | 待完成 |

两组基准逐一核对原始CSV与episode_set身份，不重新推理。comparison.json每完成一例只按同一已完成身份前缀更新SR／SPL／SoftSPL、配对新增成功与退化；不拿短前缀与完整100直接比，不回填旧成绩。

## 唯一新入口与冻结

新控制目录 `experiment_notes/qwen-semantic-first100-20261116/`，唯一入口其`launch.sh`；日志 `outputs/runtime/qwen-semantic-first100-20261116-launcher.log`。started与`outputs/runtime/q8c-single-evaluation.lock`独占保护，禁止重复启动。新输出 `outputs/runs/qwen-semantic-first100-20261116/gNNNN/`；每run局部episode1，按计划映射全局1–100、CLI零基0–99。真实状态以本目录round-status／latest-checkpoint／results／comparison及实际进程为准，旧两例和窗口入口全部停止。

启动前11项实际调度／比较／配置漂移故障注入通过；首1、尾100真实CLI+Hydra配置除episode索引与输出目录外精确同诊断。203源码、641完整overlay、原安装／父构建／权重与62,353旧文件全部核验；仅协议列出的3个ATSP临时输出例外。沿用实现阶段179项回归，无新运行源码修改。提交本批协议文档后重新冻结HEAD，再启动。运行中不改源码／配置／HEAD、不编译不换模型、不提交小时进度。

每回合严格核验身份、唯一完成attempt最后episode_end、来源Top5有序像素／mask簇回执、几何和备用圈、候选比较池、每候选最多2完成类别复查、prompt policy／哈希和有序HTTP账本链；失败视频steps+1全帧解码。成功删除临时视频，网络未完成partial／attempt和API原件保留。每10回合checkpoint，100结束controller先停四模型，再final_validate.py独立验收100身份／同版来源／全部失败视频；不重复已有独立审计目录。

共享20元账本起点3.222627元，所有历史条目／失败预留保留。单逻辑最多3HTTP、2／5秒重试、每次0.02预留；仍失败仅匹配transport／全部有序图片哈希／时间／账本的未完成回合可存证、无密钥连通检查后同身份最多2次30／60秒恢复。证书／HTTP／未知SSL／异常回答／预算／来源审计、重连失败或耗尽停查，不绕过或无限重跑。key仅config/vlm.env，不显示／提交。模型加载后MemAvailable≥2.5GiB，只有原两项malloc环境。

## 跟踪与收尾

沿用每1h跟踪实际完成数、同集合比较、GPU／RAM／磁盘、调用／联合图／观测／比较／网络恢复／费用；有意义变化才简报，异常及时调查。用户暂停立即安全停止本批与四模型并保留现场。100结束确认所有进程退出且独立验收，报告前100 SR／SPL／SoftSPL、类别场景、配对变化、调用／观测与成本、失败视频路径；更新本报告／Q8版本／交接，明确路径审diff凭据与大文件后仅commit+push origin当前分支并核对SHA。最终报告后暂停objectnav-100等待用户，不扩展范围。
