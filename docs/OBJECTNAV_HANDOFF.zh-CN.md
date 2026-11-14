# ObjectNav 新窗口交接

**最新完成（2026-11-14）：用户确认融合 0.5＋Qwen 0.5，Q5 两个固定演示已完成。** 新分支 `codex/objectnav-facing-confidence-review`、profile `linked_confidence`，179 运行源文件 `8b7c78660f4efd79162dfb858c0f4a219e972412c5dc14680bcee509e5b8948e`；独立 overlay `outputs/build/facing-confidence-20261114/install`。二次调用须到达安全点并完成朝向中心后的真实观测；输入仍用同对象全历史最高检测分数 crop，可复用旧图；Qwen 新增严格类别置信度，400 步后原可用未确认候选按 0.5/0.5 排序，未调用者用融合分数。chair10：197步成功、2次复查、20观测动作；tv885：77步成功、3次复查、39观测动作。两例均<400，加权仅离线验证；旧Q4两例/全量结果不改。7次HTTP中2次SSL按原有界规则恢复，0回合重开，0.04失败预留保持，总账本1.345770元。独立审计2例276帧/来源/朝向/分数/旧6119文件通过；91项Python/ROS/调度检查及C++排序/回执检查、独立构建通过。评测/规划器/四模型全部退出，禁止重跑任何旧入口；旧30和全量失败分析保持暂停。协议/审计在 `experiment_notes/qwen-facing-confidence-20261114/`，英文观看版在 `outputs/runs/qwen-facing-confidence-20261114/presentation-en/`。详见[本轮记录](experiments/2026-11-14-qwen-facing-confidence.zh-CN.md)。

更新：2026-11-11。先读 AGENTS.md、docker/experiment_commands.zh-CN.md；最新用户指令优先于旧默认分支与提交频率。

## 当前状态：50+50 已结束，等待下一方法对齐

用户确认的新原 ApexNav 50＋Qwen 额外复核 50 已完成，均 completed/0。独立核验 100 视频完整解码、固定身份顺序、源码/41二进制/4权重不变、49次API图像来源与账本。调度器、评测和规划器退出，四模型已停；不要重复启动或自动扩展。

结果与证据见 [最终报告](experiments/2026-11-10-qwen-paired50.zh-CN.md)：baseline 38/50、SR76%、SPL41.7953%、SoftSPL44.9111%；Qwen42/50、SR84%、SPL46.4892%、SoftSPL47.6022%。6恢复、2损失；49对首次复核前有动作差异，不能把全部增量归因Qwen。

44次请求全部STOP触发，40候选有证据，49次调用26确认/23不确定/0否定，242观测动作。正式API估算0.040046元，含检查和失败预留账本0.061730元（20元上限），不可重置账本。模型别名qwen3-vl-plus无法保证服务端权重固定。

Qwen8失败：2超时、1受阻均未触发复核；1盆栽STOP缺证据无调用；4TV均两次不确定后按原规则停止。离线像素核验发现这些TV输入主要为真实wall tv_231，而任务只认可monitor_353/354。不得把它当非电视负样本、手改成绩或向在线输入真值。23个不确定中21个模型自报证据不足，裁图/视角值得改进；不等于已经验证模型准确率。

建议待对齐：先用同批保存图像离线比较小crop与全图标框＋细节crop，再独立验证有界主动取景与更有信息的第二视图。保持原触发、探索、候选选择、STOP规则、模型及官方评测不变；具体动作预算与验证范围先询问用户，不自动运行下一轮。

## 分支、来源与保存

当前修复分支 `codex/objectnav-review-evidence-fix`。运行基底 `9d4b14e58c52421c3e3983edf7fd30cced6f165c`；本轮实际未提交运行快照指纹 `99f2fb75f4e38b9bdbd26f8656e6bad437099ee15b1ce20914068170af5c4b82`（160运行文件）。事后提交不改写运行provenance。原分支 `codex/objectnav-sim-to-tracer-mini` 保留 `e25740f`。用户要求整轮结束统一commit/push origin同名修复分支、核对SHA，不提交阶段进度。

两组固定同前50、seed100、500步、视频。baseline额外复核关闭，原BLIP2 ITM感知保留；Qwen为selected_review/linked_stop_preserve，只替换额外复核模型，未改观测运动。.65/.20不用于Qwen文字三段判定；每候选≤2调用，STOP来源≤2真实步。不采用.75触发线，不启实机，不输入在线真值。

原始输出保留于 `outputs/runs/qwen-paired50-20261110-{baseline,qwen}/`。本地协议、清单、账本、检查点、最终结果于 `experiment_notes/qwen-paired50-20261110/`，独立分析于其 `final-analysis/`。调度/一次性分析脚本归档后清理，不能恢复入口直接重跑或覆盖输出。

用户已要求清理此前实验：93个精确路径约8.75GB及十份旧报告已删除；旧数据/视频/日志不恢复、不混算，Git历史不重写。本地Qwen权重和专用环境已删除，不下载。新证据正常保留。

百炼北京地址 `https://dashscope.aliyuncs.com/compatible-mode/v1`。密钥仅在忽略的config/vlm.env，绝不显示、提交或写进快照。API异常/预算不足停止，不静默降级。正式100和失败分析完成后暂停自动跟进等待用户回复。
