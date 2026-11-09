# 100 回合实验的续跑工具修复

2026-11-09 用户要求继续实验。原目录 `outputs/runs/review-400-065-020-100-r2/` 保留 66 个完整回合，成功 51 次；第 67 回合需从头重跑。恢复前重新核验暂停归档 SHA256 及原目录全部 217 个文件，均与 2026-11-06 暂停时一致。原 provenance、固定清单和已完成结果不改写。

## 根因与修复范围

Habitat 0.3.3 的 `Env.__init__` 已消费第一个 episode，并用公开 `current_episode` setter 固定首次 reset 的回合。旧评测只推进 iterator，没有更新这个固定值，因此非零起点首次 reset 仍回到 episode 0；清单保护正确拒绝了混集。

`seek_episode` 在跳过已完成回合时同步设置 `current_episode`，供批量续跑与非零单回合入口共同使用。零起点行为不变。未改探索、目标选择、观测方法、模型、阈值、500 步上限或成功判定，未重新构建 C++。

工具源码修复改变了总源码指纹，因此保留原 `provenance.json`，通过 `docker/resume_migrations.json` 中审阅过的精确旧/新指纹对，以及显式 `--resume-migration habitat-initial-reset-v1` 允许本次迁移。未知迁移、反向迁移、任何额外源码变化均被拒绝；没有通用强制跳过检查开关。每次续跑独立写入 `resume-provenance/attempt-*.json`，记录原/当前提交与源码指纹、已完成数、CSV/config 哈希及迁移理由。解析后的配置必须与原文件逐字一致，原配置不覆盖。

比较工具遇到发生源码迁移的运行时拒绝自动声称相同源码对照，要求先审阅分段 provenance。本轮最终结果应标明第 67 回合前曾中断、恢复段含工具修复；进程重启也不会恢复回合内随机数状态，不能保证与不中断运行逐动作相同。

## 验证结果

- 容器内现有批量回归共 9 项通过，包括 CSV/汇总、视频、清单拒绝混集、进程停止、精确迁移许可及比较保护。
- 真实 Habitat 使用原 HM3Dv2 val、seed=100，对零起点、索引 5/27/66/99 定位后实际 reset/render，核验自身与后继回合（共 9 次 reset）身份均匹配固定清单；包括 27→28 跨场景和 66→67 连续顺序。正式目标起点是零基索引 66，即全局第 67 回合、场景 `6s7QHgap2fW` / ID `10` / tv。
- 证据保存在 `experiment_notes/validation/20261109-resume/batch-tests.log`、`habitat-cursor.log`、`habitat-cursor.json`；未新增一次性测试代码文件。
- 修复提交 `544405738ee16ecfa4f45086847132fbcb0f7ec0` 已推送并核对远程 SHA。恢复段源码指纹为 `baeceb2fb97f512c3608068ce081bcc7631b6eba61c8ebf71656c2dd26dd859f`，与原指纹的对应关系记录在迁移登记文件。
- 真实端到端诊断：`outputs/runs/resume-tool-diagnostic-20261109/`，3 回合、每回合最多 3 步、同样 selected_review/400/0.65/0.20、seed=100、保存视频。完成第 1 回合后 SIGINT 得到 `interrupted / 130`；随后使用 `--resume` 完成第 2/3 回合，最终 `completed / 0`。3 个回合顺序正确、唯一；原第 1 行结果、配置和 provenance 字节保持不变，独立续跑版本记录存在。证据：`end-to-end-resume.json` 及配套日志。
- 实际评测入口 `test_epi_num=66` 的独立 3 步单回合诊断也通过，实际 scene/episode/target 为 `6s7QHgap2fW` / `10` / tv，结果在 `outputs/runs/resume-single-66-diagnostic-20261109/`。这两个低步数诊断不计入正式成绩。只读核验正式 100 回合恢复参数解析出的配置与原 `config.yaml` 逐字相同，精确旧/新源码指纹迁移通过；证据为 `nonzero-single.json`。
- 诊断退出后确认没有评测/规划器残留；四模型已按原配置启动，实际 HTTP 推理检查通过，主机可用 RAM 约 6.0 GiB。接下来使用同一正式目录恢复剩余 34 回合，原参数及视频设置保持不变。
