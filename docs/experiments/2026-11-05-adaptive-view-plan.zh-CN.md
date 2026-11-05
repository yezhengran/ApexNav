# 尺寸自适应观测 + VLM 确认：首轮方案

## 假设与范围

在检测误报或视角不完整时，先用最佳历史 crop 做 VLM 确认，未接受时再到一个随物体尺寸变化的可达观测点复核，能减少错误目标追踪与误停。代价是额外动作、推理时间和可能的漏检。主指标为同 episode 的 SR/SPL，同时报告 Soft SPL、距离、步数、耗时、VLM 次数及主动观测失败原因。没有仿真结果前不声称改善。

参考 [DynZSON@f6c2bf0](https://github.com/yezhengran/DynZSON/tree/f6c2bf05257fd0cac6810a2384000c911721e565)：`modules/adaptive_observation_distance_node.py`、`modules/static_object_nav_node.py` 和 `paper_innovation_outline.md`。迁移尺寸距离公式、锁定单观测点、历史/到点证据隔离和有界复核；不整体复制它的地图、前沿排序或离散路径跟踪器。

`extent = clip(二维目标点云最长边, 0.15, 3.0)`；`edge = clip(0.75 * extent, 0.6, 2.0)`；`center_radius = edge + extent/2`。该距离仅用于确认，不改变 Habitat 成功距离或 ApexNav 确认后的靠近距离。

## 实现约定

- 默认关闭，原基线动作、检测和地图更新路径保留。实验只针对 HM3Dv2。
- 复用已有本地 BLIP2 ITM，对未经绘图的目标 crop 评分；初始阈值 0.65。这是待检验的图文匹配门控，不等同于 DynZSON 的 Qwen2.5-VL JSON 语义判别，也不是校准的目标存在概率。
- 候选只使用 RGB-D、GPS/Compass 与已建占据图；在线策略不读取目标真值、最短路或评测成功信号。原基线已有同楼层真值筛选暂保留，不能宣称整个基线完全无 oracle。
- 候选在同一 episode 内关联、保存最佳 crop；未确认的目标点云暂不进入目标地图，相似物体仍沿用原行为。每个候选最多历史/主动各一次 VLM 调用，无检测器兜底。
- 主动观测沿用 ApexNav A* 与动作规划，检查已知自由、障碍膨胀、路径和目标前方视线。固定观测点，限制总动作与无平移步数；到点朝向目标后仅用当前视角证据确认。
- 不可达、超时、到点无当前目标或第二次未通过时拒绝该候选并继续探索。VLM 服务错误使实验失败，不默认为接受。
- 保留 `baseline`、`vlm`（仅历史确认）、`fixed_view`、`adaptive_view` 四种设置；先做基线与完整方法各相同 50 回合，再决定消融规模。

## 本轮基线核验（2026-11-05）

- 工作分支 `codex/objectnav-sim-to-tracer-mini`，开始 HEAD `4dce9c1fd66529b98a4271a025787e971c6880f9`，仅文档区别于环境/评测基线 `3828f38`；开始工作区/暂存区均干净。
- origin 为 `git@github.com:yezhengran/ApexNav.git`；远程核验 SHA 与本地相同。
- `python3 docker/check_dataset.py` 通过：1000 episodes、36 scenes，geometry/navmesh 和 semantic/config 缺失均 0。
- 现有 `docker/test_batch.py` 中 5 项无需视频依赖的计数、续跑、命令和子进程清理测试通过。
- 旧 `outputs/runs/hm3dv2-local-50/status.json` 仍为 `interrupted / 130`；日志最多完成 27 回合。`summary.json`/`episodes.csv` 为容器用户 0600，当前用户无法读取，不据此补造汇总。
- 当前会话 Docker socket 访问被拒绝（包括请求沙箱外执行后），`/dev/nvidia*` 不可见，`nvidia-smi` 无法连接驱动。主机内存读数约 7.2 GiB available、无 swap；当前进程命名空间不能确认主机上的评测进程。
- 因上述限制，尚未运行本轮 Habitat/GPU 集成或 50 回合；不是基线复现成功证据。不修改 Docker 授权/系统权限。

## 对照与结果保存

代码冻结、模型健康检查、构建和单回合冒烟通过后，顺序运行两组各 50 回合；不并发运行，不在对照中途改代码。固定 seed=100、shuffle=false、500 步、0.2 m 成功阈值；两组实际 scene/episode ID 必须逐项相同。使用独立目录 `outputs/runs/av-baseline-50`、`outputs/runs/av-adaptive-50`，不续跑旧的未完整记录版本实验。

每回合增量保存 CSV/summary；保存实际 episode 清单、代码 SHA、配置和方法诊断，阶段性比较必须标明已完成数量。前 50 回合是顺序开发子集，不能称完整 val 或未见测试集。临时测试脚本只放 `/tmp`，验证后删除；保留已有可复用回归测试。

后续 Tracer Mini 迁移仍需确认底盘、传感器、标定、定位和急停接口；本轮不操作实机。
