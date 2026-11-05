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
- 当前会话 Docker socket 访问被拒绝（包括请求沙箱外执行后），`sudo -n docker ps` 需要密码。最初沙箱内 `/dev/nvidia*` 不可见；后续沙箱外只读核验确认主机 RTX 4060 正常（驱动 580.178.04，显存约 613/8188 MiB，未见模型进程），RAM available 约 7.5 GiB、无 swap。未能检查容器内评测进程。
- 因上述限制，尚未运行本轮 Habitat/GPU 集成或 50 回合；不是基线复现成功证据。不修改 Docker 授权/系统权限。

## 对照与结果保存

代码冻结、模型健康检查、构建和单回合冒烟通过后，顺序运行两组各 50 回合；不并发运行，不在对照中途改代码。固定 seed=100、shuffle=false、500 步、0.2 m 成功阈值；两组实际 scene/episode ID 必须逐项相同。使用独立目录 `outputs/runs/av-baseline-50`、`outputs/runs/av-adaptive-50`，不续跑旧的未完整记录版本实验。

每回合增量保存 CSV/summary；保存实际 episode 清单、代码 SHA、配置和方法诊断，阶段性比较必须标明已完成数量。前 50 回合是顺序开发子集，不能称完整 val 或未见测试集。临时测试脚本只放 `/tmp`，验证后删除；保留已有可复用回归测试。

后续 Tracer Mini 迁移仍需确认底盘、传感器、标定、定位和急停接口；本轮不操作实机。

## 首版实现与当前证据

已实现四个运行设置，默认 `baseline`；Python `vlm/active_view.py` 负责候选、原始 crop、历史确认、主动取证和拒绝记录；C++ `active_view.h` 加 FSM 适配负责观测点和现有 A* 的安全路径。确认后继续原 ApexNav 靠近/停止流程。

首版明确简化了 DynZSON：以 RGB-D 点云稳健二维范围估计尺寸、0.75 m 最近中心关联候选，不是完整语义实例融合；24 个角度采样一个圆环，以邻域自由空间数和行程排序，尚未迁移语义—占据边界融合或 ESDF 词典序全部项。物体内部视线豁免使用半尺寸近似，不能据此保证真实目标轮廓附近无遮挡。相邻同类实例可能关联错误，大物体局部 mask 可能低估尺寸，需用失败案例进一步检验。

历史确认至少需两次观测，保持前 26 个初始扫描动作；ITM 接受阈值 0.65。未通过时固定观测点，最多 48 个实际动作或 24 步无平移；到点位置容差 0.3125 m、航向容差 18°。到点握手复用上一个真实动作后的当前观测，不调用额外免费感知步、不伪造 Habitat STOP；实际平移/旋转仍计入 500 步。A* 输出与量化前进段另做自由/膨胀检查，失败即取消本次观测。服务错误直接使运行失败，不做静默接受。

新批次保存 `provenance.json`（提交与运行源码指纹）、`config.yaml` 和实际开始顺序的 `episode_set.json`；工作区必须干净。每回合原子保存 CSV/summary，每 10 回合及结束保存 `checkpoint-NNNN.json`，结果文件设置为主机可读的 0644。每次候选验证保存 crop、pose、分数、耗时及拒绝原因到 `verification/episode-NNNN/attempt-*/`，续跑保留未完成尝试证据。新旧代码/设置不得混合续跑；旧无 provenance 的实验保持原样，另开目录。

离线验证：12 项 Python 候选/两阶段确认/缺帧/拒绝/预算/服务错误/证据保存测试通过；独立 C++17/Eigen 观测几何、锁定、路径段安全和不可达检查通过（`-Wall -Wextra -Werror`）；7 项批量保存/续跑/对照测试通过；7 个变动 Python 文件语法、Shell 语法和 diff 空白检查通过。既有视频回归因主机缺少 `imageio` 未运行成功；未安装或修改容器依赖。完整 ROS2 编译、真实 BLIP2 推理、Habitat 冒烟与 50 回合仍未验证。

离线验证输出保存在 `experiment_notes/validation/20261105-adaptive-view/`（已忽略）：当前用户不能在容器所有的 `outputs/` 新建目录，所以仅本机离线验证记录使用此目录；仿真原始输出仍计划由容器写入 `outputs/runs/`。临时 Python/C++ 测试文件验证后删除，不提交；复用已有 `docker/test_batch.py`，不删除原有回归。

另从可读的旧 `record.txt` 首条记录和 `console.log` 交叉核实：旧批次完成 27/50、25 success、2 no frontier，全部来自 `4ok3usBNeis`；文本舍入 SR=92.59%、SPL=51.26%、Soft SPL=51.54%、平均距离=0.6565 m。其运行代码 SHA 未记录，CSV 未读取，故只保存到 `legacy-partial.json` 作历史诊断，不作新方法对照。该子集未出现 false positive，当前没有证据说明确认模块会提高其 SR。

## 解锁 Docker 后的执行顺序

先核对无其他评测、GPU/RAM，完成 `./docker/apexnav build`、模型启动与健康检查。先完整单回合、再固定 10 回合排障，冻结代码后按以下顺序各 50 回合；不要给正式对照设置 `--max-steps`：

```bash
./docker/apexnav run-batch hm3dv2 --episodes 1 --method adaptive_view --video --output outputs/runs/av-smoke-1
./docker/apexnav run-batch hm3dv2 --episodes 10 --method adaptive_view --video --output outputs/runs/av-debug-10
./docker/apexnav run-batch hm3dv2 --episodes 50 --method baseline --seed 100 --video --output outputs/runs/av-baseline-50
./docker/apexnav run-batch hm3dv2 --episodes 50 --method adaptive_view --seed 100 --video --output outputs/runs/av-adaptive-50
python3 docker/compare_runs.py outputs/runs/av-baseline-50 outputs/runs/av-adaptive-50
```

对照工具只比较两组已完成的共同前缀，严格检查 scene/episode/target、seed/上限和源码指纹，并明确标记是否完成；中途每 10 回合可保存一次输出。10 回合用于调试而非最终结论；若代码变更，后续用新目录重跑对应对照。
