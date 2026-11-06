# ObjectNav：仿真开发到 Tracer Mini 的新窗口交接

更新日期：2026-11-06。先读根目录 [AGENTS.md](../AGENTS.md)，再按本文继续。

## 最新任务：100 回合与失败修复

用户已选定：先跑当前方案 100 回合，再分析失败、提问对齐修复，建立新分支并同集复测。首批 `review-400-065-020-100` 完成 5 回合后切换后台，续跑遭 episode 顺序保护拒绝（`failed / 1`）；5 回合只作诊断，不拼入正式成绩。当前正式目录为 `outputs/runs/review-400-065-020-100-r2/`，从头完整跑 100，使用 Docker detached exec，启动日志 `outputs/runtime/review-100-r2-launcher.log`；不使用已暴露顺序问题的 `--resume`。运行中禁止修改源码/配置。后续检查 `objectnav-100` 每 15 分钟跟进，每 10 回合保存阶段结果，满 100 后先向用户对齐具体修复再改方法。启动前旧容器 GPU 访问失效，已停止模型并重启容器恢复，四模型实际推理通过。详见 [100 回合记录](experiments/2026-11-06-hundred-episode-failure-analysis.zh-CN.md)。旧的“50 回合已完成”不限制此次新授权；原实验分支保留，新修复分支权限来自本次用户指令。

## 本轮续作状态：自适应观测实验

**用户最新范围修正（2026-11-05）：** 只保留类似 DynZSON 的观测/重新验证环节，不改变 ApexNav 的探索策略、探索方式；方法改动前先向用户提问对齐。原型的目标点云前置过滤、禁用兜底选择超出此范围，不能继续视为用户认可的方法。对齐后已确认：仅当原规划器选中融合置信度未达原阈值的目标时复核；高置信度目标照旧；不确定候选保留，后期或无可用前沿时再进入原靠近/STOP 流程。用户授权分别测试 350/400 步及在 10 回合上调整三段阈值、选取效果较好的配置。新范围见 [重新验证方案](experiments/2026-11-05-selected-target-review-plan.zh-CN.md)。这不是仅在 STOP 时一律复核的旧提议。

**扩展评测门槛已由用户确认：新方法先完成同集 10 回合，至少成功 8/10，且成功数不低于基线，才扩到 50 回合。** `av-debug-10` 旧原型在 9 回合仅成功 2 次，已正常中断并保留结果；同集 `baseline` 10 回合已完成 8/10 成功、SPL=50.15%；新方案已实现并完成两组 50 回合对照（见下方），不启动旧原型 50 回合。

用户已选定迁移 [DynZSON](https://github.com/yezhengran/DynZSON) 的尺寸自适应观测并用 VLM 确认，授权方案完成后做固定 50 episode 对照；不再需要询问是否选择此方向。详细假设、实现边界、离线验证、旧结果和命令见 [首轮实验记录](experiments/2026-11-05-adaptive-view-plan.zh-CN.md)。

- 当前入口为默认 `baseline` / 可选 `selected_review`；已移除旧原型的前置点云过滤和全局兜底禁用。只复核原规划器选中的低置信度簇，三段结果保留不确定，350/400 步门槛可配置。复用 BLIP2，入口为 `vlm/active_view.py` 与 C++ FSM / `active_view.h`。历史旧原型结果保留。
- 保存逐回合指标、每 10 回合检查点、验证 crop/分数/位姿和实际 episode 清单；`docker/compare_runs.py` 检查同集同配置源码的共同前缀。新批次源码需提交，旧无 provenance 的目录不可用新启动器续跑，以免混入新版本。
- 新范围通过 10 项状态机测试、2,000 组与 `3828f38` 的默认对象候选比较、C++ 状态/路径检查、真实六包构建、批量/视频 8 项及 Habitat 动作 2 项。临时测试脚本已删除，验证日志保留在 `experiment_notes/validation/20261105-adaptive-view/`。
- 350/400 步开发组均 9/10，达到同集基线 9/10，按用户门槛已完成 baseline/400 步方法各 50 回合：**均为 42/50 成功（84%）；SPL 分别 42.79% / 47.41%。** 两组运行源码指纹一致，完整配对通过，3 次成功变失败、3 次失败变成功。覆盖两个场景并包含开发集，不能代表完整 val 或未见测试集。详见 [完整实验结果](experiments/2026-11-05-selected-target-review-results.zh-CN.md)。
- **新方法全程额外复核/观测为 0，方法有效性和 VLM 判定准确率尚未验证。** 不将 SPL 波动归因于复核收益。用户于 2026-11-06 明确选择“保持当前方案，继续到 50 回合”，本轮不采用独立 0.75 复核触发线。既定两组 50 回合已完成，该回复不表示追加一组重复评测；后续方法改动仍须先对齐。
- 本轮两组均 `completed / 0`，目录为 `outputs/runs/review-baseline-50/` / `outputs/runs/review-400-065-020-50/`。100 个完整视频、每 10 回合检查点、逐回合指标及完整配对/失败/复核统计已保存，主机备份在上述 validation 目录。评测与规划器已退出，四个本次模型服务已停止，容器保留；当前没有评测运行。下一轮先完成交接核验，并与用户对齐方法改动。
- C++ 构建前先停本次模型服务（确认没有评测），优先单进程构建；四模型与双进程编译并行曾使 RAM 余量降至约 67 MiB，停止模型后恢复，禁止重复此运行组合。
- 用户已在主机完成 Docker 授权；本会话可在沙箱外访问现有容器。`55c11c6` 已完成真实 ROS2 六包构建（并发限制为 2）、容器内批量/视频 8 项、HTTP 协议 8 项、Habitat 动作 2 项回归；真实场景 `HABITAT_ROS_SMOKE_OK` 和四个模型的 HTTP 推理通过。
- 完整旧原型 `adaptive_view` 单回合已成功：`av-smoke-1`、代码 `cbe50fe`，345 步、SPL=13.75%。后续 9/10 部分诊断 SR=22.22%、SPL=8.27%，存在真实床、马桶、沙发 crop 被拒绝的证据；该原型未通过门槛且范围不符。已验证启动 CPU 缓存回收；用户关闭其他应用后四模型可用，导航期间 available 约 2.3–2.8 GiB。详细记录见 [续作验证](experiments/2026-11-05-integration.zh-CN.md)。
- 旧 27/50 的文本日志可读：25 成功、2 无前沿，SR=92.59%、SPL=51.26%（舍入），单场景 `4ok3usBNeis`。仅为历史部分结果，无运行 SHA，不能充当本轮正式基线。

## 1. 用户意图

用户希望以 ApexNav 为起点，实现自己的目标导航版本：先在仿真和数据集上进行可复现测试，再迁移到实机。当前首项方向为尺寸自适应观测与 VLM 确认；不要把离线原型或现有环境修复当作已经验证有效的新算法。

- 当前任务是 ObjectNav：给定目标物体类别，探索环境、寻找目标并判断到达；点到点导航仅是后续控制验证的一部分。
- 工作在用户自己的 [GitHub 仓库](https://github.com/yezhengran/ApexNav)，每个完成的逻辑改动经验证后提交并推送到专用分支。
- 当前优先使用四个本地视觉模型；资源不足再通过已配置的 HTTP 接口外置部分模型。
- 用户称目标底盘为 `tacer_mini`，已有分支使用 `tracer-mini`。本文暂用 **Tracer Mini**；实际型号、驱动和传感器仍需确认。
- 用户需要简洁中文说明、可复制命令、可观看视频和可保存的实验结果。

## 2. 从哪个分支开始

2026-11-05 已 fetch 并核对用户远程仓库：

| 原有分支 | 核对结果 | 用途 |
| --- | --- | --- |
| `dev/tracer-mini` | `1ec9d15`，与 `ros2-jazzy` 完全相同 | 本次派生起点 |
| `ros2-jazzy` | ROS2 Jazzy 社区移植，含回合重置修复 | 保留上游环境参考 |
| `main` | `049971b`，规划模块仍用 ROS1/catkin | 参考算法修复，不直接合并 |
| `Lite-Apexnav` | `63c3ed4`，规划模块仍用 ROS1/catkin | 参考轻量化设计，另做对照 |
| `gh_pages` | 项目展示页 | 不用于实验开发 |

**工作分支：`codex/objectnav-sim-to-tracer-mini`。**

基线链路：`origin/dev/tracer-mini@1ec9d15` → **`3828f38` 环境/评测基线** → 本文与后续方法改动。
`3828f38` 保存了此前仅存在于本机的 Docker 工作流、Habitat/模型兼容补丁、远程接口、批量计数/续跑和视频记录功能；这是本机复现基线，不应宣称与原论文发行版完全等同。

`main` 中的 `dd4a822`（A* 节点池）、`204121e`（语义 frontier 索引）和 `049971b`（frontier/object 更新）不在 Jazzy 分支提交历史中。后续先比较代码确认是否适用，再独立移植、测试和记录，避免把基础修复收益混进算法收益。

## 3. 已有环境与当前证据

| 项目 | 信息 |
| --- | --- |
| 主机工程 | `/home/yezhengran/My_projects/ApexNav` |
| 容器 / 工程挂载 | `apexnav-jazzy-egl` / `/workspace/ApexNav` |
| Python / ROS 工作区 | `/opt/apexnav_venv` / `/opt/apexnav_ws` |
| 记录的运行环境 | 容器 Ubuntu 24.04、ROS2 Jazzy、Python 3.12、Habitat 0.3.3、Torch 2.10 / CUDA 12.8、EGL |
| 依赖版本依据 | [setup_container.sh](../docker/setup_container.sh)；Sim 固定为 `acbe6f49`，Lab 为 `v0.3.3` |
| 本机资源 | RTX 4060 Laptop 8 GiB，主机 RAM 约 16 GiB；2026-11-05 检查时无 Swap |
| 当前数据 | HM3Dv2 val；本轮检查通过：1,000 episodes、36 scenes，任务所需几何/navmesh/语义/config 缺失数均为 0 |
| 模型服务 | BLIP2 `12182`、YOLOv7 `12184`、MobileSAM `12183`、GroundingDINO `12181` |

日常操作以 [实验速查](../docker/experiment_commands.zh-CN.md) 和 [容器说明](../docker/README.md) 为准。根 README 的通用安装步骤不是本机需要重新执行的清单。

已解决的关键问题包括：Lab/Sim 的 `Agent.sensors` 与 `_sensors` 接口差异、PyTorch 2.10 的模型/扩展兼容、HM3D 不依赖 MP3D 标签文件、视频逐帧编码降低内存占用。

旧结果在 `outputs/runs/hm3dv2-local-50/`：本轮读到 `status.json` 为 `interrupted / 130`，日志最后记录完成第 **27** 个回合。汇总 JSON 需要容器权限读取；**50 回合尚不能认定完成**，不要据此发布完整 50 回合成绩。

最初因 Docker 权限受限，只完成数据完整性、离线回归与语法检查；用户处理权限后已补做构建、GPU/Habitat 冒烟、模型 HTTP 与视频回归，详见顶部续作状态。完整导航与正式成绩仍需以新批次文件为准。

曾发生主机 RAM 耗尽导致桌面应用退出。显存足够也应检查 `free -h`；模型占显存不等于评测进程仍在运行。

## 4. 新 agent 的第一轮工作

1. 核对当前分支、远程、工作区和运行进程；保留已有结果，确认 Docker 是否可访问。不要重建环境或自动启动 50/1,000 回合。
2. 阅读下方代码入口，梳理“观测 → 检测/匹配 → 语义地图 → frontier/目标选择 → 动作 → 成功判定”。
3. 核对旧结果及基线可运行性，建立一组固定 episode 清单；不同方法使用同一清单。
4. 延续上方已选定的自适应观测方案，先解锁容器构建与集成验证；不要重新询问算法方向。
5. 每项改动继续用配置开关实现、验证、记录、提交、推送；避免同时替换模型、规划器和评测规则。正式对照冻结代码，在各组之间不得改源码。

## 5. 四阶段路线

| 阶段 | 要做的事 | 完成依据 |
| --- | --- | --- |
| A：基线 | 固定配置和 episodes，确认复现流程 | 可复跑的基线结果、视频、失败分类和代码 SHA |
| B：自定义 ObjectNav | 实现一个明确假设，保留基线开关 | 同 episode 对照与消融，说明收益、代价和退化案例 |
| C：连续控制仿真 | 把高层目标/路径与底盘控制分开，验证速度控制闭环 | 里程计反馈、限速、停止、碰撞及观测延迟测试通过 |
| D：Tracer Mini | 接入实际传感器、定位/TF、底盘驱动和安全控制 | 先静态/离线，再低速短程，最后目标导航任务测试 |

先用 1 回合检查链路，再用固定 10 回合调试、50 回合比较；方法稳定后评估更大且覆盖多场景的集合。前 50 回合是顺序子集，可能集中在少数场景，不能代表完整 val 成绩。

### 评测约定

- 记录代码 SHA、配置、seed、scene/episode ID、模型/权重版本、成功判定、步数上限和输出路径；保持基线与方法一致。
- 报告 SR、SPL、Soft SPL、目标距离、步数、耗时、失败分类，并附代表性视频。新增碰撞率或推理延迟时先补充可靠采集，现有 CSV 不包含的指标不要杜撰。
- 当前配置：最大 500 步、成功距离 0.2 m、前进 0.25 m、转角 30°、半径 0.18 m。它们是仿真参数，不是已确认的实机参数。
- 不新增依赖目标真值位置、最短路或评测成功信号的在线决策；明确现有 GPS/Compass 等理想位姿假设。
- 原始数据、日志和视频留在 `outputs/`；在 `docs/experiments/` 提交简短报告。不要把调参过的 val 子集描述为未见测试集。

## 6. 代码入口

| 功能 | 首先阅读 |
| --- | --- |
| 仿真评测与配置 | `habitat_evaluation.py`、`config/habitat_eval_hm3dv2.yaml` |
| 启动/续跑/结果保存 | `docker/run_batch.py`、`basic_utils/record_episode/experiment.py` |
| 视觉服务与目标处理 | `vlm/server_wrapper.py`、`vlm/utils/get_object_utils.py`、`vlm/utils/get_itm_message.py` |
| Habitat → ROS | `habitat2ros/habitat_publisher.py` |
| 探索与决策 | `src/planner/exploration_manager/src/exploration_fsm.cpp`、`exploration_manager.cpp` |
| 语义/价值/frontier 地图 | `src/planner/plan_env/src/object_map2d.cpp`、`value_map2d.cpp`、`frontier_map2d.cpp` |
| 连续速度仿真 | `habitat_vel_control.py`、`real_world_test_example/real_world_test_habitat.py` |
| 轨迹与 MPC | `src/planner/exploration_manager/launch/exploration_traj.launch.py`、`src/planner/trajectory_manager/src/traj_server.cpp` |

轨迹示例发布 `/cmd_vel`，Habitat 速度控制脚本订阅它。现有 `real_world_test_example` 仍是 Habitat 示例，不能据此声称已经支持 Tracer Mini；连续控制路径本轮未运行验证。

实机前需确认：准确型号/驱动仓库与 ROS 版本、控制接口和运动学、轮廓尺寸、RGB-D/激光/IMU、相机内外参、深度单位、时间同步、`map→odom→base_link→camera` TF、定位来源、速度/加速度限制和急停。默认不假设底盘有可俯仰相机；Habitat 的 look-up/down 动作需要按实际传感器结构重新处理。

仿真沿用专用 ROS domain（启动器默认 79）；接入实机时确认网络和话题隔离。用户尚未授权本交接任务驱动真实机器人。

## 7. 验证与提交工作流

提交规则以 [AGENTS.md](../AGENTS.md) 为准。用户已授权每个完成的逻辑修改自动提交并推送，按文件审阅和暂存，不要把所有工作区内容一并上传。

容器可访问后，优先运行现有无模型推理回归：

```bash
docker exec apexnav-jazzy-egl /opt/apexnav_venv/bin/python docker/test_batch.py
docker exec apexnav-jazzy-egl /opt/apexnav_venv/bin/python docker/test_remote_api.py
```

C++ 改动运行 `./docker/apexnav build`；涉及 Habitat 动作时再运行 `docker/test_habitat_actions.py`（会使用 GPU）。根据改动选择测试，不要把短步数诊断算入正式成绩。

提交前检查 `git diff --cached --check`。依赖补丁的原始上下文空白会触发警告；确认只在 `.patch` 内后可用以下命令检查其他文件，并保留补丁原样：

```bash
git diff --cached --check -- . ':(exclude)docker/patches/*.patch'
git push origin HEAD:refs/heads/codex/objectnav-sim-to-tracer-mini
git ls-remote origin refs/heads/codex/objectnav-sim-to-tracer-mini
```

推送前核对当前分支、`origin` 地址以及本地/远程是否分叉；不强推，不改上游或默认分支。无法运行的测试和推送失败都应明确记录。

本仓库已配置用户 GitHub 名称和隐私邮箱用于提交；新机器应读取自己的正确配置，不从聊天复制密码/令牌。

## 8. 可直接发送给新窗口的提示词

> 请在 `/home/yezhengran/My_projects/ApexNav` 继续我的 ObjectNav 实验。先读 `AGENTS.md` 和 `docs/OBJECTNAV_HANDOFF.zh-CN.md`，核对工作分支 `codex/objectnav-sim-to-tracer-mini`。我的目标是先基于当前 ROS2 Jazzy/Habitat 环境开发自己的目标导航方法，做固定数据集上的基线对照，再迁移到 Tracer Mini。先检查现状、梳理决策链路并给出第一项可检验改进方案。每个完成且经适当验证的逻辑改动都提交并推送到我的 `origin` 同名分支，保留环境基线、原始结果和测试记录。按交接文档区分已验证内容与待核验事项。
