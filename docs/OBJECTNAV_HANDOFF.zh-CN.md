# ObjectNav：仿真开发到 Tracer Mini 的新窗口交接

更新日期：2026-11-05。先读根目录 [AGENTS.md](../AGENTS.md)，再按本文继续。

## 本轮续作状态：自适应观测实验

用户已选定迁移 [DynZSON](https://github.com/yezhengran/DynZSON) 的尺寸自适应观测并用 VLM 确认，授权方案完成后做固定 50 episode 对照；不再需要询问是否选择此方向。详细假设、实现边界、离线验证、旧结果和命令见 [首轮实验记录](experiments/2026-11-05-adaptive-view-plan.zh-CN.md)。

- 当前已实现默认关闭的 `baseline / vlm / fixed_view / adaptive_view` 设置。复用 BLIP2；没有迁入 DynZSON 整套导航，也没有更换视觉模型。入口为 `vlm/active_view.py` 与 C++ `exploration_manager/active_view.h` / FSM。
- 保存逐回合指标、每 10 回合检查点、验证 crop/分数/位姿和实际 episode 清单；`docker/compare_runs.py` 检查同集同配置源码的共同前缀。新批次源码需提交，旧无 provenance 的目录不可用新启动器续跑，以免混入新版本。
- 已通过 12 项离线确认逻辑、独立 C++17 观测点/路径安全、批量记录/对照及语法/空白检查。临时测试脚本验证后清理，记录在被忽略的 `experiment_notes/validation/20261105-adaptive-view/`。
- 用户已在主机完成 Docker 授权；本会话可在沙箱外访问现有容器。`55c11c6` 已完成真实 ROS2 六包构建（并发限制为 2）、容器内批量/视频 8 项、HTTP 协议 8 项、Habitat 动作 2 项回归；真实场景 `HABITAT_ROS_SMOKE_OK` 和四个模型的 HTTP 推理通过。
- 完整 `adaptive_view` 单回合已成功：`av-smoke-1`、代码 `cbe50fe`，345 步、SPL=13.75%，9 次额外 VLM 调用，存在真实床 crop 被拒绝的证据。**10 回合及两组 50 episode 仍待完成，尚不能宣称算法改善。** 已验证启动 CPU 缓存回收；用户关闭其他应用后四模型可用，导航期间 available 约 2.6 GiB。集成、内存与单回合记录见 [续作验证](experiments/2026-11-05-integration.zh-CN.md)。
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
