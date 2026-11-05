# ApexNav 实验指令速查

以下命令在 **Ubuntu 主机终端**执行，按需选用，不要整篇粘贴。不要同时启动两份评测。

## 1. 准备与命令区别

每个新终端先执行；查看其他实验时修改 `APEX_RUN`：

```bash
cd ~/My_projects/ApexNav
export APEX_RUN=hm3dv2-local-50
```

| 写法 | 含义 |
| --- | --- |
| `docker ps` | 系统 Docker 命令，查看运行中的容器 |
| `docker exec 容器名 命令` | 从主机请求在容器内执行命令 |
| `./docker/apexnav ...` | 项目脚本，自动调用 Docker、配置 Python/ROS 环境 |
| `./docker/play-video ...` | 主机播放器脚本，不使用容器 |

`docker/` 只是项目文件夹，`cd docker` 不会进入容器。`./` 表示当前目录。
主机项目路径是 `~/My_projects/ApexNav`，容器内对应 `/workspace/ApexNav`，共享同一份挂载文件。

## 2. 启动或续跑

先启动容器并检查旧实验：

```bash
./docker/apexnav start
docker exec apexnav-jazzy-egl pgrep -af '^python docker/run_batch\.py( |$)'
```

有输出时先核对旧实验，勿重复启动；无匹配时无输出、返回 1 是正常的。确认没有其他评测，再启动模型：

```bash
./docker/apexnav vlm-start-local
./docker/apexnav vlm-check --health-only
```

长实验建议先用 `tmux new -s apexnav-next` 创建主机会话，再执行下面 **二选一** 的命令。

**新实验**：自动创建带时间戳的目录，记下启动输出的目录名。

```bash
./docker/apexnav run-batch hm3dv2 --episodes 50 --video
```

**续跑原实验**：以下仅适用于由新启动器创建、带 provenance 的基线批次。总目标仍是 50 回合，已完成回合保留，未完成回合重试。

```bash
./docker/apexnav run-batch hm3dv2 --episodes 50 --video \
  --method baseline --seed 100 --output outputs/runs/av-baseline-50 --resume
```

续跑须保持原数据集、回合数和视频等设置；更换模型或参数请新建实验。

2026-11-05 起，新批次还记录 method、seed、源码指纹及实际 episode 顺序。工作区必须提交；续跑需保持这些设置一致。旧 `hm3dv2-local-50` 没有 provenance，保留结果，使用新目录重跑，不用新启动器混合续跑。

尺寸自适应观测的正式对照在构建和短验证通过后顺序运行（细节见 [实验记录](../docs/experiments/2026-11-05-adaptive-view-plan.zh-CN.md)）：

```bash
./docker/apexnav run-batch hm3dv2 --episodes 50 --method baseline --seed 100 --video --output outputs/runs/av-baseline-50
./docker/apexnav run-batch hm3dv2 --episodes 50 --method adaptive_view --seed 100 --video --output outputs/runs/av-adaptive-50
python3 docker/compare_runs.py outputs/runs/av-baseline-50 outputs/runs/av-adaptive-50
```

消融设置为 `--method vlm`（只做历史确认）、`--method fixed_view`（固定边缘净距 0.90 m）。各组使用独立目录。每回合保存 summary/CSV，每 10 回合另存 checkpoint；不足 50 回合的比较会标记未完成。

tmux：`Ctrl+B` 松开后按 `D` 离开但保持运行；`tmux ls` 查看；`tmux attach -t apexnav-next` 返回已有会话。
不要在已有会话中重复启动实验。tmux 不能防止内存不足、重启或休眠。

## 3. 监控进度和资源

下面的持续监控命令各开一个终端，按 `Ctrl+C` 只退出监控：

```bash
# 实时评测日志
tail -n 60 -f "outputs/runs/$APEX_RUN/console.log"

# 完成回合数、成功率、SPL（每回合结束后更新）
watch -n 5 "docker exec apexnav-jazzy-egl cat /workspace/ApexNav/outputs/runs/$APEX_RUN/summary.json"

# GPU、显存、温度
watch -n 1 nvidia-smi

# 容器 CPU、内存
docker stats apexnav-jazzy-egl

# 主机内存，重点看 available
watch -n 2 free -h
```

`top` 查看全机进程：`P` 按 CPU 排序，`M` 按内存排序，`q` 退出。
CPU 超过 100% 可表示使用多个核心；模型占显存不代表评测仍在运行。

## 4. 中断实验

原实验启动终端按一次 **Ctrl+C**，等待清理；在日志监控终端按无效。

原终端丢失时，先查看；**确认只有一个且就是要停止的批量实验后**，才执行中断命令：

```bash
docker exec apexnav-jazzy-egl pgrep -af '^python docker/run_batch\.py( |$)'

# 仅匹配批量启动器，不要改成 pkill python
docker exec apexnav-jazzy-egl pkill -INT -f '^python docker/run_batch\.py( |$)'
```

等待几十秒，再检查状态和进程：

```bash
docker exec apexnav-jazzy-egl cat "/workspace/ApexNav/outputs/runs/$APEX_RUN/status.json"
docker top apexnav-jazzy-egl
```

正常中断为 `interrupted` / `130`，评测和 ROS 规划器应退出。`status.json` 在退出时才更新，运行中可能是旧值。
已完成结果保留，`.partial.mp4` 不是完成结果。中断评测**不会停止模型服务**。

## 5. 停止模型、释放显存

**确认所有评测和规划器退出后**，再执行：

```bash
./docker/apexnav vlm-stop-local
nvidia-smi
```

维护时可只停止一个模型：

```bash
./docker/apexnav vlm-stop-local --service blip2
```

服务名：`blip2`、`yolo`、`sam`、`dino`。重新使用时按第 2 节启动模型。
停止服务不删除权重和结果；桌面等应用仍可能占少量显存。容器已停止则无需再停模型。
模型日志位于 `outputs/runtime/`，例如 `tail -n 60 outputs/runtime/blip2.log`。

若连容器也不再需要，最后可执行：

```bash
docker stop -t 30 apexnav-jazzy-egl
```

## 6. 观看视频

**在主机播放，不需要启动容器或模型**：

```bash
# 播放一个已完成回合
./docker/play-video "outputs/runs/$APEX_RUN/success/4ok3usBNeis.basis.glb_0.mp4"

# 将所有成功视频加入播放列表
./docker/play-video "outputs/runs/$APEX_RUN"/success/*.mp4

# 打开结果文件夹
xdg-open "outputs/runs/$APEX_RUN"
```

空格暂停，左右键跳转，`.` 逐帧，`[` / `]` 调速，`>` 下一视频，`q` 退出，`s` 截图。
截图在 `experiment_notes/screenshots/`。不要播放 `.partial.mp4`；内存紧张时少开窗口。
播放器已本地部署；若以后需系统级安装，可执行 `sudo apt-get install --no-install-recommends mpv`，然后使用 `mpv "视频路径"`。

## 7. 保存结果

实验结束后导出，避免容器文件权限导致主机无法读取：

```bash
mkdir -p "experiment_notes/$APEX_RUN"
docker exec apexnav-jazzy-egl cat "/workspace/ApexNav/outputs/runs/$APEX_RUN/summary.json" > "experiment_notes/$APEX_RUN/summary.json"
docker exec apexnav-jazzy-egl cat "/workspace/ApexNav/outputs/runs/$APEX_RUN/episodes.csv" > "experiment_notes/$APEX_RUN/episodes.csv"
```

完整备份（含配置、日志、视频；先确认磁盘空间）：

```bash
docker exec apexnav-jazzy-egl tar -C /workspace/ApexNav/outputs/runs -czf - "$APEX_RUN" > "experiment_notes/${APEX_RUN}-$(date +%Y%m%d-%H%M%S).tar.gz"
```

`summary.json` 是总体指标，`episodes.csv` 是逐回合结果；成功率、SPL 为 0～1。
个人备注可保存到 `experiment_notes/$APEX_RUN/`。前 50 回合只是顺序子集，不代表完整验证集。

## 8. 常用 Docker 命令

```bash
docker ps -a                         # 查看全部容器及状态
docker images                       # 查看镜像
./docker/apexnav shell              # 进入已配置环境的容器终端
```

进入容器后用 `exit` 返回主机，不会停止整个容器。
本容器重启后，模型和实验不会自动恢复，需按第 2 节重新启动或续跑。
实验日志看 `console.log`，不是 `docker logs`；镜像备份也不包含主机挂载的实验结果。

**日常顺序：启动容器 → 启动模型 → 新实验或续跑 → 查看结果 → 安全中断/完成 → 停模型 → 可选停容器。**
