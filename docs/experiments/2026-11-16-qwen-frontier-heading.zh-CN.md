# Q8 可选前沿目标朝向

2026-11-16用户要求按已对齐的方案实现探索阶段前沿朝向，支持开关，并提交同一Q8分支。基于`17d65e1`，分支`codex/objectnav-annular-top5`。本次没有启动新的成功率评测，不修改Q8-first100／window400–500／demo447原结果。

## 行为与边界

- 先运行原前沿选择、TSP／路径规划与目标稳定逻辑；选定最终探索目标后，将方向与**该目标对应的前沿簇**绑定，匹配同时导出的前沿平均点和簇，不拿全图其他边界替代。休眠前沿回退也匹配自身簇。
- 在可见、未被障碍或更近未知区域遮挡的对应边界上，选择角覆盖最宽的连续区间并面向角区间中心。角覆盖采用0.5度分桶、按地图分辨率扩展格子的角宽，正确处理±180度跨界。
- ApexNav目标通常就是前沿簇均值；当目标落在边界上，直接朝均值会得到零向量或沿边界切线。此时使用从相邻已知自由格指向未知侧的边界法向。没有有效方向则不追加转向。
- 途中仍沿原路径朝向行走；距离该探索目标小于0.5米时才执行30度离散转向。朝向误差不超过15度时不额外转；最多6个转向动作（覆盖最短180度转向），没有平移／抬头／低头。
- 到达时及每次转向前重新核对保存方向所对应的边界仍未知且可见。该段已被观测／遮挡则取消；不能因为同簇另一段仍未知就继续朝已知区域转。附近0.75米内不重复扫描，防止每次重规划重新消耗转向预算。
- 只在EXPLORE分支追加动作。物体靠近、Qwen复查、主动观测均优先并取消待执行朝向。停止／碰撞／脱困／环带规划／Top5和Qwen判定均不变。关闭时不进入新的朝向逻辑。

这是一项可独立比较的探索朝向开关，不引入Q7策略，不更改真实融合分数或候选排序。它仍消耗正常导航动作预算；本次离线与接口测试不能证明SR提升。

## 开关与独立构建

默认关闭；命令行`--frontier-look`开启，`--no-frontier-look`明确关闭，省略也关闭。ROS launch对应`frontier_look:=true/false`，实际节点参数`navigation.frontier_look`。

运行元数据仅开启时记录`frontier_look.policy=goal_bound_widest_visible_boundary_v2`、6次转向上限及0.5米到达范围。旧的`visible_unknown_boundary_gain_per_turn`记录仍代表旧实现，不覆盖历史协议。

已构建新overlay：`outputs/build/frontier-oriented-20261116/install`，只重编译exploration_manager并依赖完整Q8六包overlay。原`outputs/build/annular-top5-full-20261115/install`633文件保持。将来用新开关时必须加载新构建，不能只改Python开关却仍使用旧二进制。

容器内加载顺序（只加载环境，不启动评测）：

```bash
cd /workspace/ApexNav
source /opt/ros/jazzy/setup.bash
source /opt/apexnav_venv/bin/activate
source outputs/build/annular-top5-full-20261115/install/setup.bash
source outputs/build/frontier-oriented-20261116/install/local_setup.bash
ros2 pkg prefix exploration_manager
```

随后在同一版本的独立新批次命令中追加`--frontier-look`或`--no-frontier-look`。旧控制器／started目录／输出目录保持冻结，不用它们续跑不同源码。Q8原复查配置仍为`linked_annular`、seed100、500步、step_sync、400步后候选靠近。

## 验证与来源

- 前沿C++几何与生命周期回归：最大连续角区间、跨±180度、遮挡、范围外／空输入、边界法向、途中不转、到达转向、目标微移、已观测方向取消、同簇其他边界不能续命、物体优先取消、重复抑制、无yaw进展仍有6动作上限。ASan／UBSan通过；当前沙箱ptrace不支持LeakSanitizer，运行时关闭泄漏检测，不声称做过泄漏验证。
- 34项Python调度／Q8多图来源审计回归通过；显式关闭与默认关闭的spec和评测命令一致，开启不改变Habitat/Qwen参数。固定dry-run输出避免测试受秒级时间戳影响。
- 独立ROS域的真实节点开／关参数验证及原mapper同步观测握手2项通过。新exploration_node的动态库解析与独立Release构建通过；未调用视觉模型／Qwen，未运行导航成绩测试。
- 新运行源191文件，指纹`fee38e45086d56c7b0259f18feecc1955a076e70036ff014839edd99330c78c2`；增加的文件是ROS开关回归。实施记录在`experiment_notes/qwen-frontier-oriented-20261116/`。
- 原安装41、权重词表18、旧Q8 overlay633、既有Q2/Q8证据53311及旧控制17文件冻结核验通过。新分析与构建均使用独立目录。

同次用户请求的27例历史失败分析见[400–500失败报告](2026-11-16-qwen-annular-window-failures.zh-CN.md)，仅归因、类别与距离，没有追加算法修改意见。
