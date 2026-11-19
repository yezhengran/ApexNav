# ApexNav 测试入口

测试集中在本目录，运行算法留在`vlm/`、`habitat2ros/`和`src/`。项目架构见[设计说明](../docs/OBJECTNAV_DESIGN.zh-CN.md)。

## 分组

| 分组 | 文件数 | 覆盖 | 运行影响 |
| --- | ---: | --- | --- |
| `unit/` | 16 | Qwen严格响应/费用/重试、prompt兼容、对象来源、复查/观测/比较状态、离线审计、批处理与有界恢复 | API使用mock或本机loopback；预算、模拟输出在临时目录；不加载模型或执行导航 |
| `cpp/` | 4 | 主/环带/备用观测几何、观测顺序、排序与候选池 | 临时编译纯逻辑断言，不替换安装或构建overlay |
| `integration/` | 3 | 真实ROS节点参数/消息、mapper同步回执、Habitat相机动作兼容 | 要求无活动评测；短时启动规划器并清理，不启动模型或正式导航 |
| `fixtures/` | 1 | legacy/v1的48组提示词哈希 | 兼容基准，不能为让新测试通过而更新旧哈希 |

统一使用`run.py`。加载器仅收集定义在当前模块的TestCase，避免导入的夹具类被再跑一遍；继承用例在不同配置子类上的检查仍保留。默认只运行unit，不会隐式启动ROS。

## 在现有环境运行

主机项目根目录执行，使用既有`apexnav-jazzy-egl`容器：

```bash
docker exec -w /workspace/ApexNav -e PYTHONDONTWRITEBYTECODE=1 \
  apexnav-jazzy-egl /opt/apexnav_venv/bin/python tests/run.py unit

docker exec -w /workspace/ApexNav -e PYTHONDONTWRITEBYTECODE=1 \
  apexnav-jazzy-egl /opt/apexnav_venv/bin/python tests/run.py cpp
```

集成测试需要现有ROS环境和与源码匹配的构建；以下使用移除方向模块后独立重建的overlay。`ROS_DOMAIN_ID=179`隔离测试通信，但仍须确认没有活动评测；入口还会检查容器内相关进程。

```bash
docker exec -w /workspace/ApexNav -e PYTHONDONTWRITEBYTECODE=1 \
  -e ROS_DOMAIN_ID=179 -e ROS_LOG_DIR=/tmp/apexnav-cleanup-ros-tests \
  apexnav-jazzy-egl bash -c \
  'source /opt/ros/jazzy/setup.bash && source outputs/build/semantic-v2-final-20261119/install/setup.bash && /opt/apexnav_venv/bin/python tests/run.py integration'
```

容器内已配置ROS/overlay时可用`python tests/run.py all`依次运行三组，`-v`显示用例。命令应使用该容器的Python虚拟环境；单纯在主机安装Python并不具备Habitat/ROS依赖。

2026-11-18整理后实测：**230项unit、5项integration、5个C++测试程序全部通过**。保留208个显式Python `test_*`方法；运行数量含不同配置子类继承的边界检查，与方法定义数不同。依赖仅出现既有Gym/AMP弃用和插件重复注册提示，没有测试跳过或失败。

2026-11-19移除方向前沿后：**230项unit、4项integration、4个C++测试程序全部通过**，完整六包在`outputs/build/semantic-v2-final-20261119/`独立重建成功。只删除方向模块专用几何/启动测试；原环带、备用观测、候选比较和同步检查保留。当前为23个测试文件、207个显式Python `test_*`方法。没有运行正式导航或付费API。

## 2026-11-18合并与迁移记录

原31个测试文件变为24个；删除旧位置，保留有效断言。没有删除实验原件、冻结控制或运行中的父类。

| 原位置 | 当前归属 |
| --- | --- |
| `vlm/test_qwen_review.py` + `test_qwen_response.py` | `unit/test_qwen_review.py`；响应证据类只复用夹具，不再重复继承整套基础测试 |
| `vlm/test_semantic_prompt.py` + `test_semantic_prompt_v2.py` | `unit/test_semantic_prompt.py`；两种policy仍分别验证 |
| `docker/test_audit_annular.py` + `test_audit_choice.py` + `test_audit_crop_pose.py` | `unit/test_audit.py`；来源、几何、比较与单点边界均保留 |
| `docker/test_choice_launch.py` | `integration/test_planner_launch.py`；方向模块专用启动用例已删除 |
| `docker/test_active_view.cpp` + `test_annular_view.cpp` + `test_crop_pose_view.cpp` | `cpp/test_active_view.cpp`；三个断言组独立函数，由同一main执行 |
| 其他`vlm/test_*.py` | 同名`unit/test_*.py` |
| `docker/test_batch.py`、`test_network_recovery.py`、`test_remote_api.py` | 同名`unit/`文件 |
| `docker/test_observation_barrier.py`、`test_habitat_actions.py` | 同名`integration/`文件 |
| 其余`docker/test_*.cpp`和`docker/tests/test_candidate_choice.cpp` | 同名`cpp/`文件 |
| `vlm/semantic_prompt_compatibility.json` | `fixtures/semantic_prompt_compatibility.json` |

历史实验记录中的旧命令/路径是当时的事实，未批量改写；需要复现对应来源时使用历史提交或源码快照。新结构下不再使用`python -m vlm.test_…`等旧入口。

## 后续维护

- 新Python测试放相应分组、命名`test_*.py`；新纯C++断言放`cpp/test_*.cpp`。不要把正式评测或付费Qwen请求放进unit。
- 夹具可复用；不为缩减文件数删除仍被最终版继承的功能检查。不同阶段policy/开关是兼容合同，应保留正反例。
- 如新增了公共测试支持函数，再提取到独立helper；当前以少量夹具导入复用，不为每个历史版本增加新运行入口。
- C++逻辑改动后既要测试，也要重新构建独立overlay，不能仅因旧二进制集成测试通过就认为新源码已验证。
- 仓库源码摘要会统计测试文件，路径迁移也会改变摘要；本次整理不改变历史provenance，不允许恢复旧冻结控制器。
