# 自适应观测：容器集成与内存验证

## 目的与代码

验证自适应观测首版能否在现有 ROS2 Jazzy / Habitat / RTX 4060 环境运行，为固定 50 回合对照准备。算法代码为 `55c11c6b0c4829439ecfdf1329000405a2977143`，基线行为仍由 `baseline` 开关保留。本次另做模型服务启动内存回收，不改变权重、精度、输入、阈值或导航策略；正式运行代码以各目录 `provenance.json` 为准。

用户在主机处理 Docker 授权后，本会话可以在沙箱外访问现有 `apexnav-jazzy-egl`，无需新建环境。主机 RAM 15.3 GiB、无 swap；GPU RTX 4060 Laptop 8188 MiB，驱动 580.178.04。

## 已完成的实际集成检查

- ROS2 六包构建成功，使用 sequential executor、`CMAKE_BUILD_PARALLEL_LEVEL=2` / `MAKEFLAGS=-j2`，共约 44.6 秒。仅见既有 rclcpp QoS API 弃用警告。
- 容器内 `docker/test_batch.py` 8 项通过，包含视频编码；`docker/test_remote_api.py` 8 项通过；`docker/test_habitat_actions.py` 2 项通过。
- `./docker/apexnav smoke` 返回 `HABITAT_ROS_SMOKE_OK`：真实场景 `4ok3usBNeis`、navmesh 可用，ROS 收到 15 条深度消息，状态 `[0,3]`、返回动作 4。
- 四个本地模型启动后，`./docker/apexnav vlm-check` 真实 HTTP 推理全部通过。BLIP2 / YOLO / SAM / DINO 单次小图接口检查约 0.58 / 0.05 / 0.36 / 0.34 秒；这不是导航平均延迟。
- 容器内源码核验得到代码 `55c11c6`、指纹 `aaa1920d80fda30aa9dbc85e7954aa1640b29a8c74f17f0c0e8c7c2935b8243c`，与主机一致。

以上检查没有完成 ObjectNav episode，不能计算 SR/SPL，也不能证明改进有效。

## 内存阻塞与修复验证

首次同时启动四模型后容器约占 6.1 GiB，GPU 约占 6829 MiB；HTTP 推理后主机 available 曾降到 808 MiB。启动导航前暂停了本次模型服务，没有用 OOM 风险换取评测进度。用户随后关闭浏览器和 VS Code；进程检查未发现 `msedge` 或 `code` 残留。

实测 YOLO 的 CPU tracing 产生较多已释放但未返还操作系统的堆内存。现有 `APEXNAV_VLM_RELEASE_IDLE_CACHE=1` 开关增加一次启动后的 `gc.collect()` 和 glibc `malloc_trim(0)`；不支持该函数的系统跳过，不在每帧执行 CPU 回收，GPU 缓存处理保持原状。

四模型分别在独立进程、相同真实 Habitat RGB 帧上比较回收前后的输出，检测框/类别/分数、SAM mask/IoU、BLIP2 cosine/ITM 均逐项完全相同。RSS 为单进程测量，不能简单相加作为全机节省量。

| 模型 | 推理后回收前 RSS (MiB) | 回收后 RSS (MiB) | 下降 (MiB) |
| --- | ---: | ---: | ---: |
| YOLOv7 | 3140.7 | 1880.6 | 1260.2 |
| MobileSAM | 1425.4 | 1425.1 | 0.3 |
| GroundingDINO | 1880.3 | 1871.5 | 8.8 |
| BLIP2 | 1618.6 | 1617.5 | 1.2 |

这是单帧等价性与内存测量，不保证所有场景的内存峰值。第一次临时验证脚本误将 SAM 的 `(mask, IoU)` 元组当作单数组，修正脚本后通过；产品代码没有为此改动。临时脚本验证后删除。

修复后再次通过 8 项 HTTP 协议回归；重新启动全部四个服务并通过真实 HTTP 推理，主机 available 约 5.3 GiB，GPU 6571/8188 MiB。该余量同时受用户关闭其他应用影响，不能将全机变化全部归因于代码。

## 保存位置与下一步

- 构建日志、四模型内存 JSON：`outputs/validation/20261105-integration/`。
- 离线和内存测试日志：`experiment_notes/validation/20261105-adaptive-view/`，均已被 Git 忽略。
- 继续 10 回合排障，再冻结源码，用相同 seed=100、顺序 episode 集、500 步、视频设置分别运行 baseline/adaptive_view 各 50 回合。
- 每回合保存指标与实际 episode 身份，每 10 回合保存检查点；部分结果明确报告完成数量。新批次不续接旧无 provenance 的 27/50 数据。

## 完整单回合（已完成）

`outputs/runs/av-smoke-1/`：代码 `cbe50fe1dbe63738595494e4499419967646f3b6`，运行源码指纹 `bfc60bd7e915e8fea2f4004bf1f1e2aa55a839200a601d798246b05df021a6d9`；`adaptive_view`、seed=100、500 步上限、视频开启。实际场景 `4ok3usBNeis`、episode ID `0`、目标 `bed`。

- 完成 1/1、success；345 步、96.73 秒，SPL=13.7467%、Soft SPL=13.6802%、终点距离=0.0263 m。
- 额外候选验证 VLM 调用 9 次（合计 0.917 秒）、接受 1 个、拒绝 6 个；主动观测尝试 6 次、动作 68 次。包含真实移动与到点当前 crop 复核，证明新增链路已执行，不代表效果改善。
- 失败证据：candidate 4 的主动 crop 可见床，ITM=0.5207，低于初始 0.65 阈值而被拒绝；candidate 7 的历史 crop 得到 0.7147 被接受。当前存在误拒绝和同一床的部分几何被拆成不同候选的迹象，需在 10 回合及同集基线中检查代价；此处不凭单样本调整阈值。
- `summary.json`、`episodes.csv`、`checkpoint-0001.json`、实际 episode 清单、验证事件/裁剪图与成功 MP4 均已保存；启动器状态 `completed / 0`。导航期间 available 约 2.6–2.7 GiB，未见 OOM。

## 旧原型 10 回合诊断：9 回合后停止

`outputs/runs/av-debug-10/`：代码 `9edbd17c4c6149d017c1ee118b73efd57c39fb8b`，源码指纹与上述单回合一致；`adaptive_view`、ITM 阈值 0.65、seed=100、500 步、视频开启，场景 `4ok3usBNeis`。完成 episode ID `0`–`8`，第 `9` 个 ID 的尝试未完成。状态 `interrupted / 130`，不能称为完整 10 回合成绩。

| 完成数 | 成功数 | SR | SPL | Soft SPL | 平均终点距离 | 平均步数 | 平均回合秒数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9/10 | 2 | 22.22% | 8.27% | 15.51% | 3.9286 m | 339.0 | 103.06 |

结果分类：success 2、`[stepout] false negative` 2、`no frontier` 2、`stepout feasible` 1、`[no frontier] false negative` 2。首个 bed 回合重复运行 352 步（单回合为 345），表明 ROS 异步链路下同 seed 并非逐动作确定。

代表性证据：episode ID 1 的主动 crop 清楚可见马桶，ITM=0.5023 被拒；episode ID 4 的主动 crop 清楚可见沙发，ITM=0.3940 被拒。简单降到 0.5 也不能解决全部误拒绝。episode ID 3 的 chair 回合没有任何候选/VLM 调用而因 no frontier 结束，不能将所有失败归因于阈值。

用户已收窄方法范围：仅增加观测/重新验证，不改变 ApexNav 探索策略或方式，后续方法变更前需先提问对齐。当前原型前置过滤目标点云、禁用部分兜底目标，超出该范围；因此保留为失败原型证据，不再扩展其 50 回合。停止时即使最后回合成功也最多 3/10，无法达到用户确认的门槛：**新方法至少成功 8/10，且不低于同集基线，才扩到 50 回合。**

原始汇总、CSV、已完成视频、候选裁剪图和事件均保留；最后 `.partial.mp4` 不是完成结果。先验证同集 baseline 10 回合，并与用户对齐 STOP 复核触发、一次观测及不确定结果处理，再实现新范围。

## 同集基线 10 回合（已完成）

`outputs/runs/av-baseline-debug-10/`：代码 `6ae4ed883df7bbad96c9454415f259af11a7b49b`，运行源码指纹 `bfc60bd7e915e8fea2f4004bf1f1e2aa55a839200a601d798246b05df021a6d9`；baseline、seed=100、500 步、视频开启。实际场景 `4ok3usBNeis`，episode ID `0`–`9`，状态 `completed / 0`。

| 完成数 | 成功数 | SR | SPL | Soft SPL | 平均终点距离 | 平均步数 | 平均回合秒数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10/10 | 8 | 80.00% | 50.15% | 50.83% | 2.0256 m | 157.8 | 232.72 |

失败：ID 3（chair）无前沿，ID 7 误停。与旧原型的共同 9 回合严格比较通过同源码指纹、同设置和实际 episode 身份检查：baseline 7/9、旧原型 2/9，SPL 51.53% 对 8.27%。这只是旧原型的部分配对诊断，不代表新方案成绩。基线部分动作明显较慢，当前不能将墙钟耗时差归因于方法；SR/SPL 更有解释力。

每回合汇总、CSV、10 回合检查点、实际 episode 清单和视频已保存。可读备份与共同前缀比较 JSON 在 `experiment_notes/validation/20261105-adaptive-view/baseline-debug-10/`。新方案会改变运行源码指纹，后续需在新冻结源码上重跑同集基线；用户确认的扩展门槛仍为新方法成功至少 8/10 且不低于对应基线。
