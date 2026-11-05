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
- 尚无本轮导航成绩。先完成完整单回合、10 回合排障，再冻结源码，用相同 seed=100、顺序 episode 集、500 步、视频设置分别运行 baseline/adaptive_view 各 50 回合。
- 每回合保存指标与实际 episode 身份，每 10 回合保存检查点；部分结果明确报告完成数量。新批次不续接旧无 provenance 的 27/50 数据。
