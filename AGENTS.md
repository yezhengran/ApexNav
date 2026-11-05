# ApexNav 实验协作规则

开始工作前阅读 [新窗口交接](docs/OBJECTNAV_HANDOFF.zh-CN.md) 和 [实验指令](docker/experiment_commands.zh-CN.md)。

## 用户目标与范围

- 先在 Habitat/HM3Dv2 上实现用户自己的 ObjectNav 方法，完成基线对照和消融；再迁移到 Tracer Mini 实机。
- 保留 ROS2 Jazzy、现有 Docker 环境和可复现基线；具体算法创新点尚未选定。
- 本地视觉模型优先，显存不足时可使用已有 HTTP 接口外置部分模型；更换模型属于独立实验变量。
- 回复、实验记录与交接文档以简洁中文为主。

## 分支与提交

- 工作分支：`codex/objectnav-sim-to-tracer-mini`。
- 推送目标仅为 `origin`：`git@github.com:yezhengran/ApexNav.git`；`upstream` 仅作参考。
- 用户已授权：每个完成且通过适当验证的逻辑改动，应主动 commit 并 push 到上述分支，无需每次重复询问。纯解释/诊断请求不因此变成修改请求。
- 动手前核对分支、远程、工作区与暂存区；保留并区分用户未提交的工作。只按明确文件路径暂存本次修改，不盲目 `git add .`。
- 提交前审阅 diff、做相关验证，检查凭据与大文件；不要上传数据集、权重、视频、日志、`config/vlm.env` 或本机密码/令牌。
- 不向 `main`、`ros2-jazzy`、`dev/tracer-mini` 或 `upstream` 推送；不 force push、不重写已发布提交、不自行合并分支。
- 正常推送命令：`git push origin HEAD:refs/heads/codex/objectnav-sim-to-tracer-mini`。先确认当前 HEAD 就在目标工作分支。
- 推送完成后核对远程 SHA；最后报告 commit、测试和推送结果。推送失败须明确说明，保留本地提交，不声称已同步。

## 实验与运行

- 基线提交为 `3828f38`；算法变更采用可配置开关，保留基线行为，并使用独立输出目录。
- 同时只跑一份评测。启动前检查现有进程、GPU、主机 RAM；新窗口先完成交接核验，不自动开启长时间评测。
- 代码或配置修改会通过 bind mount 立即影响容器；不得在一组对照实验中途修改其运行代码。C++ 改动需要重新构建。
- 不把未完成、诊断用低步数上限或不同 episode 集合的结果当作正式对照结果。
- 每组实验在 `docs/experiments/` 保存简洁 Markdown：假设、代码 SHA、配置/seed、episode 集、指标、失败案例和本地结果路径；原始输出保留在被忽略的 `outputs/`。
- 现有 `.patch` 文件保留依赖源码的原始上下文空白。不要为消除格式警告破坏补丁；其他文件正常执行 diff 空白检查。
- 不改动用户的 Docker 授权或系统权限配置来绕过权限问题。权限受限时继续可做的检查，并准确记录未验证部分。
- 实机型号、传感器与驱动接口尚待确认；在用户明确要求运行实机且完成接口、急停和限速核验前，不发送真实底盘运动指令。
