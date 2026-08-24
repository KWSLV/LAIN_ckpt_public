# W&B 续训心跳 Crash 修复记录

- 修改时间：2026-08-21 17:30:56（Asia/Shanghai）
- 实验：`UO_lain5_object15_dynamic_retry20_seed66`
- W&B Run ID：`j7q84llg`
- 影响范围：仅 staged policy 的同目录续训模式

## 现象

服务器训练主进程和 GPU 持续工作，W&B Core 的 file-stream 请求也持续返回 HTTP 200，但 W&B API 的 `heartbeatAt` 只停留在续训初始化时间。约 10 分钟后，网页把仍在运行的 Run 标记为 `crashed`。

## 原因

W&B 系统指标约每 15 秒产生 file-stream chunk。这些 chunk 抑制了客户端的显式空 heartbeat，但在该次 crashed Run 的恢复场景中没有刷新服务端 `heartbeatAt`，导致网页错误判断训练失联。

## 修改

在 `main.py` 的 staged 同目录续训 W&B 初始化中显式设置：

- `heartbeat_seconds=30`
- `x_stats_sampling_interval=60.0`

系统指标仍保留，但采样周期长于 heartbeat 周期，使 W&B Core 能发送显式 keepalive。该修改不增加训练历史行，不改变 Unseen/Seen/Full、贡献度或 LR 曲线的 epoch 记录结构。

首次验证发现 W&B Core 确实发出了 `total_files=0` 的显式 keepalive，服务器返回 HTTP 200，但该恢复 Run 的服务端 `heartbeatAt` 仍未刷新。因此补充应用层活跃记录：

- staged 同目录续训每 240 秒写入一条 `runtime/*` W&B history。
- 字段为 in-progress logical epoch、iteration、retry count 和三个当前 LR。
- 不使用 `logical_epoch`、Unseen/Seen/Full 或 `module_contribution/*` 字段，因此不进入 accepted-epoch 正常曲线。
- 本地 `scene_gate_diagnostics.json`、`staged_retry_attempts.jsonl` 和失败 retry 规则完全不变。

该记录的目的仅是让 W&B 服务端实际接收活跃 history，防止恢复 Run 因服务端 heartbeat 状态异常再次被标记为 crashed。

## 恢复约束

- 从第 6 轮已接受 checkpoint 重新训练逻辑第 7 轮。
- Object LR 保持 `2e-4`，Head=`2.5e-4`，ViT=`1e-4`。
- 继续使用原日志、原 JSON/JSONL、原输出目录和原 W&B Run ID。
- 第 7 轮此前未完成且未被写入 accepted 表格，因此重新训练不会覆盖已接受数据。
