# 从 Epoch 5 以 Object LR 2e-4 重训及 SIGHUP 修复

- 时间：2026-08-21 23:48（Asia/Shanghai）
- 实验目录：`/root/autodl-tmp/Lain/UO_lain5_object15_dynamic_retry20_seed66`
- 指定起点：`ckpt_23370_05.pt`
- W&B Run ID：`j7q84llg`

## 真实停训原因

最后一次训练在 2026-08-21 17:47:47 收到 `SIGHUP`。`torchrun` 会注册自己的 SIGHUP handler，因此普通 `nohup ... &` 仍可能在 SSH 会话结束时被关闭。此前反复恢复第 7 轮但没有完整接受新 epoch，造成实际训练进度停留在第 6 轮。

## 本次恢复

1. 备份现有第 6 轮及全部状态/诊断文件。
2. 将 staged policy、diagnostics、attempt 表恢复到第 5 轮已接受状态。
3. 从 `ckpt_23370_05.pt` 复制生成续训 checkpoint，仅把 Object optimizer LR 设置为未减半的 `2e-4`。
4. Head LR 保持 checkpoint 内的 `2.5e-4`，ViT LR 保持 `1e-4`。
5. 从逻辑第 6 轮重新训练；新第 6 轮完成后覆盖/更新对应本地 epoch 记录。
6. W&B 使用 `resume_from=j7q84llg?_step=9` 回退到第 5 轮之后，再在同一个 Run 内续写，删除旧第 6 轮和失败恢复留下的远端 history 尾部。
7. 使用 `setsid` 创建独立 session，并将 stdin 重定向到 `/dev/null`，避免 SSH 断开再次向 `torchrun` 发送 SIGHUP。
8. 保留每 240 秒一次的 `runtime/*` 活跃记录，防止 W&B 把长 epoch 误判为 crashed；accepted 指标和本地 JSON 仍只在 epoch 评估通过后写入。

## 连贯性

- 本地 JSON/CSV/JSONL 保留第 1～5 轮，随后从第 6 轮继续追加。
- W&B 保留第 1～5 轮 history，随后从 step 10 继续写入。
- 原文件在实验目录下的 rollback backup 中保留，可恢复。
