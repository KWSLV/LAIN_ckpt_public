# a3all/a4all 每轮 Checkpoint 与 W&B 连续续跑记录

- 修改时间：2026-08-12 00:27（Asia/Shanghai）
- 服务器代码目录：`/root/Lain/LAIN-main`
- 当前续跑目录：`/root/autodl-tmp/Lain/UO_a4all_seed66`
- 当前 tmux：`lain_a4all_s66`
- W&B 在线 run ID：`k1rgp1f6`
- W&B 页面：`https://wandb.ai/saaatkj-null/LAIN/runs/k1rgp1f6`

## 1. 用户要求

1. a3all 和 a4all 改为每一轮都保留 checkpoint。
2. 当前运行中的 a4all 修改后继续训练。
3. W&B 曲线必须续接到原图表，不创建第二个在线 run。
4. checkpoint、诊断结果和训练日志继续写入原目录/原文件体系。

## 2. 代码修改

### `scripts/training/UO_a3all_a4all_common.sh`

- 将 `--keep-last-checkpoints 1 --keep-epoch-checkpoints 5 10 15 20` 改为 `--keep-last-checkpoints 0`。
- `keep_last_checkpoints=0` 且没有里程碑列表时，`engine.py` 不执行 checkpoint 删除，因此每轮保存的 `ckpt_*.pt` 永久保留。
- 完成校验由固定检查 5/10/15/20 改为检查保留起始轮至 epoch 20 的每一轮。
- 新增 `checkpoint_retention_start_epoch.txt`，用于记录旧任务切换保存策略后的可验证起始轮。
- 新增剩余磁盘空间估算：按最新 checkpoint 大小估算剩余轮次所需空间，并预留 2 GiB，空间不足时拒绝启动，避免训练后期写盘失败。
- 兼容旧 `args.txt` 中 `keep_last_checkpoints=1`、`keep_epoch_checkpoints=[5,10,15,20]` 的一次性迁移。
- 恢复训练时优先读取 `wandb_run_id.txt`，否则从 `wandb/wandb/latest-run` 识别原 run ID；设置 `WANDB_RUN_ID` 和 `WANDB_RESUME=must`。

### `main.py`

- `wandb.init()` 现在显式读取环境中的 `WANDB_RUN_ID` 和 `WANDB_RESUME`。
- 恢复训练时传入 `id=k1rgp1f6, resume=must`，保证在线数据继续写入同一个 W&B run。

### 后台脚本

- `UO_a3all_background.sh` 与 `UO_a4all_background.sh` 会优先复用运行目录中最新的 `train_*.log`。
- 当前 a4all 恢复后继续追加到原日志：
  `/root/autodl-tmp/Lain/UO_a4all_seed66/logs/train_20260811_090204.log`
- 没有生成第二份训练日志。

## 3. 当前 a4all 的迁移与恢复

修改前：

- epoch 5 已完整训练、评估并保存为 `ckpt_19475_05.pt`。
- epoch 6 只运行约 25/3895 个训练 iteration，尚未产生 epoch 6 诊断或 checkpoint。
- 旧轮转策略已删除 epoch 1–3 的 checkpoint，无法恢复。
- `best_unseen.pt` 对应 epoch 4，元数据记录原文件名 `ckpt_15580_04.pt`。

执行操作：

1. 使用硬链接把 `best_unseen.pt` 恢复为 `ckpt_15580_04.pt`，两者 inode 相同，不增加约 703 MiB 的物理占用。
2. 写入 `checkpoint_retention_start_epoch.txt=4`。
3. 写入 `wandb_run_id.txt=k1rgp1f6`。
4. 在 epoch 6 开始阶段停止原进程。
5. 使用原 tmux 名和原日志，从 `ckpt_19475_05.pt` 恢复。
6. 恢复后日志确认：`[Resume] Completed epochs: 5; running 15 remaining epoch(s) to reach 20.`

因此当前 a4all 最终可保留 epoch 4–20；epoch 1–3 已在修改前被删除。以后新启动的 a3all/a4all 从 epoch 1 起每轮保留，可完整得到 epoch 1–20。

## 4. 连续性验证

- 在线 W&B 日志明确显示：`Resuming run LAIN_hicodet_text_adapter_obj_cond_scene_v5`。
- 在线 run ID 和页面未改变：`k1rgp1f6`。
- `scene_gate_diagnostics.json` 保留原有 epoch 1–5 五行；恢复后的 epoch 6 会追加到同一个 JSON/CSV。
- `best_unseen.pt` 与 `best_unseen_checkpoint.json` 保留在原 checkpoint 目录。
- 训练日志数量仍为 1，恢复信息追加到原日志。
- 实际新进程命令包含：`--resume .../ckpt_19475_05.pt --keep-last-checkpoints 0`。
- `args.txt` 已原子更新为：
  - `keep_last_checkpoints=0`
  - `keep_epoch_checkpoints=[]`
  - `checkpoint_retention_start_epoch=4`
  - `wandb_run_id=k1rgp1f6`
- 使用临时 epoch 1–6 checkpoint 做保存轮转单元测试，六个文件全部保留，测试通过。

说明：W&B SDK 恢复同一个在线 run 时，会在本地创建新的内部 service segment 目录（目录名时间不同但 run ID 仍为 `k1rgp1f6`）。该二进制内部目录不能安全手工拼接；在线图表、训练日志、checkpoint 和诊断 JSON/CSV 均保持连续。

## 5. 服务器备份

- 第一批代码修改前备份：
  `/root/Lain/LAIN-main/server_code_backups/20260812_002412_keep_every_epoch_wandb_resume`
- W&B ID 标记逻辑修改前备份：
  `/root/Lain/LAIN-main/server_code_backups/20260812_002449_wandb_id_marker`

