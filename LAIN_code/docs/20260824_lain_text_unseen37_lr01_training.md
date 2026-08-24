# LAIN + Text：Unseen > 37 后全 LR ×0.1

- 修改时间：2026-08-24 11:45:48 +08:00
- 训练脚本：`scripts/training/UO_refineep04_lain_text_joint_r4a16_unseen37_lr01.sh`
- 输出目录：`/root/autodl-tmp/Lain/UO_refineep04_lain_text_joint_r4a16_unseen37_lr01_seed66`
- W&B 名称：`UO_refineep04_lain_text_joint_r4a16_unseen37_lr01`

## 目标与初始化

- 从 `/root/autodl-tmp/Lain/UO_objectbest_gateonly_v5h128_a004_refine_from_ep01_lr5e5_seed66/checkpoints/ckpt_15580_04.pt` 初始化 LAIN、Prompt 和 Visual Adapter。
- Text Adapter 使用 `--reset_text_adapter_on_load` 重新初始化。
- Object Adapter 与 Scene Gate 不构建、不参与 forward、没有 optimizer 参数组。
- 不使用 `--train_text_adapter_only`，因此训练 LAIN 原本可训练部分、Prompt、Visual Adapter 和 Text Adapter。

## 训练参数

- 20 epochs，seed 66，batch size 8。
- LAIN：`lr_head=1e-3`、`lr_vit=1e-3`。
- Text：rank 4、alpha 16、dim 64、residual scale 0.05、dropout 0.1、LR `5e-4`。
- `CSC=True`、`N_CTX=36`。
- Visual Adapter：`adapt_dim=32`、`adapter_pos=all`、`adapter_alpha=1.0`、1 layer。
- `alpha=0.5`、`gamma=0.2`、`hyper_lambda=2.8`。
- `box_score_thresh=0.2`、`min_instances=3`、`max_instances=15`。
- AMP 开启，真实梯度在裁剪前执行 unscale，`clip_max_norm=0.1` 有效。

## Unseen 阈值 LR 策略

- 新增通用参数：
  - `--unseen-lr-threshold-schedule`
  - `--unseen-lr-threshold 37.0`
  - `--unseen-lr-threshold-factor 0.1`
- 当某轮评估结果首次满足 `Unseen > 37.0` 时，在保存该轮 checkpoint 前，将全部有效 optimizer 参数组设置为各自初始 LR 的 0.1 倍：
  - head：`1e-3 → 1e-4`
  - vit：`1e-3 → 1e-4`
  - text：`5e-4 → 5e-5`
- 只触发一次，后续保持降低后的 LR；阈值轮已经训练完成，因此低 LR 从下一轮训练开始生效。
- 固定 StepLR 被明确关闭，保留的 `lr_drop=10` 仅用于与原参数记录对齐，不产生第二次衰减。
- 每轮将实际 LR、是否本轮触发和是否已经触发写入 W&B。
- 策略状态写入 `unseen_lr_threshold_state.json`；同目录 `--resume` 时恢复触发状态，避免再次 ×0.1。

## 保存和诊断

- 每轮保留 checkpoint，并持续维护 `best_unseen.pt`。
- 每轮评估 Text ON/OFF contribution，写入现有 diagnostics JSON/CSV 和 W&B。
- 本次只编写和部署脚本，不自动停止或启动训练，不影响当前正在运行的全模块联合实验。

## 部署验证

- 服务器部署及验证时间：2026-08-24 13:00:14 +08:00。
- 服务器端 `main.py`、`engine.py`、`utils/args.py` 通过 `py_compile`，训练脚本通过 `bash -n`。
- 使用三个真实 AdamW 参数组验证：Unseen 等于 37.00 时不触发；37.01 时仅触发一次并得到 `head=1e-4`、`vit=1e-4`、`text_adapter=5e-5`；后续 Unseen 38.00 不再次衰减。
- 部署前后原全模块联合实验进程保持运行；新脚本没有启动。

## 正式启动

- 用户后续指示启动；正式启动时间：2026-08-24 13:32:27 +08:00。
- 启动时上一项全模块联合实验已经结束，GPU 空闲，因此没有停止或覆盖其他训练。
- tmux：`uo_refineep04_lain_text_u37`。
- 日志：`/root/autodl-tmp/Lain/UO_refineep04_lain_text_joint_r4a16_unseen37_lr01_seed66/logs/train_20260824_133227.log`。
- W&B run ID：`uqhv5zzm`；API 核验名称为 `UO_refineep04_lain_text_joint_r4a16_unseen37_lr01`、状态为 `running`。
- 启动日志确认 `use_obj_cond_adapter=False`、`use_scene_gate=False`、`use_text_adapter=True`、`disable_lr_scheduler=True`。
- Text 的6个参数按预期重新初始化；Object/Gate checkpoint 键因模块未构建而被忽略，没有进入可训练列表。
- 实际可训练参数量为 5,027,745（2.94%）；GPU 多次采样达到 91%～97%，确认已进入训练计算。
