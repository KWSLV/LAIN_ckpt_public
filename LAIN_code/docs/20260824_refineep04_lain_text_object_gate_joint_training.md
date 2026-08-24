# Refine ep4 全模块联合训练脚本记录

- 修改时间：2026-08-24 01:41:56 +08:00
- 脚本：`scripts/training/UO_refineep04_lain_text_object_gate_joint_r4a16.sh`
- 输出目录：`/root/autodl-tmp/Lain/UO_refineep04_lain_text_object_gate_joint_r4a16_seed66`
- W&B 名称：`UO_refineep04_lain_text_object_gate_joint_r4a16`

## 训练目的

从已经完成 Object + Gate 分阶段训练的 refine epoch 4 权重出发，同时解冻并联合训练原始 LAIN、Prompt、Visual Adapter、Text Semantic Adapter、Object Adapter 和 Scene Gate，验证 Text 在非冻结的联合优化环境中能否重新与视觉/交互特征对齐。

## 初始化

- checkpoint：`/root/autodl-tmp/Lain/UO_objectbest_gateonly_v5h128_a004_refine_from_ep01_lr5e5_seed66/checkpoints/ckpt_15580_04.pt`
- checkpoint 已验证可读取：`epoch=4`、`iteration=15580`、共 1520 个模型状态键。
- 继承：原始 LAIN、Prompt、Visual Adapter、Object Adapter、Scene Gate。
- Text：源 checkpoint 中没有 Text Adapter 权重，使用 `--reset_text_adapter_on_load` 明确重新初始化。
- 未使用 `--train_text_adapter_only`、`--train_obj_cond_adapter_only` 或 `--train_scene_gate_only`，因此上述训练模块参与联合优化；检测器和框架原本冻结的 CLIP 主干仍遵循 LAIN 默认冻结策略。

## 关键参数

### 原始 LAIN、Prompt 与 Visual Adapter（对应用户给出的 38.12 配置）

- 20 epochs，seed 66，batch size 8。
- `lr_head=1e-3`、`lr_vit=1e-3`、`lr_drop=10`、`weight_decay=1e-4`。
- `alpha=0.5`、`gamma=0.2`、`hyper_lambda=2.8`。
- `box_score_thresh=0.2`、`min_instances=3`、`max_instances=15`。
- Prompt：`CSC=True`、`N_CTX=36`。
- Visual Adapter：`adapt_dim=32`、`adapter_pos=all`、`adapter_alpha=1.0`、`adapter_num_layers=1`。

### Text Semantic Adapter

- `text_adapter_dim=64`、`lora_rank=4`、`lora_alpha=16`。
- residual scale 0.05、dropout 0.1、LR `5e-4`。

### Object Adapter

- 参数来源：`UO_a4_ep9_objreset_objectonly_r4`。
- rank 4、LR `2e-4`；继承 refine ep4 中已经训练的 Object 权重，不重置、不冻结。

### Scene Gate

- 参数来源：`UO_objectbest_gateonly_v5h128_a004`。
- 普通 V5 pair Gate，hidden 128、dropout 0.1、alpha 0.04。
- `standardized_tanh`、init std `1e-3`、post-gate L2 normalization。
- LR `3e-4`；继承 refine ep4 Gate 权重，不重置、不冻结。
- `scene_gate_rank=16` 仅保留源实验配置字段；普通 V5 H128 Gate 不使用低秩结构。

## 保存与诊断

- 每轮保存 checkpoint（`--keep-last-checkpoints 0`），并保存历史最佳 Unseen checkpoint。
- 每轮运行 Text/Object/Gate contribution diagnostics，并测试 Gate off 对照。
- 启用 AMP；当前训练引擎会先 unscale 再执行 `clip_max_norm=0.1`。
- W&B run 名与脚本名一致。

## 启动与实跑核验

- 启动时间：2026-08-24 01:44:15 +08:00。
- tmux：`uo_refineep04_joint_r4a16`。
- 训练日志：`/root/autodl-tmp/Lain/UO_refineep04_lain_text_object_gate_joint_r4a16_seed66/logs/train_20260824_014415.log`。
- W&B run ID：`tacg7zxb`；名称已通过 W&B API 核验为 `UO_refineep04_lain_text_object_gate_joint_r4a16`，启动后状态为 `running`。
- 权重加载仅报告预期的 6 个 Text Adapter 参数缺失并重新初始化；Object 和 Gate checkpoint 参数成功继承。
- 日志确认 `train_text_adapter_only=False`、`train_obj_cond_adapter_only=False`、`train_scene_gate_only=False`。
- 实际可训练参数量为 5,169,358（3.02%），可训练列表包含 Prompt、Visual Adapter、Text Adapter、Object Adapter 和 Scene Gate。
- 启动后 GPU 采样达到 79%～97% 计算利用率，确认不是只有空进程或 W&B 进程。
