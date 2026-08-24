# Object + Gate 联合 Best 回滚训练

- 修改时间：2026-08-20 13:46:44 +08:00
- 本次只编写、部署并校验脚本，不启动训练。

## 训练锚点

- 来源 checkpoint：`/root/autodl-tmp/Lain/UO_objectbest_gateonly_v5h128_a004_refine_from_ep01_lr5e5_seed66/checkpoints/ckpt_15580_04.pt`
- 来源轮次：原 refine 训练第 4 轮。
- 来源 Unseen mAP：`37.775465258102386`。
- 新输出目录：`/root/autodl-tmp/Lain/UO_objectgate_joint_bestrollback_from_gateep04_seed66/checkpoints`
- 启用模块：ObjectConditionedAdapter、SceneGate V5 H128 alpha=0.04。
- 冻结模块：DETR、原始 LAIN、CLIP/Prompt、Text Adapter。
- 初始学习率：Object `5e-6`，Gate `1e-5`。

## 新增控制逻辑

新增 `--train-object-and-scene-gate-only`，只允许 Object Adapter 和 SceneGate 参数进入优化器。两个参数组分别命名为 `obj_cond_adapter`、`scene_gate`，便于独立记录和同步调整实际学习率。

新增 `--joint-best-rollback-policy`：

1. 启动时先把来源第 4 轮复制/硬链接为输出目录的 `best_unseen.pt`，并以 `37.775465258102386` 作为训练开始前的历史 best。
2. 每轮完成 Full、Gate-OFF、Object-OFF 评估，得到 Full/Unseen/Seen、Gate contribution 和 Object contribution。
3. 当前 Unseen 创造新 best 且提升不少于 `0.03`：更新 `best_unseen.pt`，下一轮保持当前 LR。
4. 当前 Unseen 创造新 best但提升小于 `0.03`：更新 best，Object/Gate LR 均乘 `0.5`。
5. 当前 Unseen 未创新 best，但下降不超过 `0.03`：不回滚，Object/Gate LR 均乘 `0.5`，并累计一次未创新 best。
6. 当前 Unseen 比历史 best 下降超过 `0.03`：立即重新载入 `best_unseen.pt`；清除 AdamW 的全部一阶/二阶动量；重建 StepLR；Object/Gate LR 均乘 `0.2`。当轮的坏权重不会用于下一轮。
7. 连续 2 轮不能创造新 best：当轮诊断与 checkpoint 写完后提前停止。

说明：发生 rollback 的轮次仍会保存对应 epoch checkpoint，但其中模型权重已经是回滚后的历史 best 权重，评估指标则保留回滚前坏更新的真实结果，以便追溯。

## 记录内容

`scene_gate_diagnostics.json/.csv` 每轮新增 `joint_training_policy` 字段，记录：

- `best_unseen_before`、`current_unseen`、`delta_vs_best`
- `new_best`、`rollback`、`optimizer_reset`、`action`
- Object/Gate 当轮实际 LR 与下一轮 LR
- 连续未创新 best 次数及是否 early stop

原有诊断继续记录 Full/Unseen/Seen、Gate contribution、Object contribution。W&B 同步记录这些指标、实际 LR、rollback、新 best 和连续失败计数；W&B 名称与训练脚本名一致。

## 文件

- `scripts/training/UO_objectgate_joint_bestrollback_from_gateep04.sh`
- `scripts/training/UO_objectgate_joint_bestrollback_from_gateep04_background.sh`
- `main.py`
- `engine.py`
- `utils/args.py`

