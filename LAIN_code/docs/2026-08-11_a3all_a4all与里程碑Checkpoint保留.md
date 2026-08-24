# A4 Gate 检查与 a3all/a4all 训练脚本变更记录

- 变更时间：2026-08-11 01:11（Asia/Shanghai）
- 服务器代码目录：`/root/Lain/LAIN-main`
- 服务器变更前备份：`/root/Lain/LAIN-main/server_code_backups/20260811_011105_a3all_a4all`
- 本次未启动、未重启、未中断任何训练；原 A4 继续在 `lain_a4_text_object_gate_r4a16_s66` 中运行。

## 1. 当前 A4 Gate 状态

截至诊断 JSON 已完成的第 14 轮：

- 最好 UO Unseen mAP：36.22555（epoch 13）。
- Gate ON - Gate OFF 的 Unseen 贡献：14 轮中 12 轮为正，均值约 +0.02830 mAP；epoch 13 为 +0.05472。
- epoch 14：`u.std=0.61190`，`u_rel.std=0.60816`，`ratio.mean=0.01058`。
- `u_rel` 并未接近常数或全零，故没有发生数值意义上的 Gate 坍缩。
- 但 Gate 注入残差的平均范数只有 HO 特征范数的约 1.06%，Gate 的实际影响偏弱，不能据此宣称 Gate 已带来显著增益。

## 2. 当前 Gate 算法来源

当前使用 `models/scene_gate_v5.py` 中的 `SceneGatePairV5`，A4 参数为：

- `type=pair`，`version=v5`
- `hidden_dim=128`，`rank=16`，`dropout=0.1`
- `alpha=0.02`，`activation=standardized_tanh`
- `init_std=1e-3`，`start_epoch=1`

计算过程为：先对 HO/CLS 的副本做 LayerNorm，并使用 detach 后的输入预测每对 HOI 的原始门控值；再按同一图像的 pair 维度做中心化和标准化，经过 tanh 后再次减去 pair 均值，最后将 `0.02 * u_rel * detach(CLS)` 加到 HO 特征。

来源核对：

- 当前文件 SHA256：`5A3C4ACD7C5D021876518350E340350EF831F0EA8A202A5237FA941D5F324E23`
- `lain0712/Scenegateresult-master` 原始文件 SHA256：`DF524E7D88DEC39A78FADA16C5519EC7BE32C2C85DEAA18363FAB400A0B78EE3`
- `lain0712/server_training_records_20260727_155632/server_code_snapshot` SHA256：与当前文件相同。

结论：当前服务器不是使用 `Scenegateresult-master` 中未经修改的原始 V5，而是使用此前对话中改过的 pair-relative / standardized-tanh V5；该修改版也已被收录进 lain0712 的 2026-07-27 服务器快照。

## 3. lain0712 JSON 扫描结果

递归解析 `G:\vision\lain0712` 下 67 个 JSON，共找到 1208 个键名含 `unseen` 的数值：

- `unseen >= 38`：0 条。
- 全目录最高值是 37.36920，来自 `Scenegateresult-master/results/UO_v5_joint_diag/epoch_diagnostics.json` 的 Gate OFF 评估。
- 同一文件中最高 Gate ON 是 37.21041。
- 服务器归档的 Gate refine 最高 Gate ON 是 37.20192。

因此，lain0712 的场景门控文件夹/归档 JSON 中没有能够证明 Unseen 38 的结构化记录。

## 4. 本次代码修改

### Checkpoint 保留

- `utils/args.py` 新增 `--keep-epoch-checkpoints EPOCH ...`。
- `engine.py` 的轮转策略改为：始终保留指定里程碑，同时保留 `--keep-last-checkpoints` 指定的最近 N 轮。
- a3all/a4all 均使用：`--keep-last-checkpoints 1 --keep-epoch-checkpoints 5 10 15 20 --keep-best-unseen-checkpoint`。
- 训练中最近一轮用于断点续训；到 epoch 20 后，常规 checkpoint 恰好保留 5/10/15/20，另有独立 `best_unseen.pt` 和元数据。

### a3all（无 Gate）

- 前台：`scripts/training/UO_a3all.sh`
- 后台：`scripts/training/UO_a3all_background.sh`
- 输出：`/root/autodl-tmp/Lain/UO_a3all_seed66`
- 模块：原 LAIN Visual/HO token/Prompt/Instance Adapter/Prior 全部保留；Text Adapter r4/a16/dim64 + Object Adapter r4；Gate OFF。
- `adapter_pos=all`，训练 20 轮。
- 学习率：head `5e-4`，ViT/Text/Object `2e-4`，drop epoch 10。
- 开启 Text/Object 条件消融贡献诊断。

### a4all（有 Gate）

- 前台：`scripts/training/UO_a4all.sh`
- 后台：`scripts/training/UO_a4all_background.sh`
- 输出：`/root/autodl-tmp/Lain/UO_a4all_seed66`
- a3all 的全部模块和参数不变，再加入上述 V5 SceneGate。
- Gate 学习率 `1e-3`，训练 20 轮。
- 开启 Gate ON/OFF、Text OFF、Object OFF 贡献诊断；结束后生成 `module_contributions.png`。

## 5. 验证

- 服务器 `python -m py_compile`：通过。
- 五个 Bash 文件 `bash -n`：通过。
- `main.py --help` 已显示 `--keep-epoch-checkpoints`：通过。
- 使用 1 到 20 轮模拟 checkpoint 进行轮转实测：最终仅保留 epoch 5/10/15/20，测试通过。

