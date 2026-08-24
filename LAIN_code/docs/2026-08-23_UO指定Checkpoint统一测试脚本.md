# UO 指定 checkpoint 统一测试脚本记录

- 时间：2026-08-23（Asia/Shanghai）
- 目的：同口径测试 Object 继承训练 epoch 1、epoch 5，以及 ObjectBest + Gate epoch 4。

## 原测试脚本的问题

原 `scripts/eval/UO.sh` 固定开启 Text、Object 和 Scene Gate，并使用 `adapter_pos=last`、Text r8 等旧配置，不能直接测试本次 checkpoint：

- `UO_a4_ep9_objinherit_objectonly_r4` 实际为 Text OFF、Gate OFF、Object r4、`adapter_pos=all`。
- `UO_objectbest_gateonly_v5h128_a004` 实际为 Text OFF、Object r4、Gate V5 H128、alpha 0.04、`standardized_tanh`、post-gate L2 ON、`adapter_pos=all`。

如果直接使用旧脚本，会构建错误的模型配置，出现未训练模块参与 forward 或 checkpoint 键不匹配，测试结果不可用于比较。

## 新增脚本

- `scripts/eval/UO_checkpoint_module_eval.sh`
  - 支持 `object_only` 和 `object_gate` 两种模式。
  - Object-only：测试完整 Object 输出及 Object OFF 消融。
  - Object+Gate：测试全开、Gate OFF 和 Object OFF，计算对应贡献度。
  - Text 始终关闭。
- `scripts/eval/UO_test_requested_checkpoints_20260823.sh`
  - 顺序测试三个指定 checkpoint。
  - 每个 checkpoint 使用独立结果目录，保存 `eval.log`、`scene_gate_diagnostics.json` 和 CSV。

## 本批测试 checkpoint

1. `UO_a4_ep9_objinherit_objectonly_r4_seed66/checkpoints/ckpt_03895_01.pt`
2. `UO_a4_ep9_objinherit_objectonly_r4_seed66/checkpoints/ckpt_19475_05.pt`
3. `UO_objectbest_gateonly_v5h128_a004_seed66/checkpoints/ckpt_15580_04.pt`

## 独立测试结果

结果目录：`/root/autodl-tmp/Lain/UO_checkpoint_module_eval_20260823_194959`

| checkpoint | Full | Unseen | Seen | Object OFF Unseen | Object contribution | Gate OFF Unseen | Gate contribution |
|---|---:|---:|---:|---:|---:|---:|---:|
| Object 继承 epoch 1 | 34.0991 | 37.5769 | 33.4035 | 37.1554 | +0.4214 | — | — |
| Object 继承 epoch 5 | 34.1734 | 37.5866 | 33.4907 | 37.1559 | +0.4307 | — | — |
| ObjectBest + Gate epoch 4 | 34.1925 | 37.5351 | 33.5239 | 37.1187 | +0.4164 | 37.6344 | -0.0992 |

## 直接结论

- Object 继承训练从 epoch 1 到 epoch 5，Unseen 只增加 `0.0098`，Object 的 Unseen 贡献只增加 `0.0093`，已经基本进入平台期。
- 三个 checkpoint 中，独立测试最高 Unseen 是 Object-only epoch 5 的 `37.5866`。
- Gate epoch 4 在本次独立测试中对 Unseen 是负贡献：全开 `37.5351`，关闭 Gate 后 `37.6344`，差值 `-0.0992`。
- Gate epoch 4 的训练期记录同样是负贡献（约 `-0.0279`）；本次负贡献幅度更大，但方向一致，因此该 checkpoint 不能证明 Gate 对 UO 有效。
