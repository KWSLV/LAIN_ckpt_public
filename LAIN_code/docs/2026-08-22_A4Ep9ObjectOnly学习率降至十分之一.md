# A4 Epoch 9 Object-only 学习率降至十分之一

- 修改时间：2026-08-22（Asia/Shanghai）
- 对照实验：`UO_a4_ep9_objreset_objectonly_r4`
- 新实验：`UO_a4_ep9_objreset_objectonly_0.1lr`

## 修改内容

复制原训练及后台脚本，保持来源 checkpoint、模块开关、冻结策略、Object rank、
训练轮数、数据参数、优化参数、诊断参数和 checkpoint 策略不变。

唯一训练超参数变化：

- `lr_obj_cond_adapter`: `2e-4` → `2e-5`

实验名、输出目录、W&B 名称和 tmux 名称随新脚本名称变化，用于隔离实验结果。
tmux session 名使用等价的 `0_1lr`（tmux 将点号作为目标分隔符），不影响实验名、
输出目录或 W&B 名称中的 `0.1lr`。
