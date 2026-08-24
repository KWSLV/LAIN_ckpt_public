# A4ALL 逐 Checkpoint 关闭 SceneGate 复测脚本记录

- 修改时间：2026-08-13 16:54（Asia/Shanghai）
- 服务器代码目录：`/root/Lain/LAIN-main`
- 目标训练目录：`/root/autodl-tmp/Lain/UO_a4all_seed66`
- 目标诊断文件：`checkpoints/scene_gate_diagnostics.json`
- 后台 tmux：`lain_a4all_gate_off_eval_s66`
- 批处理日志：`gate_off_checkpoint_eval/logs/batch_20260813_170427.log`

## 1. 当前已有数据说明

当前 A4ALL 训练本身开启了 `--scene-gate-compare-off`，所以每轮 JSON 原本就包含：

- `on`：完全体 Gate ON 的 full/unseen/seen mAP；
- `off`：同轮内临时关闭 Gate 的 full/unseen/seen mAP；
- `gap`：`on - off`，即场景门控条件贡献度。

本次按用户要求增加独立的逐 checkpoint 复测流程。该流程重新加载每个磁盘 checkpoint，强制关闭 SceneGate，只运行一次测试，并把带 checkpoint 来源和测试时间的结果合并回原 JSON。

## 2. 新增入口与脚本

### `main.py` / `utils/args.py`

新增：

- `--eval-scene-gate-off-only`
- `--eval-result-json PATH`

该模式加载包含 SceneGate 权重的 checkpoint，然后把 `use_scene_gate` 临时设为 `False`，只执行一次 HICO-DET 测试。它不会再执行 Gate ON、Text OFF、Object OFF 等额外评估，也不会直接改训练目录中的共享 JSON，而是输出一个机器可读的临时结果文件。

### `scripts/eval/UO_a4all_gate_off_checkpoints.sh`

- 等待 `/tmp/lain_cuda_0.lock`，不会和当前 A4ALL 训练抢 GPU。
- 获取锁后重新扫描 `checkpoints/ckpt_*.pt`，因此会包含训练结束后的 epoch 20。
- 按 checkpoint 文件名解析真实 epoch，逐个执行 Gate-OFF-only 测试。
- 启动前备份原 `scene_gate_diagnostics.json`。
- 每轮结果保存到 `gate_off_checkpoint_eval/results/epoch_XX_gate_off.json`。
- 每轮独立日志保存到 `gate_off_checkpoint_eval/logs/epoch_XX.log`。
- 默认跳过 JSON 中已有完整 `gate_ablation` 且 checkpoint 名一致的轮次；`FORCE=1` 可强制重测。

### `scripts/diagnostics/merge_gate_off_checkpoint_eval.py`

- 检查结果确实来自指定 checkpoint 且 `scene_gate_enabled_during_evaluation=false`。
- 用同轮 JSON 的 `on` 计算：`contribution = gate_on - gate_off`。
- 使用临时文件、`fsync` 和 `os.replace` 原子更新共享 JSON。
- 外层 Bash 使用 `.scene_gate_diagnostics.lock` 文件锁，避免两个复测进程同时写入。
- 更新兼容旧分析代码的顶层 `off` 和 `gap`。

每轮新增结构：

```json
"gate_ablation": {
  "definition": "scene_gate_contribution = full_model_gate_on - same_checkpoint_gate_off",
  "evaluation_mode": "posthoc_checkpoint_scene_gate_off_only",
  "checkpoint": "ckpt_XXXXX_XX.pt",
  "checkpoint_epoch": 0,
  "checkpoint_iteration": 0,
  "evaluated_at": "UTC ISO timestamp",
  "gate_on": {"full": 0, "unseen": 0, "seen": 0},
  "gate_off": {"full": 0, "unseen": 0, "seen": 0},
  "contribution": {"full": 0, "unseen": 0, "seen": 0},
  "training_epoch_gate_off_before_posthoc": {},
  "posthoc_minus_training_epoch_gate_off": {},
  "scene_gate_enabled_during_evaluation": false
}
```

## 3. 当前执行状态

脚本已通过服务器：

- `python -m py_compile`
- `bash -n`
- `main.py --help` 参数检查
- 使用第 19 轮诊断数据的隔离副本完成合并测试，正确计算 Unseen 贡献度 `+0.118471`，且没有改写源 JSON。
- 断点跳过检查同时兼容顶层对象（`{"rows": [...]}`）与旧式顶层数组两种诊断 JSON 结构。

后台任务已启动，但 A4ALL 正在训练 epoch 20，因此当前正确阻塞在：

```text
[Gate-OFF Batch] Waiting for CUDA device 0...
```

训练释放 GPU 后批处理会自动开始。现有 epoch 4–19，加上即将产生的 epoch 20，共预计 17 个 checkpoint；每个仅一次测试，按当前测试速度预计总耗时约 3 小时。

## 4. 服务器备份

修改前服务器文件备份：

`/root/Lain/LAIN-main/server_code_backups/20260813_165348_a4all_gate_off_ckpt_eval`

## 5. 完成结果与折线图

- 完成时间：2026-08-13 22:58（Asia/Shanghai）
- 已独立复测并合并 epoch 4–20，共 17 个 checkpoint。
- 合并后的服务器文件：`/root/autodl-tmp/Lain/UO_a4all_seed66/checkpoints/scene_gate_diagnostics.json`
- 本地绘图脚本：`scripts/diagnostics/plot_scene_gate_ablation.py`
- 本地折线图：`docs/figures/a4all_gate_off/a4all_scene_gate_on_off_comparison.png`
- 本地明细表：`docs/figures/a4all_gate_off/a4all_scene_gate_on_off_data.csv`
- 本地 JSON 副本：`docs/figures/a4all_gate_off/scene_gate_diagnostics.json`

17 轮的平均 Unseen SceneGate 贡献为 `+0.002407 mAP`；8 轮为正、9 轮为负。最高 Gate ON Unseen 为 epoch 19 的 `37.300963`，同 checkpoint 关闭 Gate 后为 `37.148238`，贡献 `+0.152724`。

## 6. 三模块单独关闭的 Unseen 对比图

- 绘图时间：2026-08-13（Asia/Shanghai）
- 曲线：All ON、Text OFF、Object OFF、SceneGate OFF（post-hoc）。
- 数据范围：具有完整三种消融数据的 epoch 4–20，共 17 轮。
- 绘图脚本：`scripts/diagnostics/plot_all_module_ablation_unseen.py`
- 折线图：`docs/figures/a4all_gate_off/a4all_all_modules_off_unseen_comparison.png`
- 明细 CSV：`docs/figures/a4all_gate_off/a4all_all_modules_off_unseen_data.csv`

各曲线最高 Unseen：All ON `37.300963`（epoch 19）；Text OFF `37.330080`（epoch 18）；Object OFF `37.169815`（epoch 9）；SceneGate OFF `37.264392`（epoch 9）。这里的 OFF 均是相同 checkpoint 上只关闭一个模块，其余模块保持开启。
