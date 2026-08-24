# Object重置与Object-only冻结机制

- 修改时间：2026-08-18（Asia/Shanghai）
- 修改文件：`main.py`、`utils/args.py`

## 新增参数

### `--reset_obj_cond_adapter_on_load`

- 只能与`--init_from`配合使用，不能用于`--resume`。
- 加载checkpoint时跳过所有名称含`obj_cond_adapter`的张量。
- ObjectConditionedAdapter保留当前模型构造时的全新初始化。
- 要求同时开启`--use_obj_cond_adapter`。

### `--train_obj_cond_adapter_only`

- 冻结Detector、CLIP、Prompt、Instance Adapter、Text Adapter、Scene Gate、`query_proj`、`priors_downproj`等全部非Object参数。
- 唯一允许训练的参数名称必须包含`obj_cond_adapter`。
- 启动时执行运行期检查；如果没有Object参数或出现其他可训练参数，立即报错退出。
- 与`--train_scene_gate_only`互斥。

## 第9轮Object重置实验的必要用法

```text
--init_from /root/autodl-tmp/Lain/UO_a4all_seed66/checkpoints/ckpt_35055_09.pt
--use_obj_cond_adapter
--obj_cond_rank 4
--reset_obj_cond_adapter_on_load
--train_obj_cond_adapter_only
```

不传`--use_text_adapter`和`--use_scene_gate`，即可关闭Text与Scene Gate。建议保留`--adapter-contribution-diagnostics`，逐轮记录Object ON/OFF及贡献值。

## 兼容性

未使用上述新参数的既有训练和评估逻辑不变。
