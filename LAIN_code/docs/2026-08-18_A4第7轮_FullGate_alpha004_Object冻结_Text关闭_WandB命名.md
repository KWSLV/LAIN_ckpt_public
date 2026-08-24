# A4第7轮起点：Full Gate alpha=0.04训练脚本与W&B命名修改

- 修改时间：2026-08-18（Asia/Shanghai）
- 服务器代码目录：`/root/Lain/LAIN-main`
- 训练脚本名：`UO_a4full_ep7_gateonly_v5h128_a004.sh`

## 实验配置

- 初始化checkpoint：`/root/autodl-tmp/Lain/UO_a4all_seed66/checkpoints/ckpt_27265_07.pt`
- Text Adapter：关闭
- Object Adapter：开启，载入第7轮权重并冻结
- Scene Gate：跳过checkpoint中的旧Gate权重，重新初始化
- Gate结构：普通V5，`1024 -> 128 -> 1`
- `scene_gate_alpha=0.04`
- `lr_scene_gate=3e-4`
- 仅训练Scene Gate
- 默认训练10轮
- 每轮保存checkpoint，额外保存最佳Unseen checkpoint
- 每轮执行Gate关闭对比与Object贡献诊断
- 默认输出：`/root/autodl-tmp/Lain/UO_a4full_ep7_gateonly_v5h128_a004_seed66`

## W&B命名修改

`main.py`新增可选环境变量`WANDB_NAME`。新脚本将它设置为训练脚本的文件名主体：

```text
UO_a4full_ep7_gateonly_v5h128_a004
```

没有设置`WANDB_NAME`的旧脚本继续使用原有自动命名规则，不受影响。

## 后台启动命令

```bash
bash /root/Lain/LAIN-main/scripts/training/UO_a4full_ep7_gateonly_v5h128_a004_background.sh
```

本次只编写并部署脚本，未启动训练。
