# G3：alpha=0.08 Gate-only 训练脚本修改记录

- 修改时间：2026-08-17（Asia/Shanghai）
- 服务器代码目录：`/root/Lain/LAIN-main`
- 数据输出目录：`/root/autodl-tmp/Lain/UO_a4lr_ep6_gateonly_g3_a008_seed66`

## 修改内容

1. 扩展 `scripts/training/UO_a4lr_ep6_gateonly_common.sh`，新增 `g3` 变体。
2. G3 设置 `scene_gate_alpha=0.08`。
3. 新增前台脚本 `UO_a4lr_ep6_gateonly_g3_a008.sh`。
4. 新增后台脚本 `UO_a4lr_ep6_gateonly_g3_a008_background.sh`。

## 保持不变的受控条件

- 初始化 checkpoint：A4all-lowrank 第6轮 `ckpt_23370_06.pt`
- Text Adapter：关闭
- Object Adapter：开启并冻结
- Scene Gate：加载时重置，从零训练
- Gate结构：V5-lowrank，rank=16
- 仅训练Scene Gate
- Gate学习率：`3e-4`
- 训练轮数：10
- 每轮保存checkpoint，同时保存最佳Unseen checkpoint
- 每轮执行Gate开关对比和Adapter贡献诊断，并上传W&B曲线

## 启动命令

```bash
bash /root/Lain/LAIN-main/scripts/training/UO_a4lr_ep6_gateonly_g3_a008_background.sh
```

本次仅添加脚本，未启动G3训练。
