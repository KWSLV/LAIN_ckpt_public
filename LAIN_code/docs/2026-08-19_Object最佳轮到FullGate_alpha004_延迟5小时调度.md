# Object最佳轮到Full Gate alpha=0.04：延迟调度记录

- 修改时间：2026-08-19（Asia/Shanghai）
- 服务器代码目录：`/root/Lain/LAIN-main`
- Object来源目录：`/root/autodl-tmp/Lain/UO_a4_ep9_objreset_objectonly_r4_seed66/checkpoints`
- Gate输出目录：`/root/autodl-tmp/Lain/UO_objectbest_gateonly_v5h128_a004_seed66`

## 新增内容

1. `UO_objectbest_gateonly_v5h128_a004.sh`
   - 根据调度器传入的Object-only最佳checkpoint初始化。
   - 冻结原始LAIN和Object Adapter。
   - 关闭Text Adapter。
   - 跳过旧Gate权重，重新初始化普通V5 H128 Gate。
   - 仅训练Gate，`scene_gate_alpha=0.04`、`lr_scene_gate=3e-4`。
   - 默认训练6轮、第4轮降低学习率。
   - 每轮保存checkpoint并执行Gate/Object贡献诊断。
   - W&B名称与训练脚本一致：`UO_objectbest_gateonly_v5h128_a004`。

2. `UO_objectbest_gateonly_v5h128_a004_wait5h.sh`
   - 启动后先等待18000秒（5小时）。
   - 此后每600秒（10分钟）检查GPU锁、`main.py/torchrun`进程和GPU计算进程。
   - 服务器繁忙时只等待，不抢占GPU，也不影响当前Object-only训练。
   - 服务器空闲后读取`scene_gate_diagnostics.json`，以`on.unseen`选择最高轮。
   - 验证对应epoch checkpoint唯一、存在、路径位于指定目录且文件大小/时间稳定。
   - 记录选中的路径、epoch和Unseen mAP，然后启动Gate-only训练。

3. `UO_objectbest_gateonly_v5h128_a004_wait5h_background.sh`
   - 在独立tmux中启动上述等待与训练流程。

## 调度会话

```text
lain_wait5h_objectbest_gate_a004_s66
```

## 手动查看

```bash
tmux attach -t lain_wait5h_objectbest_gate_a004_s66
```

本次已启动延迟调度器；启动调度器时Object-only训练仍在运行。

## 实际启动状态

- 调度器启动：2026-08-19 01:47:20（服务器时间）
- 第一次空闲检查：预计2026-08-19 06:47:20
- 启动时Object-only已完成6轮；当时JSON最高为第6轮，Unseen mAP为37.599244，但最终会在服务器空闲时重新读取完整JSON并选择，不会固定使用启动时结果。
