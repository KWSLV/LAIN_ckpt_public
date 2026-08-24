# GitHub代码与Release归档上传记录

- 操作日期：2026-08-25
- 目标仓库：`KWSLV/LAIN_ckpt_public`
- 代码目标目录：`LAIN_code/`
- 服务器源代码：`/root/Lain/LAIN-main`
- 训练输出根目录：`/root/autodl-tmp/Lain`

## 用户要求

- 上传服务器训练代码，但排除冻结模型、checkpoint和原始数据集。
- 包含本地与服务器的实验文档记录。
- 训练脚本和测试/诊断脚本分别集中到独立目录。
- 尚未存在于GitHub Release的checkpoint按实验归档。
- 每组实验的训练记录分别放到对应Release。
- 上传完成后不删除服务器源checkpoint和训练记录。
- 稳定开始上传后按约1小时频率检查，不高频轮询。

## 代码归档结构

- `LAIN_code/`：服务器代码快照。
- `LAIN_code/training_scripts/`：训练及后台启动脚本集合。
- `LAIN_code/testing_scripts/`：评估、checkpoint测试、诊断、合并和绘图脚本集合。
- `LAIN_code/docs/`：服务器文档与 `G:\vision\LAIN-main\docs` 的合并结果。

明确排除：

- `*.pt`、`*.pth`、`*.bin`及checkpoint目录。
- 服务器 `hicodet/` 原始数据目录及外部HICO图像数据。
- `.git`、Python缓存和编辑器元数据。

## Release上传机制

- 上传器：`upload_remaining_lain_assets_20260825.py`。
- 扫描 `/root/autodl-tmp/Lain` 下各独立实验目录。
- 获取全部现有Release资产的SHA-256 digest，内容已存在时跳过，防止重复上传。
- 每个实验使用自身名称作为Release tag；对已有旧tag的两个实验保留旧tag映射。
- checkpoint保持原名；嵌套checkpoint使用相对路径展平后的可追溯名称。
- 每个实验创建确定性的 `<实验名>__training_records.tar.gz`，包含非`.pt`训练日志、JSON、CSV、图、参数及W&B记录，排除`.git`和Git LFS缓存。
- 只删除上传过程中生成的临时压缩包；绝不删除原始服务器文件。
- 状态文件：`/root/lain_public_release_upload_20260825_status.json`。
- 清单文件：`/root/lain_public_release_upload_20260825_manifest.json`。
