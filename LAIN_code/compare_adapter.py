import os
import subprocess

DATA_ROOT = "./hicodet"
PRETRAINED = "checkpoints/pretrained_detr/detr-r50-hicodet.pth"
EPOCHS = 5
BATCH_SIZE = 4
LR_HEAD = "1e-4"

# 实验定义 (参数名全部使用中划线 -)
experiments = [
    {
        "name": "Baseline_No_Adapter",
        "args": "--use_prompt",  # 必须带上，保证文本特征生成
        "output_dir": "checkpoints/baseline"
    },
    {
        "name": "With_Semantic_Adapter",
        "args": "--use_prompt --use_semantic_adapter --adapt_dim 64",
        "output_dir": "checkpoints/adapter_exp"
    }
]


def run_cmd(cmd):
    # 清理多余空格
    cmd = " ".join(cmd.split())
    print(f"\n🚀 Executing: {cmd}\n")
    process = subprocess.Popen(cmd, shell=True)
    process.wait()


def main():
    for exp in experiments:
        print(f"\n" + "=" * 40)
        print(f"统计开始: {exp['name']}")
        print("=" * 40)

        # 确保输出目录存在
        os.makedirs(exp['output_dir'], exist_ok=True)

        # 1. 训练命令 (修正参数名为中划线)
        train_cmd = f"""WANDB_MODE=disabled MASTER_ADDR=localhost MASTER_PORT=12345 \
    python main.py \
    --dataset hicodet \
    --batch-size {BATCH_SIZE} \
    --epochs {EPOCHS} \
    --num-workers 0 \
    --lr-head {LR_HEAD} \
    --backbone resnet50 \
    --data-root {DATA_ROOT} \
    --pretrained {PRETRAINED} \
    --output-dir {exp['output_dir']} \
    --use_prompt \
    --zs --zs_type rare_first {exp['args']}"""
        run_cmd(train_cmd)

        # 2. 测试命令
        # 注意：resume 参数通常是下划线，或者是根据 usage 提示
        eval_cmd = f"""python main.py --eval \
            --dataset hicodet \
            --data-root {DATA_ROOT} \
            --resume {exp['output_dir']}/checkpoint.pth \
            --zs --zs_type rare_first {exp['args']}"""

        print(f"\n📊 Evaluating {exp['name']}...")
        run_cmd(eval_cmd)

    print("\n✅ 所有实验已完成！")


if __name__ == "__main__":
    main()