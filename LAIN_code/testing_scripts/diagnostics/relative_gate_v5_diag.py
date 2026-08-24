import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_POCKET = ROOT / "pocket"
if str(LOCAL_POCKET) not in sys.path:
    sys.path.insert(0, str(LOCAL_POCKET))

import pocket
from datasets import DataFactory, custom_collate
from models.LAIN import build_detector
from pocket.utils import BoxPairAssociation, DetectionAPMeter
from utils.args import get_args as get_lain_args
from utils.hico_text_label import hico_unseen_index


def parse_args():
    parser = argparse.ArgumentParser(description="Raw SceneGate V5 pair/zero/shuffle/random diagnostics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--pretrained",
        default="/root/Lain/checkpoints/pretrained_detr/detr-r50-hicodet.pth",
    )
    parser.add_argument(
        "--clip-dir-vit",
        default="/root/Lain/checkpoints/pretrained_clip/ViT-B-16.pt",
    )
    parser.add_argument("--data-root", default="./hicodet")
    parser.add_argument(
        "--hico-image-root",
        default="/root/Lain/hico_20160224_det",
    )
    parser.add_argument("--output", default="checkpoints/UO_scene_text_obj/scene_gate_ablation.json")
    parser.add_argument("--test-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--modes", default="pair,zero,shuffle,random")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", default="29681")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--adapter-pos", default="last")
    parser.add_argument("--no-text-adapter", action="store_true")
    parser.add_argument("--no-obj-cond-adapter", action="store_true")
    return parser.parse_args()


def make_lain_args(cli):
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = get_lain_args()
    finally:
        sys.argv = old_argv

    args.pretrained = cli.pretrained
    args.clip_dir_vit = cli.clip_dir_vit
    args.data_root = cli.data_root
    args.hico_image_root = cli.hico_image_root
    args.dataset = "hicodet"
    args.partitions = ["train2015", "test2015"]
    args.num_classes = 117
    args.zs = True
    args.zs_type = "unseen_object"
    args.use_hotoken = True
    args.use_prompt = True
    args.use_exp = True
    args.CSC = True
    args.N_CTX = 36
    args.use_insadapter = True
    args.adapt_dim = 32
    args.adapter_alpha = 1.0
    args.adapter_pos = cli.adapter_pos
    args.use_prior = True
    args.use_text_adapter = not cli.no_text_adapter
    args.use_obj_cond_adapter = not cli.no_obj_cond_adapter
    args.use_scene_gate = True
    args.scene_gate_type = "pair"
    args.scene_gate_version = "v5"
    args.scene_gate_rank = 16
    args.scene_gate_dropout = 0.1
    args.device = cli.device
    args.local_rank = 0
    args.world_size = 1
    args.human_idx = 0
    args.clip_model_name = Path(args.clip_dir_vit).stem
    if args.clip_model_name == "ViT-B-16":
        args.clip_model_name = "ViT-B/16"
    elif args.clip_model_name == "ViT-L-14-336px":
        args.clip_model_name = "ViT-L/14@336px"
    return args


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_compatible(model, state_dict):
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    print(
        f"[LOAD] loaded={len(compatible)} missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)}",
        flush=True,
    )


def subset_num_gt(dataset):
    if len(dataset.keep) == len(dataset.dataset):
        return dataset.dataset.anno_interaction
    counts = [0] * 600
    for external_idx in dataset.keep:
        internal_idx = dataset.dataset._idx[external_idx]
        for hoi_id in dataset.dataset.annotations[internal_idx]["hoi"]:
            counts[int(hoi_id)] += 1
    return counts


@torch.no_grad()
def evaluate(model, dataloader, args, num_gt, mode, seed):
    model.eval()
    model.tp = None
    model.scene_gate.set_ablation_mode(mode, seed)
    model.scene_gate.enable_diagnostics(True)

    conversion = torch.from_numpy(np.asarray(model.object_n_verb_to_interaction, dtype=float))
    associate = BoxPairAssociation(min_iou=0.5)
    meter = DetectionAPMeter(600, nproc=1, num_gt=num_gt, algorithm="11P")

    for batch in dataloader:
        inputs = pocket.ops.relocate_to_cuda(batch[0])
        outputs = model(inputs, batch[1])
        if outputs is None:
            continue
        for output, target in zip(outputs, batch[-1]):
            output = pocket.ops.relocate_to_cpu(output, ignore=True)
            scores = output["scores"]
            interactions = conversion[output["objects"], output["labels"]]
            labels = torch.zeros_like(scores)
            if output["pairing"].numel() > 0:
                boxes_h, boxes_o = output["boxes"][output["pairing"]].unbind(0)
                gt_bx_h = model.recover_boxes(target["boxes_h"], target["size"])
                gt_bx_o = model.recover_boxes(target["boxes_o"], target["size"])
                for hoi_idx in interactions.unique():
                    gt_idx = torch.nonzero(target["hoi"] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4), gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4), boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1),
                        )
            meter.append(scores, interactions, labels)

    ap = meter.eval()
    unseen = set(hico_unseen_index[args.zs_type])
    metrics = {
        "full": float(ap.mean().item() * 100),
        "unseen": float(torch.as_tensor([v for i, v in enumerate(ap) if i in unseen]).mean().item() * 100),
        "seen": float(torch.as_tensor([v for i, v in enumerate(ap) if i not in unseen]).mean().item() * 100),
    }
    diagnostics = model.scene_gate.diagnostics()
    model.scene_gate.enable_diagnostics(False)
    return {"mode": mode, "seed": seed, "metrics": metrics, "diagnostics": diagnostics}


def main():
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", cli.port)
        backend = "nccl" if cli.device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    if cli.device.startswith("cuda"):
        torch.cuda.set_device(0)

    args = make_lain_args(cli)
    testset = DataFactory(
        args.dataset, args.partitions[1], args.data_root, args.clip_model_name, args=args
    )
    if cli.test_images > 0:
        testset.keep = testset.keep[:cli.test_images]
    num_gt = subset_num_gt(testset)
    loader = DataLoader(
        testset, collate_fn=custom_collate, batch_size=1,
        num_workers=cli.num_workers, pin_memory=True, shuffle=False,
    )

    model = build_detector(
        args,
        testset.dataset.object_class_to_target_class,
        object_n_verb_to_interaction=testset.dataset.object_n_verb_to_interaction,
        clip_model_path=args.clip_dir_vit,
    )
    checkpoint = safe_torch_load(cli.checkpoint)
    load_compatible(model, checkpoint["model_state_dict"])
    model.object_class_to_target_class = testset.dataset.object_class_to_target_class
    model.to(cli.device)

    modes = [mode.strip() for mode in cli.modes.split(",") if mode.strip()]
    seeds = [int(value) for value in cli.seeds.split(",") if value.strip()]
    jobs = []
    for mode in modes:
        if mode in {"shuffle", "random"}:
            jobs.extend((mode, seed) for seed in seeds)
        else:
            jobs.append((mode, 0))

    results = []
    for mode, seed in jobs:
        row = evaluate(model, loader, args, num_gt, mode, seed)
        results.append(row)
        metrics = row["metrics"]
        print(
            f"[DIAG] mode={mode} seed={seed} full={metrics['full']:.2f} "
            f"unseen={metrics['unseen']:.2f} seen={metrics['seen']:.2f}",
            flush=True,
        )

    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"checkpoint": cli.checkpoint, "results": results}, handle, indent=2)
    print(f"[INFO] saved diagnostics: {output_path}", flush=True)


if __name__ == "__main__":
    main()
