"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision

Modified for Semantic Adapter Integration
"""
import os
import json
import torch
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import sys
import argparse  # Added for argument patch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_POCKET_DIR = os.path.join(REPO_DIR, 'pocket')
if LOCAL_POCKET_DIR not in sys.path:
    sys.path.insert(0, LOCAL_POCKET_DIR)

from models.LAIN import build_detector
from utils.args import get_args

from engine import CustomisedDLE, EarlyStopTraining
from datasets import DataFactory, custom_collate
from utils.hico_text_label import hico_unseen_index

warnings.filterwarnings("ignore")


class FilterStream:
    def __init__(self, target):
        self.target = target

    def write(self, s):
        # 如果字符串里包含 tensor(0，就直接丢弃，不打印
        if "tensor(0" in s:
            return
        self.target.write(s)

    def flush(self):
        self.target.flush()


sys.stdout = FilterStream(sys.stdout)


def load_state_dict_compatible(
    model,
    state_dict,
    strict_adapter_shapes=False,
):
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state[key] = value

    msg = model.load_state_dict(compatible_state, strict=False)
    if skipped:
        print("[INFO] Skipped incompatible checkpoint tensors:")
        for key, ckpt_shape, model_shape in skipped:
            print(f"  {key}: checkpoint={ckpt_shape}, model={model_shape}")
    if msg.missing_keys:
        print(f"[WARN] Missing checkpoint keys ({len(msg.missing_keys)}):")
        for key in msg.missing_keys[:30]:
            print(f"  {key}")
        if len(msg.missing_keys) > 30:
            print("  ...")
    if msg.unexpected_keys:
        print(f"[WARN] Unexpected checkpoint keys ({len(msg.unexpected_keys)}):")
        for key in msg.unexpected_keys[:30]:
            print(f"  {key}")
        if len(msg.unexpected_keys) > 30:
            print("  ...")
    if strict_adapter_shapes:
        critical_tokens = (
            'adaptermlp',
            'text_adapter',
            'obj_cond_adapter',
            'scene_gate',
            'prompt_learner.ctx',
        )
        critical_skipped = [
            key
            for key, _, _ in skipped
            if any(token in key for token in critical_tokens)
        ]
        critical_missing = [
            key
            for key in msg.missing_keys
            if any(token in key for token in critical_tokens)
        ]
        critical_unexpected = [
            key
            for key in msg.unexpected_keys
            if any(token in key for token in critical_tokens)
        ]
        if critical_skipped or critical_missing or critical_unexpected:
            raise RuntimeError(
                'Evaluation checkpoint architecture does not match the '
                'requested Adapter/SceneGate configuration. '
                f'shape_mismatch={critical_skipped[:20]}, '
                f'missing={critical_missing[:20]}, '
                f'unexpected={critical_unexpected[:20]}'
            )
    return msg


def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(rank)
    if getattr(args, 'fast_cuda', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16'
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'

    if getattr(args, 'use_scene_gate', False) and not getattr(args, 'use_hotoken', False):
        raise ValueError('--use_scene_gate requires --use_hotoken because SceneGate fuses HO tokens with the CLS token.')
    if (
        getattr(args, 'use_scene_gate', False)
        and getattr(args, 'scene_gate_version', 'v5') in {'v5', 'v5_lowrank', 'v5_legacy'}
    ):
        args.scene_gate_type = 'pair'
    # Every SceneGate run must report both ON and OFF mAP at the end of each
    # epoch. Keep this enforced centrally so a training script cannot
    # accidentally omit the diagnostic flags.
    if getattr(args, 'use_scene_gate', False):
        args.scene_gate_compare_off = True
        args.scene_gate_diagnostics = True
    if getattr(args, 'scene_gate_compare_off', False) and not getattr(args, 'use_scene_gate', False):
        raise ValueError('--scene-gate-compare-off requires --use_scene_gate.')
    if getattr(args, 'scene_gate_compare_off', False):
        args.scene_gate_diagnostics = True
    if getattr(args, 'scene_gate_diagnostics', False) and not getattr(args, 'use_scene_gate', False):
        raise ValueError('--scene-gate-diagnostics requires --use_scene_gate.')
    if getattr(args, 'eval_scene_gate_off_only', False):
        if not args.eval:
            raise ValueError('--eval-scene-gate-off-only requires --eval.')
        if not getattr(args, 'use_scene_gate', False):
            raise ValueError('--eval-scene-gate-off-only requires --use_scene_gate.')
        if not getattr(args, 'resume', ''):
            raise ValueError('--eval-scene-gate-off-only requires --resume CKPT.')
        if not getattr(args, 'eval_result_json', ''):
            raise ValueError('--eval-scene-gate-off-only requires --eval-result-json PATH.')
    if (
        getattr(args, 'adapter_contribution_diagnostics', False)
        and not (
            getattr(args, 'use_text_adapter', False)
            or getattr(args, 'use_obj_cond_adapter', False)
        )
    ):
        raise ValueError(
            '--adapter-contribution-diagnostics requires --use_text_adapter '
            'or --use_obj_cond_adapter.'
        )
    if getattr(args, 'override_resume_scene_gate_lr', False):
        if not getattr(args, 'resume', ''):
            raise ValueError(
                '--override-resume-scene-gate-lr requires --resume.'
            )
        if not getattr(args, 'use_scene_gate', False):
            raise ValueError(
                '--override-resume-scene-gate-lr requires --use_scene_gate.'
            )
        if getattr(args, 'lr_scene_gate', 0.0) <= 0:
            raise ValueError('--lr-scene-gate must be positive.')
    if getattr(args, 'scene_gate_start_epoch', 1) < 1:
        raise ValueError('--scene_gate_start_epoch must be at least 1.')
    if getattr(args, 'scene_gate_start_epoch', 1) > 1 and not getattr(args, 'use_scene_gate', False):
        raise ValueError('--scene_gate_start_epoch greater than 1 requires --use_scene_gate.')
    if getattr(args, 'reset_scene_gate_on_load', False) and not getattr(args, 'init_from', ''):
        raise ValueError('--reset_scene_gate_on_load requires --init_from, not --resume.')
    if getattr(args, 'reset_obj_cond_adapter_on_load', False) and not getattr(args, 'init_from', ''):
        raise ValueError('--reset_obj_cond_adapter_on_load requires --init_from, not --resume.')
    if getattr(args, 'reset_obj_cond_adapter_on_load', False) and not getattr(args, 'use_obj_cond_adapter', False):
        raise ValueError('--reset_obj_cond_adapter_on_load requires --use_obj_cond_adapter.')
    if getattr(args, 'train_obj_cond_adapter_only', False) and not getattr(args, 'use_obj_cond_adapter', False):
        raise ValueError('--train_obj_cond_adapter_only requires --use_obj_cond_adapter.')
    if getattr(args, 'train_object_and_scene_gate_only', False):
        if not getattr(args, 'use_obj_cond_adapter', False):
            raise ValueError('--train-object-and-scene-gate-only requires --use_obj_cond_adapter.')
        if not getattr(args, 'use_scene_gate', False):
            raise ValueError('--train-object-and-scene-gate-only requires --use_scene_gate.')
    if getattr(args, 'reset_text_adapter_on_load', False) and not getattr(args, 'init_from', ''):
        raise ValueError('--reset_text_adapter_on_load requires --init_from, not --resume.')
    if getattr(args, 'reset_text_adapter_on_load', False) and not getattr(args, 'use_text_adapter', False):
        raise ValueError('--reset_text_adapter_on_load requires --use_text_adapter.')
    if getattr(args, 'train_text_adapter_only', False) and not getattr(args, 'use_text_adapter', False):
        raise ValueError('--train_text_adapter_only requires --use_text_adapter.')
    exclusive_modes = [
        getattr(args, 'train_text_adapter_only', False),
        getattr(args, 'train_obj_cond_adapter_only', False),
        getattr(args, 'train_scene_gate_only', False),
        getattr(args, 'train_object_and_scene_gate_only', False),
    ]
    if sum(bool(mode) for mode in exclusive_modes) > 1:
        raise ValueError(
            '--train_text_adapter_only, --train_obj_cond_adapter_only, '
            '--train_scene_gate_only and --train-object-and-scene-gate-only '
            'are mutually exclusive.'
        )
    if getattr(args, 'joint_best_rollback_policy', False):
        if not getattr(args, 'train_object_and_scene_gate_only', False):
            raise ValueError('--joint-best-rollback-policy requires --train-object-and-scene-gate-only.')
        if not getattr(args, 'keep_best_unseen_checkpoint', False):
            raise ValueError('--joint-best-rollback-policy requires --keep-best-unseen-checkpoint.')
        if not getattr(args, 'init_from', ''):
            raise ValueError('--joint-best-rollback-policy requires --init_from.')
        if args.joint_rollback_initial_best_unseen < 0:
            raise ValueError('--joint-rollback-initial-best-unseen must be non-negative.')
        if args.joint_rollback_significant_delta <= 0:
            raise ValueError('--joint-rollback-significant-delta must be positive.')
        if not 0 < args.joint_rollback_small_lr_factor <= 1:
            raise ValueError('--joint-rollback-small-lr-factor must be in (0, 1].')
        if not 0 < args.joint_rollback_bad_lr_factor < 1:
            raise ValueError('--joint-rollback-bad-lr-factor must be in (0, 1).')
        if args.joint_rollback_patience < 1:
            raise ValueError('--joint-rollback-patience must be at least 1.')
    if getattr(args, 'lain_object_staged_retry_policy', False):
        if not getattr(args, 'use_obj_cond_adapter', False):
            raise ValueError('--lain-object-staged-retry-policy requires --use_obj_cond_adapter.')
        if getattr(args, 'use_text_adapter', False):
            raise ValueError('--lain-object-staged-retry-policy requires Text Adapter to stay OFF.')
        if getattr(args, 'use_scene_gate', False):
            raise ValueError('--lain-object-staged-retry-policy requires SceneGate to stay OFF.')
        if not getattr(args, 'disable_lr_scheduler', False):
            raise ValueError('--lain-object-staged-retry-policy requires --disable-lr-scheduler.')
        if not getattr(args, 'adapter_contribution_diagnostics', False):
            raise ValueError('--lain-object-staged-retry-policy requires --adapter-contribution-diagnostics.')
        if not getattr(args, 'keep_best_unseen_checkpoint', False):
            raise ValueError('--lain-object-staged-retry-policy requires --keep-best-unseen-checkpoint.')
        if not args.zs:
            raise ValueError('--lain-object-staged-retry-policy requires zero-shot Unseen evaluation.')
        if getattr(args, 'init_from', ''):
            raise ValueError('--lain-object-staged-retry-policy does not accept --init_from.')
        if args.resume and not getattr(args, 'staged_resume_existing_run', False):
            raise ValueError('Staged --resume requires --staged-resume-existing-run.')
        if getattr(args, 'staged_resume_existing_run', False) and not args.resume:
            raise ValueError('--staged-resume-existing-run requires --resume.')
        if not 1 < args.staged_object_start_epoch <= args.epochs:
            raise ValueError('--staged-object-start-epoch must be in [2, epochs].')
        if args.staged_lr_trigger_unseen <= 0:
            raise ValueError('--staged-lr-trigger-unseen must be positive.')
        if args.staged_pre_trigger_max_drop <= 0 or args.staged_post_trigger_max_drop <= 0:
            raise ValueError('Staged retry decline thresholds must be positive.')
        if not 0 < args.staged_rollback_lr_factor < 1:
            raise ValueError('--staged-rollback-lr-factor must be in (0, 1).')
    if getattr(args, 'text_lr_contribution_schedule', False):
        if not getattr(args, 'train_text_adapter_only', False):
            raise ValueError('--text-lr-contribution-schedule requires --train_text_adapter_only.')
        if not getattr(args, 'adapter_contribution_diagnostics', False):
            raise ValueError('--text-lr-contribution-schedule requires --adapter-contribution-diagnostics.')
        if not (
            0 <= args.text_lr_contribution_threshold_low
            < args.text_lr_contribution_threshold_high
        ):
            raise ValueError('Text contribution thresholds must satisfy 0 <= low < high.')
        if not (
            0 < args.text_lr_after_high_threshold
            < args.text_lr_after_low_threshold
            < args.lr_text_adapter
        ):
            raise ValueError(
                'Text LR schedule must satisfy 0 < high-threshold LR < '
                'low-threshold LR < initial Text LR.'
            )
    if getattr(args, 'scene_gate_lr_contribution_schedule', False):
        if not getattr(args, 'train_scene_gate_only', False):
            raise ValueError(
                '--scene-gate-lr-contribution-schedule requires '
                '--train_scene_gate_only.'
            )
        if not getattr(args, 'scene_gate_compare_off', False):
            raise ValueError(
                '--scene-gate-lr-contribution-schedule requires '
                '--scene-gate-compare-off.'
            )
        if args.scene_gate_lr_contribution_threshold < 0:
            raise ValueError(
                '--scene-gate-lr-contribution-threshold must be non-negative.'
            )
        if not (
            0 < args.scene_gate_lr_after_threshold < args.lr_scene_gate
        ):
            raise ValueError(
                'SceneGate LR schedule must satisfy 0 < threshold LR < '
                'initial SceneGate LR.'
            )

    # [Debug Info] Print Adapter Config
    if rank == 0:
        print("-" * 50)
        print(f"Config: Use Semantic Adapter: {getattr(args, 'use_semantic_adapter', False)}")
        print(
            f"Config: Use Text Adapter: {getattr(args, 'use_text_adapter', False)}"
            f" | dim={getattr(args, 'text_adapter_dim', 64)}"
            f" | lr={getattr(args, 'lr_text_adapter', 5e-4)}"
        )
        print(
            f"Config: Use ObjCond Adapter: {getattr(args, 'use_obj_cond_adapter', False)}"
            f" | rank={getattr(args, 'obj_cond_rank', 4)}"
            f" | lr={getattr(args, 'lr_obj_cond_adapter', 5e-4)}"
        )
        print(
            f"Config: Use SceneGate: {getattr(args, 'use_scene_gate', False)}"
            f" | version={getattr(args, 'scene_gate_version', 'v5')}"
            f" | type={getattr(args, 'scene_gate_type', 'pair')}"
        )
        print(f"Config: Adapter Dimension: {getattr(args, 'adapt_dim', 64)}")
        print("-" * 50)

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root,
                           clip_model_name=args.clip_model_name, zero_shot=args.zs, zs_type=args.zs_type,
                           num_classes=args.num_classes, args=args)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root,
                          clip_model_name=args.clip_model_name, args=args)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=getattr(args, 'prefetch_factor', 2),
        )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        drop_last=True,
        sampler=DistributedSampler(
            trainset,
            num_replicas=args.world_size,
            rank=rank
        ),
        **loader_kwargs
    )

    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=args.test_batch_size,
        drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            testset, shuffle=False, drop_last=False
        ),
        **loader_kwargs
    )

    args.human_idx = 0
    object_n_verb_to_interaction = trainset.dataset.object_n_verb_to_interaction
    object_to_target = trainset.dataset.object_class_to_target_class

    print('[INFO]: num_classes', args.num_classes)

    # Building Detector (Parameters regarding adapter are handled inside LAIN based on args)
    lain = build_detector(args, object_to_target, object_n_verb_to_interaction=object_n_verb_to_interaction,
                          clip_model_path=args.clip_dir_vit)

    if args.dataset == 'hicodet' and args.eval:
        lain.object_class_to_target_class = testset.dataset.object_class_to_target_class

    resume_training = False
    checkpoint = None
    if args.resume and os.path.exists(args.resume):
        print(f"===>>> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = safe_torch_load(args.resume, map_location='cpu')
        load_state_dict_compatible(
            lain,
            checkpoint['model_state_dict'],
            strict_adapter_shapes=bool(args.eval),
        )
        resume_training = True
    elif getattr(args, "init_from", "") and os.path.exists(args.init_from):
        print(f"===>>> Rank {rank}: initialise model weights from {args.init_from}")
        checkpoint = safe_torch_load(args.init_from, map_location='cpu')
        checkpoint_state = checkpoint['model_state_dict']
        if getattr(args, 'reset_scene_gate_on_load', False):
            checkpoint_state = {
                key: value for key, value in checkpoint_state.items()
                if 'scene_gate' not in key
            }
            print('[INFO] SceneGate checkpoint tensors skipped; using fresh gate initialization.')
        if getattr(args, 'reset_obj_cond_adapter_on_load', False):
            checkpoint_state = {
                key: value for key, value in checkpoint_state.items()
                if 'obj_cond_adapter' not in key
            }
            print('[INFO] ObjectConditionedAdapter checkpoint tensors skipped; using fresh initialization.')
        if getattr(args, 'reset_text_adapter_on_load', False):
            checkpoint_state = {
                key: value for key, value in checkpoint_state.items()
                if 'text_adapter' not in key
            }
            print('[INFO] Text Adapter checkpoint tensors skipped; using fresh initialization.')
        load_state_dict_compatible(
            lain,
            checkpoint_state,
            strict_adapter_shapes=bool(args.eval),
        )
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    engine = CustomisedDLE(
        lain, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        args=args
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval and checkpoint is not None:
        # Evaluation-only diagnostics must use the checkpoint's real epoch so
        # multiple post-hoc ablations append distinct, correctly labelled rows.
        engine.update_state_key(
            epoch=int(checkpoint.get('epoch', 0)),
            iteration=int(checkpoint.get('iteration', 0)),
        )

    if args.eval:
        lain.eval()
        if args.dataset == 'vcoco':
            import eval_vcoco
            ret = engine.cache_vcoco(test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file = 'vcoco/data/splits/vcoco_test.ids'
            vcocoeval = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
            det_file = 'vcoco_cache/cache.pkl'
            ap = vcocoeval._do_eval(ret, ovr_thresh=0.5)
            print(ap)
            return
        if getattr(args, 'eval_scene_gate_off_only', False):
            if checkpoint is None:
                raise RuntimeError('Gate-OFF-only evaluation did not load a checkpoint.')
            real_net = (
                engine._state.net.module
                if hasattr(engine._state.net, 'module')
                else engine._state.net
            )
            original_gate_state = bool(getattr(real_net, 'use_scene_gate', False))
            real_net.tp = None
            real_net.use_scene_gate = False
            try:
                ap_gate_off_only = engine.test_hico(
                    test_loader,
                    args,
                    report=False,
                    collect_gate_diagnostics=False,
                )
            finally:
                real_net.use_scene_gate = original_gate_state
                real_net.tp = None

            gate_off_summary = engine._ap_summary(ap_gate_off_only)
            if rank == 0:
                result = {
                    'schema_version': 1,
                    'evaluation': 'scene_gate_off_only',
                    'checkpoint': os.path.abspath(args.resume),
                    'checkpoint_epoch': int(checkpoint.get('epoch', 0)),
                    'checkpoint_iteration': int(checkpoint.get('iteration', 0)),
                    'scene_gate_enabled_in_checkpoint': original_gate_state,
                    'scene_gate_enabled_during_evaluation': False,
                    'metrics': gate_off_summary,
                    'evaluated_at': __import__('datetime').datetime.now(
                        __import__('datetime').timezone.utc
                    ).isoformat(),
                }
                result_path = os.path.abspath(args.eval_result_json)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                temporary_result = result_path + '.tmp'
                with open(temporary_result, 'w', encoding='utf-8') as handle:
                    json.dump(result, handle, indent=2, ensure_ascii=False)
                os.replace(temporary_result, result_path)
                print(f'[Gate-OFF Eval] checkpoint={args.resume}')
                print(f'[Gate-OFF Eval] metrics={gate_off_summary}')
                print(f'[Gate-OFF Eval] result={result_path}')
            return

        ap = engine.test_hico(test_loader, args)
        scene_gate_diagnostics = engine._last_scene_gate_diagnostics
        ap_gate_off = None
        if getattr(args, 'scene_gate_compare_off', False) and getattr(lain, 'use_scene_gate', False):
            lain.tp = None
            lain.use_scene_gate = False
            try:
                ap_gate_off = engine.test_hico(test_loader, args, report=False)
            finally:
                lain.use_scene_gate = True
                lain.tp = None
        adapter_ablations = engine._evaluate_adapter_ablations(test_loader)
        if getattr(args, 'scene_gate_diagnostics', False) or adapter_ablations:
            engine._save_scene_gate_diagnostics(
                ap,
                ap_gate_off,
                scene_gate_diagnostics,
                adapter_ablations=adapter_ablations,
            )

        # Safe AP calculation handling
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)

        # Handle cases where ap might be shorter than num_classes due to dataset splits
        if len(ap) == args.num_classes:
            print(
                f"The mAP is {ap.mean() * 100:.2f},"
                f" rare: {ap[rare].mean() * 100:.2f},"
                f" none-rare: {ap[non_rare].mean() * 100:.2f},"
            )
        else:
            print(f"The mAP is {ap.mean() * 100:.2f}")

        if args.zs:
            zs_hoi_idx = hico_unseen_index[args.zs_type]
            print(f'>>> zero-shot setting({args.zs_type}!!)')
            ap_unseen = []
            ap_seen = []
            for i, value in enumerate(ap):
                if i in zs_hoi_idx:
                    ap_unseen.append(value)
                else:
                    ap_seen.append(value)

            ap_unseen = torch.as_tensor(ap_unseen).mean()
            ap_seen = torch.as_tensor(ap_seen).mean()
            print(
                f"full mAP: {ap.mean() * 100:.2f}",
                f"unseen: {ap_unseen * 100:.2f}",
                f"seen: {ap_seen * 100:.2f}",
            )

        return

    # -------------------------------------------------------------------------
    # [CRITICAL] Parameter Freezing and Optimizer Configuration
    # -------------------------------------------------------------------------

    # 1. Freeze Detector (DETR)
    for p in lain.detector.parameters():
        p.requires_grad = False

    # 2. Configure CLIP Head (Visual Side + Prompt Learner)
    for n, p in lain.clip_head.named_parameters():
        if (
            getattr(args, 'train_scene_gate_only', False)
            or getattr(args, 'train_obj_cond_adapter_only', False)
            or getattr(args, 'train_object_and_scene_gate_only', False)
        ):
            p.requires_grad = False
            continue
        if getattr(args, 'train_text_adapter_only', False):
            p.requires_grad = 'text_adapter' in n
            continue
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj'):
            p.requires_grad = True
        elif 'adaptermlp' in n or "prompt_learner" in n:
            p.requires_grad = True
        elif 'text_adapter' in n and getattr(args, 'use_text_adapter', False):
            p.requires_grad = True
        elif 'visual_prompt' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 3. Configure Semantic Adapter (Text Side) and other LAIN components
    # 'lain' contains: detector, clip_head, semantic_adapter, priors_downproj, query_proj
    for n, p in lain.named_parameters():
        if getattr(args, 'train_text_adapter_only', False):
            p.requires_grad = 'text_adapter' in n
            continue
        if getattr(args, 'train_scene_gate_only', False):
            p.requires_grad = 'scene_gate' in n
            continue
        if getattr(args, 'train_obj_cond_adapter_only', False):
            p.requires_grad = 'obj_cond_adapter' in n
            continue
        if getattr(args, 'train_object_and_scene_gate_only', False):
            p.requires_grad = 'obj_cond_adapter' in n or 'scene_gate' in n
            continue
        if 'clip_head' in n or 'detector' in n:
            continue  # Already handled above

        if 'semantic_adapter' in n:
            # Only train if args say so
            if getattr(args, 'use_semantic_adapter', False):
                p.requires_grad = True
            else:
                p.requires_grad = False
        elif 'obj_cond_adapter' in n:
            p.requires_grad = getattr(args, 'use_obj_cond_adapter', False)
        elif 'scene_gate' in n:
            p.requires_grad = getattr(args, 'use_scene_gate', False)
        else:
            # Default for other small MLPs (query_proj, priors_downproj) is usually True
            # Assuming we want to train them:
            p.requires_grad = True

    if getattr(args, 'train_scene_gate_only', False) and not getattr(args, 'use_scene_gate', False):
        raise RuntimeError('--train_scene_gate_only requires --use_scene_gate')
    if getattr(args, 'train_object_and_scene_gate_only', False):
        joint_trainable_names = [
            n for n, p in lain.named_parameters() if p.requires_grad
        ]
        object_names = [n for n in joint_trainable_names if 'obj_cond_adapter' in n]
        gate_names = [n for n in joint_trainable_names if 'scene_gate' in n]
        unexpected_names = [
            n for n in joint_trainable_names
            if 'obj_cond_adapter' not in n and 'scene_gate' not in n
        ]
        if not object_names or not gate_names:
            raise RuntimeError(
                '--train-object-and-scene-gate-only requires trainable Object '
                'and SceneGate parameters.'
            )
        if unexpected_names:
            raise RuntimeError(
                '--train-object-and-scene-gate-only left unrelated parameters '
                f'trainable: {unexpected_names[:20]}'
            )
    if getattr(args, 'train_obj_cond_adapter_only', False):
        object_trainable_names = [
            n for n, p in lain.named_parameters() if p.requires_grad
        ]
        if not object_trainable_names:
            raise RuntimeError('--train_obj_cond_adapter_only found no trainable ObjectConditionedAdapter parameters.')
        unexpected_trainable_names = [
            n for n in object_trainable_names if 'obj_cond_adapter' not in n
        ]
        if unexpected_trainable_names:
            raise RuntimeError(
                '--train_obj_cond_adapter_only left non-object parameters trainable: '
                f'{unexpected_trainable_names[:20]}'
            )
    if getattr(args, 'train_text_adapter_only', False):
        text_trainable_names = [
            n for n, p in lain.named_parameters() if p.requires_grad
        ]
        if not text_trainable_names:
            raise RuntimeError('--train_text_adapter_only found no trainable Text Adapter parameters.')
        unexpected_trainable_names = [
            n for n in text_trainable_names if 'text_adapter' not in n
        ]
        if unexpected_trainable_names:
            raise RuntimeError(
                '--train_text_adapter_only left non-text parameters trainable: '
                f'{unexpected_trainable_names[:20]}'
            )

    # 4. Print Trainable Parameters (Sanity Check)
    if rank == 0:
        print("\n" + "=" * 30)
        print("TRAINABLE PARAMETERS:")
        total_params = 0
        trainable_params = 0
        for n, p in lain.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
                print(f" -> {n} ({p.shape})")
        print(f"Total Params: {total_params:,}")
        print(f"Trainable Params: {trainable_params:,} ({trainable_params / total_params:.2%})")
        print("=" * 30 + "\n")

    # 5. Optimizer Grouping
    clip_params = [
        p for n, p in lain.clip_head.named_parameters()
        if p.requires_grad and 'text_adapter' not in n
    ]
    text_adapter_params = [
        p for n, p in lain.named_parameters()
        if p.requires_grad and 'text_adapter' in n
    ]
    obj_cond_params = [
        p for n, p in lain.named_parameters()
        if p.requires_grad and 'obj_cond_adapter' in n
    ]
    delayed_scene_gate = (
        getattr(args, 'use_scene_gate', False)
        and getattr(args, 'scene_gate_start_epoch', 1) > 1
    )
    scene_gate_params = [
        p for n, p in lain.named_parameters()
        if p.requires_grad and 'scene_gate' in n
    ]
    other_params = [
        p for n, p in lain.named_parameters()
        if p.requires_grad
        and 'clip_head' not in n
        and 'detector' not in n
        and 'text_adapter' not in n
        and 'obj_cond_adapter' not in n
        and 'scene_gate' not in n
    ]

    param_dicts = []
    if clip_params:
        param_dicts.append({
            "params": clip_params,
            "lr": args.lr_vit,
            "name": "vit",
        })
    if text_adapter_params:
        param_dicts.append({
            "params": text_adapter_params,
            "lr": getattr(args, "lr_text_adapter", 5e-4),
            "weight_decay": 0.0,
            "name": "text_adapter",
        })
    if obj_cond_params:
        param_dicts.append({
            "params": obj_cond_params,
            "lr": getattr(args, "lr_obj_cond_adapter", 5e-4),
            "weight_decay": 0.0,
            "name": "obj_cond_adapter",
        })
    if scene_gate_params:
        param_dicts.append({
            "params": scene_gate_params,
            "lr": 0.0 if delayed_scene_gate else getattr(args, "lr_scene_gate", 1e-3),
            "name": "scene_gate",
        })
    if other_params:
        param_dicts.append({
            "params": other_params,
            "lr": args.lr_head,
            "name": "head",
        })

    if not param_dicts:
        raise RuntimeError("No trainable parameters found.")

    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_vit,
        weight_decay=args.weight_decay
    )
    lr_scheduler = None
    if not getattr(args, 'disable_lr_scheduler', False):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)

    if resume_training and checkpoint is not None:
        if 'optim_state_dict' in checkpoint:
            try:
                optim.load_state_dict(checkpoint['optim_state_dict'])
            except ValueError as exc:
                print(f"[WARN] Could not load optimizer state: {exc}")
        if 'scheduler_state_dict' in checkpoint and lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except ValueError as exc:
                print(f"[WARN] Could not load scheduler state: {exc}")
        if getattr(args, 'override_resume_scene_gate_lr', False):
            new_gate_lr = float(args.lr_scene_gate)
            gate_group_indices = [
                index
                for index, group in enumerate(optim.param_groups)
                if group.get('name') == 'scene_gate'
            ]
            if not gate_group_indices:
                raise RuntimeError(
                    'Could not override SceneGate LR: optimizer has no '
                    'scene_gate parameter group.'
                )
            for index in gate_group_indices:
                group = optim.param_groups[index]
                group['lr'] = new_gate_lr
                group['initial_lr'] = new_gate_lr
                if lr_scheduler is not None and index < len(lr_scheduler.base_lrs):
                    lr_scheduler.base_lrs[index] = new_gate_lr
                if (
                    lr_scheduler is not None
                    and hasattr(lr_scheduler, '_last_lr')
                    and index < len(lr_scheduler._last_lr)
                ):
                    lr_scheduler._last_lr[index] = new_gate_lr
            print(
                '[Resume] SceneGate LR overridden after optimizer restore: '
                f'{new_gate_lr:.2e}; groups={gate_group_indices}'
            )
        epoch = checkpoint.get('epoch', 0)
        iteration = checkpoint.get('iteration', 0)
        if 'scaler_state_dict' in checkpoint:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch, iteration=iteration,
                                    scaler=scaler)
        else:
            engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch, iteration=iteration)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    completed_epochs = int(checkpoint.get('epoch', 0)) if resume_training and checkpoint is not None else 0
    epochs_to_run = max(args.epochs - completed_epochs, 0)
    if rank == 0 and completed_epochs:
        print(
            f"[Resume] Completed epochs: {completed_epochs}; "
            f"running {epochs_to_run} remaining epoch(s) to reach {args.epochs}."
        )
    if epochs_to_run == 0:
        if rank == 0:
            print(f"[Resume] Checkpoint already reached the requested {args.epochs} epochs.")
        return
    try:
        engine(epochs_to_run)
    except EarlyStopTraining as exc:
        if rank == 0:
            print(f"[Early Stop] {exc}")


if __name__ == '__main__':
    args = get_args()

    # --- [Modification] Argument Patching ---
    # Manually inject arguments if they are missing from utils/args.py
    # This prevents 'AttributeError' without needing to edit the other file immediately.
    if not hasattr(args, 'use_semantic_adapter'):
        setattr(args, 'use_semantic_adapter', False)  # Default to False
    if not hasattr(args, 'adapt_dim'):
        setattr(args, 'adapt_dim', 64)
    # Check command line args manually to overwrite defaults if args.py didn't parse them
    if '--use_semantic_adapter' in sys.argv:
        args.use_semantic_adapter = True

    # Simple parser to grab adapt_dim if passed via CLI but skipped by get_args
    # (This is a quick fix, proper way is editing utils/args.py)
    if '--adapt_dim' in sys.argv:
        try:
            idx = sys.argv.index('--adapt_dim')
            args.adapt_dim = int(sys.argv[idx + 1])
        except:
            pass
    # ----------------------------------------

    print(args)

    if args.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ["MASTER_PORT"] = args.port
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    print('WORLD_SIZE ' + str(os.environ.get("WORLD_SIZE", 1)))

    os.environ["MASTER_ADDR"] = "localhost"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.local_rank = local_rank
    if local_rank == 0:
        # Launch scripts can provide a descriptive experiment name. Fall back
        # to the legacy module-derived name for scripts that do not set it.
        run_name = os.environ.get('WANDB_NAME', '').strip()
        if not run_name:
            run_name = f"LAIN_{args.dataset}"
            if args.use_semantic_adapter:
                run_name += "_w_Adapter"
            if getattr(args, 'use_text_adapter', False):
                run_name += "_text_adapter"
            if getattr(args, 'use_obj_cond_adapter', False):
                run_name += "_obj_cond"
            if getattr(args, 'use_scene_gate', False):
                run_name += f"_scene_{getattr(args, 'scene_gate_version', 'v5')}"
        # Reuse the same online W&B run when a training job is resumed. The
        # launch script restores WANDB_RUN_ID from the run directory and sets
        # WANDB_RESUME=must, so curves continue in one dashboard instead of
        # silently creating a second run.
        wandb_init_kwargs = {
            'project': 'LAIN',
            'name': run_name,
        }
        wandb_run_id = os.environ.get('WANDB_RUN_ID', '').strip()
        wandb_resume = os.environ.get('WANDB_RESUME', '').strip()
        if wandb_run_id:
            wandb_init_kwargs['id'] = wandb_run_id
        if wandb_resume:
            wandb_init_kwargs['resume'] = wandb_resume
        wandb.init(**wandb_init_kwargs)

    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    main(local_rank, args)
