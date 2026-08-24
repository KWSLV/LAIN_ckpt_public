from analysis import HOIErrorAnalyzer
import csv
import json
import os
import shutil
import sys
import time
import torch
import numpy as np
import scipy.io as sio
import wandb
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_POCKET_DIR = os.path.join(REPO_DIR, 'pocket')
if LOCAL_POCKET_DIR not in sys.path:
    sys.path.insert(0, LOCAL_POCKET_DIR)

from utils.hico_text_label import hico_unseen_index
import utils.ddp as ddp
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation
import datetime
import cv2
import torchvision.ops.boxes as box_ops


def to_device(data, device):
    """递归地将数据转移到指定设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if hasattr(data, 'to'):
        return data.to(device)
    return data


class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v

    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]


from torch.cuda import amp
from pocket.ops import relocate_to_cuda


class EarlyStopTraining(RuntimeError):
    """Signal an intentional metric-policy early stop to main.py."""


class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, test_loader=None, args=None, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.net = net
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.train_loader = dataloader
        self.test_loader = test_loader
        self.best_unseen = -1
        self.best_seen = -1
        self.args = args

        # 更加健壮的 Device 获取
        self.device = torch.device(args.device if args and hasattr(args, 'device') else 'cuda')

        if self.args.amp:
            self.scaler = amp.GradScaler(enabled=True)

        self._last_scene_gate_diagnostics = {}
        self._wandb_module_metrics_defined = False
        self._joint_best_unseen = float(
            getattr(args, 'joint_rollback_initial_best_unseen', -1.0)
        )
        self._joint_no_best_streak = 0
        self._joint_best_checkpoint = os.path.join(
            self.args.output_dir, 'best_unseen.pt'
        )
        if getattr(self.args, 'joint_best_rollback_policy', False):
            self._initialise_joint_best_anchor()
        self._staged_policy_ready = False
        self._staged_retry_requested = False
        self._staged_retry_count = 0
        self._staged_total_rollbacks = 0
        self._staged_threshold_reached = False
        self._staged_best_unseen = -float('inf')
        self._staged_initial_lrs = {}
        self._staged_attempt_index = 0
        self._staged_attempt_iteration_start = 0
        self._wandb_resume_heartbeat_last = -float('inf')
        self._unseen_lr_threshold_triggered = False
        self._unseen_lr_threshold_initial_lrs = {}
        if getattr(self.args, 'unseen_lr_threshold_schedule', False):
            threshold_state_path = os.path.join(
                self.args.output_dir, 'unseen_lr_threshold_state.json'
            )
            if os.path.isfile(threshold_state_path):
                try:
                    with open(
                        threshold_state_path, 'r', encoding='utf-8'
                    ) as handle:
                        threshold_state = json.load(handle)
                    self._unseen_lr_threshold_triggered = bool(
                        threshold_state.get('triggered', False)
                    )
                    self._unseen_lr_threshold_initial_lrs = {
                        str(name): float(lr)
                        for name, lr in threshold_state.get(
                            'initial_lrs', {}
                        ).items()
                    }
                    print(
                        '[Unseen LR Schedule] Restored state: '
                        f'triggered={self._unseen_lr_threshold_triggered}, '
                        f'initial_lrs={self._unseen_lr_threshold_initial_lrs}'
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(
                        '[WARN] Could not restore Unseen LR schedule state '
                        f'from {threshold_state_path}: {exc}'
                    )

    def _apply_unseen_lr_threshold_schedule(self, current_unseen):
        """Apply a one-time all-group LR reduction after a strict UO threshold."""
        if not getattr(self.args, 'unseen_lr_threshold_schedule', False):
            return None

        threshold = float(self.args.unseen_lr_threshold)
        factor = float(self.args.unseen_lr_threshold_factor)
        current_unseen = float(current_unseen)

        if not self._unseen_lr_threshold_initial_lrs:
            self._unseen_lr_threshold_initial_lrs = {
                group.get('name', f'group_{index}'): float(group['lr'])
                for index, group in enumerate(self._state.optimizer.param_groups)
            }

        before = {
            group.get('name', f'group_{index}'): float(group['lr'])
            for index, group in enumerate(self._state.optimizer.param_groups)
        }
        triggered_now = False
        if not self._unseen_lr_threshold_triggered and current_unseen > threshold:
            for index, group in enumerate(self._state.optimizer.param_groups):
                name = group.get('name', f'group_{index}')
                reduced_lr = self._unseen_lr_threshold_initial_lrs[name] * factor
                group['lr'] = reduced_lr
                group['initial_lr'] = reduced_lr
            self._unseen_lr_threshold_triggered = True
            triggered_now = True
            print(
                '[Unseen LR Schedule] '
                f'Unseen={current_unseen:.6f} > {threshold:.6f}; '
                f'all active optimizer LRs set to initial LR x {factor:g}.'
            )

        after = {
            group.get('name', f'group_{index}'): float(group['lr'])
            for index, group in enumerate(self._state.optimizer.param_groups)
        }
        payload = {
            'epoch': int(self._state.epoch),
            'current_unseen': current_unseen,
            'threshold': threshold,
            'factor': factor,
            'triggered_now': triggered_now,
            'triggered': bool(self._unseen_lr_threshold_triggered),
            'initial_lrs': dict(self._unseen_lr_threshold_initial_lrs),
            'lrs_before': before,
            'lrs_after': after,
        }

        if self._rank == 0:
            state_path = os.path.join(
                self.args.output_dir, 'unseen_lr_threshold_state.json'
            )
            temporary_path = state_path + '.tmp'
            with open(temporary_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(temporary_path, state_path)

        return payload

    def __call__(self, n):
        if not getattr(self.args, 'lain_object_staged_retry_policy', False):
            return super().__call__(n)

        self.epochs = int(self.args.epochs)
        self._initialise_staged_retry_policy()
        self._on_start()
        logical_epoch = int(self._state.epoch) + 1
        while logical_epoch <= self.epochs:
            retry_count = 0
            while True:
                self._staged_retry_count = retry_count
                self._staged_retry_requested = False
                self._staged_attempt_index += 1
                self._staged_attempt_iteration_start = int(self._state.iteration)
                self._running_loss_sum = None
                self._running_loss_count = 0
                self._state.running_loss.reset()
                self._state.t_data.reset()
                self._state.t_iteration.reset()

                # The base engine increments epoch in _on_start_epoch. Force
                # retries back onto the same logical epoch before calling it.
                self._state.epoch = logical_epoch - 1
                if self._rank == 0:
                    print(
                        '[Staged Retry] Starting logical epoch '
                        f'{logical_epoch}/{self.epochs}, attempt={retry_count + 1}, '
                        f'retry_count={retry_count}'
                    )
                self._on_start_epoch()
                timestamp = time.time()
                for batch in self._train_loader:
                    self._state.inputs = batch[:-1]
                    self._state.targets = batch[-1]
                    self._on_start_iteration()
                    self._state.t_data.append(time.time() - timestamp)

                    self._on_each_iteration()
                    detached_loss = self._state.loss.detach()
                    if self._running_loss_sum is None:
                        self._running_loss_sum = detached_loss
                    else:
                        self._running_loss_sum = self._running_loss_sum + detached_loss
                    self._running_loss_count += 1
                    if self._state.iteration % self._print_interval == 0:
                        mean_loss = self._running_loss_sum / self._running_loss_count
                        self._state.running_loss.append(mean_loss.item())
                        self._running_loss_sum = None
                        self._running_loss_count = 0
                    self._on_end_iteration()
                    self._state.t_iteration.append(time.time() - timestamp)
                    timestamp = time.time()

                self._on_end_epoch()
                if self._staged_retry_requested:
                    retry_count += 1
                    continue
                logical_epoch += 1
                break
        self._on_end()

    def _initialise_staged_retry_policy(self):
        if self._staged_policy_ready:
            return
        self._staged_initial_lrs = {
            'head': float(self.args.lr_head),
            'vit': float(self.args.lr_vit),
            'object': float(self.args.lr_obj_cond_adapter),
        }
        self._staged_best_checkpoint = os.path.join(
            self.args.output_dir, 'best_unseen.pt'
        )
        if getattr(self.args, 'staged_resume_existing_run', False):
            state_path = os.path.join(
                self.args.output_dir, 'staged_retry_state.json'
            )
            if not os.path.isfile(state_path):
                raise FileNotFoundError(
                    f'Staged resume state does not exist: {state_path}'
                )
            with open(state_path, 'r', encoding='utf-8') as handle:
                restored = json.load(handle)
            self._staged_best_unseen = float(restored['best_unseen'])
            self._staged_threshold_reached = bool(
                restored['threshold_reached']
            )
            self._staged_total_rollbacks = int(
                restored.get('total_rollbacks', 0)
            )
            attempts_path = os.path.join(
                self.args.output_dir, 'staged_retry_attempts.jsonl'
            )
            if os.path.isfile(attempts_path):
                with open(attempts_path, 'r', encoding='utf-8') as handle:
                    attempts = [
                        json.loads(line)
                        for line in handle
                        if line.strip()
                    ]
                if attempts:
                    self._staged_attempt_index = max(
                        int(row['attempt_index']) for row in attempts
                    )
            if self._rank == 0:
                print(
                    '[Staged Retry] Restored run state: '
                    f'completed_epoch={self._state.epoch}, '
                    f'best_unseen={self._staged_best_unseen:.6f}, '
                    f'threshold_reached={self._staged_threshold_reached}, '
                    f'total_rollbacks={self._staged_total_rollbacks}'
                )
        self._staged_policy_ready = True
        if self._rank == 0:
            active_lrs = self._staged_group_lrs()
            print(
                '[Staged Retry] Fixed LR scheduler disabled; baseline LRs: '
                f"head={self._staged_initial_lrs['head']:.2e}, "
                f"vit={self._staged_initial_lrs['vit']:.2e}, "
                f"object={self._staged_initial_lrs['object']:.2e}"
            )
            print(
                '[Staged Retry] Active optimizer LRs: '
                f"head={active_lrs['head']:.2e}, "
                f"vit={active_lrs['vit']:.2e}, "
                f"object={active_lrs['object']:.2e}"
            )

    def _staged_group_lrs(self):
        name_map = {
            'head': 'head',
            'vit': 'vit',
            'obj_cond_adapter': 'object',
        }
        result = {}
        for group in self._state.optimizer.param_groups:
            output_name = name_map.get(group.get('name'))
            if output_name is not None:
                result[output_name] = float(group['lr'])
        if set(result) != {'head', 'vit', 'object'}:
            raise RuntimeError(
                'Staged LAIN+Object policy requires exactly the named head, '
                'vit and obj_cond_adapter optimizer groups.'
            )
        return result

    def _apply_staged_trigger_lrs(self):
        targets = {
            name: initial_lr / 10.0
            for name, initial_lr in self._staged_initial_lrs.items()
        }
        optimizer_names = {
            'head': 'head',
            'vit': 'vit',
            'obj_cond_adapter': 'object',
        }
        for group in self._state.optimizer.param_groups:
            output_name = optimizer_names.get(group.get('name'))
            if output_name is None:
                continue
            # Never increase an LR that earlier rollbacks already reduced.
            group['lr'] = min(float(group['lr']), targets[output_name])
            group['initial_lr'] = group['lr']

    def _recreate_staged_optimizer_after_rollback(self):
        factor = float(self.args.staged_rollback_lr_factor)
        new_groups = []
        for group in self._state.optimizer.param_groups:
            new_group = {
                key: value
                for key, value in group.items()
                if key not in {'params', 'lr', 'initial_lr'}
            }
            new_group['params'] = list(group['params'])
            new_group['lr'] = float(group['lr']) * factor
            new_group['initial_lr'] = new_group['lr']
            new_groups.append(new_group)
        self._state.optimizer = torch.optim.AdamW(
            new_groups,
            lr=float(self.args.lr_head),
            weight_decay=float(self.args.weight_decay),
        )
        self._state.lr_scheduler = None
        if self.args.amp:
            self.scaler = amp.GradScaler(enabled=True)
            self._state.scaler = self.scaler

    def _load_staged_best_weights(self):
        if not os.path.isfile(self._staged_best_checkpoint):
            raise FileNotFoundError(
                'Staged retry requested before a protected best checkpoint '
                f'existed: {self._staged_best_checkpoint}'
            )
        try:
            checkpoint = torch.load(
                self._staged_best_checkpoint,
                map_location='cpu',
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self._staged_best_checkpoint,
                map_location='cpu',
            )
        real_net = (
            self._state.net.module
            if hasattr(self._state.net, 'module')
            else self._state.net
        )
        real_net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        real_net.tp = None

    def _append_staged_attempt_log(self, row):
        if self._rank != 0:
            return
        os.makedirs(self.args.output_dir, exist_ok=True)
        attempts_jsonl = os.path.join(
            self.args.output_dir, 'staged_retry_attempts.jsonl'
        )
        attempts_csv = os.path.join(
            self.args.output_dir, 'staged_retry_attempts.csv'
        )
        with open(attempts_jsonl, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        write_header = not os.path.exists(attempts_csv)
        with open(attempts_csv, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        if not row['accepted']:
            failures_jsonl = os.path.join(
                self.args.output_dir, 'staged_retry_failures.jsonl'
            )
            with open(failures_jsonl, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    def _evaluate_staged_attempt(self, summary):
        lrs_used = self._staged_group_lrs()
        best_before = float(self._staged_best_unseen)
        has_best = np.isfinite(best_before)
        decline = best_before - float(summary['unseen']) if has_best else 0.0
        max_drop = float(
            self.args.staged_post_trigger_max_drop
            if self._staged_threshold_reached
            else self.args.staged_pre_trigger_max_drop
        )
        rejected = has_best and decline > max_drop
        threshold_before = bool(self._staged_threshold_reached)
        rollback_reason = ''
        rollback_target = ''

        if rejected:
            rollback_reason = (
                f"unseen_drop_{decline:.6f}_exceeds_"
                f"{'post' if threshold_before else 'pre'}_trigger_limit_"
                f'{max_drop:.6f}'
            )
            rollback_target = os.path.abspath(self._staged_best_checkpoint)
            self._load_staged_best_weights()
            self._recreate_staged_optimizer_after_rollback()
            self._staged_total_rollbacks += 1
            self._staged_retry_requested = True
        else:
            if (
                not self._staged_threshold_reached
                and float(summary['unseen'])
                >= float(self.args.staged_lr_trigger_unseen)
            ):
                self._apply_staged_trigger_lrs()
                self._staged_threshold_reached = True

        lrs_next = self._staged_group_lrs()
        row = {
            'timestamp_utc': datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            'logical_epoch': int(self._state.epoch),
            'attempt': int(self._staged_retry_count + 1),
            'retry_count': int(self._staged_retry_count),
            'attempt_index': int(self._staged_attempt_index),
            'accepted': not rejected,
            'status': 'accepted' if not rejected else 'rejected_rollback_retry',
            'full_mAP': float(summary['full']),
            'unseen_mAP': float(summary['unseen']),
            'seen_mAP': float(summary['seen']),
            'historical_best_before': best_before if has_best else '',
            'decline_from_best': float(decline),
            'allowed_decline': float(max_drop),
            'threshold_reached_before': threshold_before,
            'threshold_reached_after': bool(self._staged_threshold_reached),
            'one_time_lr_drop_applied': (
                not threshold_before and self._staged_threshold_reached
            ),
            'lr_head_used': lrs_used['head'],
            'lr_vit_used': lrs_used['vit'],
            'lr_object_used': lrs_used['object'],
            'lr_head_next': lrs_next['head'],
            'lr_vit_next': lrs_next['vit'],
            'lr_object_next': lrs_next['object'],
            'rollback_count': int(self._staged_retry_count + (1 if rejected else 0)),
            'rollback_count_total': int(self._staged_total_rollbacks),
            'rollback_reason': rollback_reason,
            'rollback_target_checkpoint': rollback_target,
            'iteration_end': int(self._state.iteration),
            'object_forward_active': bool(
                int(self._state.epoch)
                >= int(self.args.staged_object_start_epoch)
            ),
        }
        self._append_staged_attempt_log(row)
        if self._rank == 0:
            print(
                '[Staged Retry] '
                f"logical_epoch={row['logical_epoch']} attempt={row['attempt']} "
                f"unseen={row['unseen_mAP']:.6f} "
                f"best_before={row['historical_best_before']} "
                f"decline={row['decline_from_best']:.6f}/"
                f"{row['allowed_decline']:.6f} status={row['status']} "
                f"lr(head/vit/object)={row['lr_head_used']:.2e}/"
                f"{row['lr_vit_used']:.2e}/{row['lr_object_used']:.2e} -> "
                f"{row['lr_head_next']:.2e}/{row['lr_vit_next']:.2e}/"
                f"{row['lr_object_next']:.2e}"
            )
        return row

    def _initialise_joint_best_anchor(self):
        """Protect the supplied checkpoint before the first joint update."""
        source_path = os.path.abspath(self.args.init_from)
        best_path = os.path.abspath(self._joint_best_checkpoint)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f'Joint rollback anchor checkpoint does not exist: {source_path}'
            )

        if self._rank == 0:
            os.makedirs(self.args.output_dir, exist_ok=True)
            temporary_best = best_path + '.tmp'
            if os.path.exists(temporary_best):
                os.remove(temporary_best)
            try:
                try:
                    os.link(source_path, temporary_best)
                except OSError:
                    shutil.copy2(source_path, temporary_best)
                os.replace(temporary_best, best_path)
            finally:
                if os.path.exists(temporary_best):
                    os.remove(temporary_best)

            metadata = {
                'selection_metric': 'unseen_mAP',
                'selection_partition': self.args.partitions[1],
                'unseen_mAP': self._joint_best_unseen,
                'epoch': 0,
                'iteration': 0,
                'source_checkpoint': source_path,
                'checkpoint': 'best_unseen.pt',
                'initial_joint_search_anchor': True,
                'warning': (
                    'Selected on the test partition; use for analysis only, '
                    'not as an unbiased final paper result.'
                ),
            }
            metadata_path = os.path.join(
                self.args.output_dir, 'best_unseen_checkpoint.json'
            )
            temporary_metadata = metadata_path + '.tmp'
            with open(temporary_metadata, 'w', encoding='utf-8') as handle:
                json.dump(metadata, handle, indent=2, ensure_ascii=False)
            os.replace(temporary_metadata, metadata_path)
            print(
                '[Joint Best-Rollback] Protected initial anchor: '
                f'unseen={self._joint_best_unseen:.6f} path={best_path}'
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self.best_unseen = self._joint_best_unseen

    def _joint_group_lrs(self):
        result = {}
        for group in self._state.optimizer.param_groups:
            name = group.get('name')
            if name in {'obj_cond_adapter', 'scene_gate'}:
                result[name] = float(group['lr'])
        if set(result) != {'obj_cond_adapter', 'scene_gate'}:
            raise RuntimeError(
                'Joint best-rollback policy requires named obj_cond_adapter '
                'and scene_gate optimizer groups.'
            )
        return result

    def _scale_joint_lrs(self, factor):
        factor = float(factor)
        for index, group in enumerate(self._state.optimizer.param_groups):
            if group.get('name') not in {'obj_cond_adapter', 'scene_gate'}:
                continue
            group['lr'] = float(group['lr']) * factor
            group['initial_lr'] = group['lr']
            scheduler = self._state.lr_scheduler
            if scheduler is not None and index < len(scheduler.base_lrs):
                scheduler.base_lrs[index] = group['lr']
            if (
                scheduler is not None
                and hasattr(scheduler, '_last_lr')
                and index < len(scheduler._last_lr)
            ):
                scheduler._last_lr[index] = group['lr']

    def _rollback_to_joint_best(self, lr_factor):
        """Discard bad weights/momenta and restart locally from protected best."""
        try:
            checkpoint = torch.load(
                self._joint_best_checkpoint,
                map_location='cpu',
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self._joint_best_checkpoint,
                map_location='cpu',
            )
        real_net = (
            self._state.net.module
            if hasattr(self._state.net, 'module')
            else self._state.net
        )
        real_net.load_state_dict(checkpoint['model_state_dict'], strict=True)

        # Adam moments belong to the rejected point and must not leak into the
        # next search step. Keep parameter groups, but rebuild all state.
        self._state.optimizer.state.clear()
        self._scale_joint_lrs(lr_factor)
        self._state.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._state.optimizer,
            step_size=int(self.args.lr_drop),
        )
        if self.args.amp:
            self.scaler = amp.GradScaler(enabled=True)
            self._state.scaler = self.scaler

    def _apply_joint_best_rollback_policy(self, current_unseen):
        best_before = float(self._joint_best_unseen)
        current_unseen = float(current_unseen)
        delta = current_unseen - best_before
        threshold = float(self.args.joint_rollback_significant_delta)
        lrs_before = self._joint_group_lrs()
        new_best = delta > 0.0
        rollback = False
        optimizer_reset = False
        action = 'keep_lr_significant_improvement'

        if new_best:
            self._joint_no_best_streak = 0
            self._joint_best_unseen = current_unseen
            if delta < threshold:
                self._scale_joint_lrs(
                    self.args.joint_rollback_small_lr_factor
                )
                action = 'halve_lr_small_improvement'
        else:
            self._joint_no_best_streak += 1
            if delta < -threshold:
                self._rollback_to_joint_best(
                    self.args.joint_rollback_bad_lr_factor
                )
                rollback = True
                optimizer_reset = True
                action = 'rollback_best_reset_optimizer_reduce_lr'
            else:
                self._scale_joint_lrs(
                    self.args.joint_rollback_small_lr_factor
                )
                action = 'halve_lr_nonsevere_no_best'

        lrs_after = self._joint_group_lrs()
        early_stop = (
            self._joint_no_best_streak
            >= int(self.args.joint_rollback_patience)
        )
        policy = {
            'best_unseen_before': best_before,
            'current_unseen': current_unseen,
            'delta_vs_best': delta,
            'significant_delta': threshold,
            'new_best': new_best,
            'rollback': rollback,
            'optimizer_reset': optimizer_reset,
            'action': action,
            'object_lr_evaluated_epoch': lrs_before['obj_cond_adapter'],
            'gate_lr_evaluated_epoch': lrs_before['scene_gate'],
            'object_lr_next_epoch': lrs_after['obj_cond_adapter'],
            'gate_lr_next_epoch': lrs_after['scene_gate'],
            'no_best_streak': self._joint_no_best_streak,
            'patience': int(self.args.joint_rollback_patience),
            'early_stop': early_stop,
        }
        if self._rank == 0:
            print(
                '[Joint Best-Rollback] '
                f'epoch={self._state.epoch} current={current_unseen:.6f} '
                f'best_before={best_before:.6f} delta={delta:+.6f} '
                f'action={action} rollback={rollback} '
                f"object_lr={lrs_before['obj_cond_adapter']:.2e}->"
                f"{lrs_after['obj_cond_adapter']:.2e} "
                f"gate_lr={lrs_before['scene_gate']:.2e}->"
                f"{lrs_after['scene_gate']:.2e} "
                f'no_best={self._joint_no_best_streak}/'
                f'{self.args.joint_rollback_patience}'
            )
        return policy

    def save_checkpoint(self, unseen_map=None):
        """Save a checkpoint, retain the best unseen alias, and rotate old files."""
        super().save_checkpoint()

        if self._rank != 0:
            return

        current_filename = (
            f"ckpt_{int(self._state.iteration):05d}_"
            f"{int(self._state.epoch):02d}.pt"
        )
        current_path = os.path.join(self.args.output_dir, current_filename)

        if (
            getattr(self.args, 'keep_best_unseen_checkpoint', False)
            and unseen_map is not None
            and os.path.isfile(current_path)
        ):
            metadata_path = os.path.join(
                self.args.output_dir,
                'best_unseen_checkpoint.json',
            )
            previous_best = float(self.best_unseen)
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as handle:
                        previous_best = float(json.load(handle).get('unseen_mAP', previous_best))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    previous_best = float(self.best_unseen)

            current_unseen = float(unseen_map)
            if current_unseen > previous_best:
                best_path = os.path.join(self.args.output_dir, 'best_unseen.pt')
                temporary_best = best_path + '.tmp'
                try:
                    if os.path.exists(temporary_best):
                        os.remove(temporary_best)
                    try:
                        os.link(current_path, temporary_best)
                    except OSError:
                        shutil.copy2(current_path, temporary_best)
                    os.replace(temporary_best, best_path)

                    metadata = {
                        'selection_metric': 'unseen_mAP',
                        'selection_partition': self.args.partitions[1],
                        'unseen_mAP': current_unseen,
                        'epoch': int(self._state.epoch),
                        'iteration': int(self._state.iteration),
                        'source_checkpoint': current_filename,
                        'checkpoint': 'best_unseen.pt',
                        'warning': (
                            'Selected on the test partition; use for analysis only, '
                            'not as an unbiased final paper result.'
                        ),
                    }
                    temporary_metadata = metadata_path + '.tmp'
                    with open(temporary_metadata, 'w', encoding='utf-8') as handle:
                        json.dump(metadata, handle, indent=2, ensure_ascii=False)
                    os.replace(temporary_metadata, metadata_path)
                    self.best_unseen = current_unseen
                    print(
                        f"[Checkpoint] New best unseen mAP={current_unseen:.4f} "
                        f"at epoch {self._state.epoch}: {best_path}"
                    )
                finally:
                    if os.path.exists(temporary_best):
                        os.remove(temporary_best)

        keep = max(0, int(getattr(self.args, 'keep_last_checkpoints', 0) or 0))
        milestone_epochs = {
            int(epoch)
            for epoch in (getattr(self.args, 'keep_epoch_checkpoints', None) or [])
            if int(epoch) > 0
        }
        if keep <= 0 and not milestone_epochs:
            return

        checkpoints = []
        try:
            filenames = os.listdir(self.args.output_dir)
        except OSError as exc:
            print(f"[WARN] Could not inspect checkpoints for rotation: {exc}")
            return

        for filename in filenames:
            if not filename.startswith('ckpt_') or not filename.endswith('.pt'):
                continue
            fields = filename[5:-3].split('_')
            if len(fields) != 2:
                continue
            try:
                iteration, epoch = (int(value) for value in fields)
            except ValueError:
                continue
            checkpoints.append((epoch, iteration, filename))

        checkpoints.sort(reverse=True)
        survivors = {
            filename
            for epoch, _, filename in checkpoints
            if epoch in milestone_epochs
        }
        survivors.update(filename for _, _, filename in checkpoints[:keep])

        for _, _, filename in checkpoints:
            if filename in survivors:
                continue
            path = os.path.join(self.args.output_dir, filename)
            try:
                os.remove(path)
                print(f"[Checkpoint] Removed old checkpoint: {path}")
            except OSError as exc:
                print(f"[WARN] Could not remove old checkpoint {path}: {exc}")

    def _on_start_epoch(self):
        super()._on_start_epoch()
        real_net = self._state.net.module if hasattr(self._state.net, 'module') else self._state.net
        if getattr(self.args, 'lain_object_staged_retry_policy', False):
            object_active = (
                int(self._state.epoch)
                >= int(self.args.staged_object_start_epoch)
            )
            real_net.use_obj_cond_adapter = object_active
            real_net.tp = None
            for name, parameter in real_net.named_parameters():
                if 'obj_cond_adapter' in name:
                    parameter.requires_grad = object_active
            object_trainable = [
                name for name, parameter in real_net.named_parameters()
                if 'obj_cond_adapter' in name and parameter.requires_grad
            ]
            forbidden_trainable = [
                name for name, parameter in real_net.named_parameters()
                if parameter.requires_grad
                and (
                    'text_adapter' in name
                    or 'scene_gate' in name
                    or name.startswith('detector.')
                )
            ]
            if object_active and not object_trainable:
                raise RuntimeError('Object stage is active but Object parameters are frozen.')
            if not object_active and object_trainable:
                raise RuntimeError('Object parameters must be frozen during LAIN-only epochs.')
            if forbidden_trainable:
                raise RuntimeError(
                    'Staged training left forbidden Text/Gate/DETR parameters '
                    f'trainable: {forbidden_trainable[:20]}'
                )
            if getattr(real_net, 'use_scene_gate', False):
                raise RuntimeError('SceneGate must remain OFF in staged LAIN+Object training.')
            if getattr(real_net, 'use_text_adapter', False):
                raise RuntimeError('Text Adapter must remain OFF in staged LAIN+Object training.')
            if self._state.lr_scheduler is not None:
                raise RuntimeError('Fixed LR scheduler unexpectedly active in staged LAIN+Object training.')
            if self._rank == 0:
                stage = 'LAIN-only, Object forward BYPASS' if not object_active else 'LAIN+Object'
                print(
                    f'[Staged Modules] logical_epoch={self._state.epoch}: '
                    f'{stage}; Gate=OFF; Text=OFF'
                )
        if not getattr(self.args, 'use_scene_gate', False) or not hasattr(real_net, 'scene_gate'):
            return

        start_epoch = getattr(self.args, 'scene_gate_start_epoch', 1)
        gate_active = self._state.epoch >= start_epoch
        real_net.use_scene_gate = gate_active

        if start_epoch > 1:
            for group in self._state.optimizer.param_groups:
                if group.get('name') == 'scene_gate':
                    if gate_active and self._state.epoch == start_epoch:
                        group['lr'] = getattr(self.args, 'lr_scene_gate', 1e-3)
                    elif not gate_active:
                        group['lr'] = 0.0

        if self._rank == 0:
            if gate_active:
                print(
                    f"[SceneGate Schedule] Epoch {self._state.epoch}: enabled "
                    f"(start epoch={start_epoch}, lr={getattr(self.args, 'lr_scene_gate', 1e-3):.2e})"
                )
            else:
                print(
                    f"[SceneGate Schedule] Epoch {self._state.epoch}: bypassed "
                    f"(will enable at epoch {start_epoch})"
                )

    def _ap_summary(self, ap):
        summary = {"full": float(ap.mean().item() * 100)}
        if self.args.zs:
            unseen = set(hico_unseen_index[self.args.zs_type])
            unseen_ap = torch.as_tensor([value for idx, value in enumerate(ap) if idx in unseen]).mean()
            seen_ap = torch.as_tensor([value for idx, value in enumerate(ap) if idx not in unseen]).mean()
            summary.update({
                "unseen": float(unseen_ap.item() * 100),
                "seen": float(seen_ap.item() * 100),
            })
        return summary

    def _save_scene_gate_diagnostics(
        self,
        ap_on,
        ap_off,
        diagnostics,
        adapter_ablations=None,
        joint_training_policy=None,
        staged_training_metrics=None,
    ):
        if self._rank != 0:
            return

        adapter_ablations = adapter_ablations or {}
        row = {
            "epoch": int(self._state.epoch),
            "on": self._ap_summary(ap_on),
            "off": self._ap_summary(ap_off) if ap_off is not None else {},
            "diagnostics": diagnostics,
        }
        if joint_training_policy:
            row['joint_training_policy'] = joint_training_policy
        if staged_training_metrics:
            row['staged_training_metrics'] = staged_training_metrics
        if row["off"]:
            row["gap"] = {
                key: row["on"][key] - row["off"][key]
                for key in row["on"].keys()
            }
        else:
            row["gap"] = {}

        # Contributions are conditional single-module ablations:
        #   contribution = full model mAP - mAP with that adapter bypassed
        # All other modules, including SceneGate, remain enabled. Preserve the
        # existing ON/OFF/GAP schema so historical readers keep working.
        adapter_off = {
            name: self._ap_summary(ap)
            for name, ap in adapter_ablations.items()
            if ap is not None
        }
        adapter_contribution = {
            name: {
                key: row["on"][key] - summary[key]
                for key in row["on"].keys()
            }
            for name, summary in adapter_off.items()
        }
        row["adapter_ablation"] = {
            "definition": "full_model_on_minus_single_adapter_off",
            "off": adapter_off,
            "contribution": adapter_contribution,
        }

        # Optional monotonic Text Adapter LR schedule. The contribution is
        # measured in mAP points as full-model Unseen minus Text-OFF Unseen.
        # Evaluation happens after the current epoch, so the updated LR is
        # stored in the checkpoint and takes effect from the next epoch.
        text_lr_schedule = {}
        if getattr(self.args, 'text_lr_contribution_schedule', False):
            text_contribution = adapter_contribution.get(
                'text_adapter', {}
            ).get('unseen')
            if text_contribution is None:
                raise RuntimeError(
                    'Text contribution LR schedule requires a Text Adapter '
                    'Unseen contribution result.'
                )
            text_group_indices = [
                index
                for index, group in enumerate(self._state.optimizer.param_groups)
                if group.get('name') == 'text_adapter'
            ]
            if not text_group_indices:
                raise RuntimeError(
                    'Text contribution LR schedule found no text_adapter optimizer group.'
                )

            low_threshold = float(
                self.args.text_lr_contribution_threshold_low
            )
            high_threshold = float(
                self.args.text_lr_contribution_threshold_high
            )
            requested_lr = None
            trigger = 'none'
            if text_contribution > high_threshold:
                requested_lr = float(self.args.text_lr_after_high_threshold)
                trigger = f'unseen_contribution>{high_threshold}'
            elif text_contribution > low_threshold:
                requested_lr = float(self.args.text_lr_after_low_threshold)
                trigger = f'unseen_contribution>{low_threshold}'

            lr_before = min(
                float(self._state.optimizer.param_groups[index]['lr'])
                for index in text_group_indices
            )
            lr_after = lr_before
            if requested_lr is not None:
                lr_after = min(lr_before, requested_lr)
                for index in text_group_indices:
                    group = self._state.optimizer.param_groups[index]
                    group['lr'] = min(float(group['lr']), requested_lr)
                    group['initial_lr'] = min(
                        float(group.get('initial_lr', group['lr'])),
                        requested_lr,
                    )
                    if (
                        self._state.lr_scheduler is not None
                        and index < len(self._state.lr_scheduler.base_lrs)
                    ):
                        self._state.lr_scheduler.base_lrs[index] = min(
                            float(self._state.lr_scheduler.base_lrs[index]),
                            requested_lr,
                        )
                    if (
                        self._state.lr_scheduler is not None
                        and hasattr(self._state.lr_scheduler, '_last_lr')
                        and index < len(self._state.lr_scheduler._last_lr)
                    ):
                        self._state.lr_scheduler._last_lr[index] = group['lr']

            text_lr_schedule = {
                'unseen_contribution': float(text_contribution),
                'lr_before': float(lr_before),
                'lr_next_epoch': float(lr_after),
                'trigger': trigger,
            }
            row['text_lr_schedule'] = text_lr_schedule
            print(
                '[Text LR Schedule] '
                f"epoch={row['epoch']} contribution={text_contribution:+.6f} "
                f'lr={lr_before:.2e}->{lr_after:.2e} trigger={trigger}'
            )

        # Optional monotonic SceneGate LR schedule. Gate contribution is the
        # full-model Unseen mAP minus Gate-OFF Unseen mAP. The update happens
        # after this epoch's evaluation and applies from the next epoch.
        scene_gate_lr_schedule = {}
        if getattr(self.args, 'scene_gate_lr_contribution_schedule', False):
            gate_contribution = row.get('gap', {}).get('unseen')
            if gate_contribution is None:
                raise RuntimeError(
                    'SceneGate contribution LR schedule requires a Gate-OFF '
                    'Unseen contribution result.'
                )
            gate_group_indices = [
                index
                for index, group in enumerate(self._state.optimizer.param_groups)
                if group.get('name') == 'scene_gate'
            ]
            if not gate_group_indices:
                raise RuntimeError(
                    'SceneGate contribution LR schedule found no scene_gate '
                    'optimizer group.'
                )

            threshold = float(
                self.args.scene_gate_lr_contribution_threshold
            )
            requested_lr = float(
                self.args.scene_gate_lr_after_threshold
            )
            lr_before = min(
                float(self._state.optimizer.param_groups[index]['lr'])
                for index in gate_group_indices
            )
            lr_after = lr_before
            trigger = 'none'
            if gate_contribution > threshold:
                lr_after = min(lr_before, requested_lr)
                trigger = f'unseen_contribution>{threshold}'
                for index in gate_group_indices:
                    group = self._state.optimizer.param_groups[index]
                    group['lr'] = min(float(group['lr']), requested_lr)
                    group['initial_lr'] = min(
                        float(group.get('initial_lr', group['lr'])),
                        requested_lr,
                    )
                    if (
                        self._state.lr_scheduler is not None
                        and index < len(self._state.lr_scheduler.base_lrs)
                    ):
                        self._state.lr_scheduler.base_lrs[index] = min(
                            float(self._state.lr_scheduler.base_lrs[index]),
                            requested_lr,
                        )
                    if (
                        self._state.lr_scheduler is not None
                        and hasattr(self._state.lr_scheduler, '_last_lr')
                        and index < len(self._state.lr_scheduler._last_lr)
                    ):
                        self._state.lr_scheduler._last_lr[index] = group['lr']

            scene_gate_lr_schedule = {
                'unseen_contribution': float(gate_contribution),
                'threshold': float(threshold),
                'lr_before': float(lr_before),
                'lr_next_epoch': float(lr_after),
                'trigger': trigger,
            }
            row['scene_gate_lr_schedule'] = scene_gate_lr_schedule
            print(
                '[SceneGate LR Schedule] '
                f"epoch={row['epoch']} contribution={gate_contribution:+.6f} "
                f'lr={lr_before:.2e}->{lr_after:.2e} trigger={trigger}'
            )

        # Expose all three conditional module contributions in W&B after every
        # epoch. Gate contribution is full model minus Gate-OFF; Text/Object
        # contributions are full model minus the corresponding adapter-OFF.
        if wandb.run is not None:
            contribution_log = {"module_contribution/epoch": row["epoch"]}
            if text_lr_schedule:
                contribution_log[
                    'optimization/text_adapter_lr_next_epoch'
                ] = text_lr_schedule['lr_next_epoch']
            if scene_gate_lr_schedule:
                contribution_log[
                    'optimization/scene_gate_lr_next_epoch'
                ] = scene_gate_lr_schedule['lr_next_epoch']
            if joint_training_policy:
                contribution_log.update({
                    'optimization/object_lr_evaluated_epoch': (
                        joint_training_policy['object_lr_evaluated_epoch']
                    ),
                    'optimization/gate_lr_evaluated_epoch': (
                        joint_training_policy['gate_lr_evaluated_epoch']
                    ),
                    'optimization/object_lr_next_epoch': (
                        joint_training_policy['object_lr_next_epoch']
                    ),
                    'optimization/gate_lr_next_epoch': (
                        joint_training_policy['gate_lr_next_epoch']
                    ),
                    'optimization/rollback': int(
                        joint_training_policy['rollback']
                    ),
                    'optimization/new_best': int(
                        joint_training_policy['new_best']
                    ),
                    'optimization/no_best_streak': (
                        joint_training_policy['no_best_streak']
                    ),
                    'optimization/delta_vs_best': (
                        joint_training_policy['delta_vs_best']
                    ),
                })
            if staged_training_metrics:
                contribution_log.update({
                    'lr/head': staged_training_metrics['lr_head_used'],
                    'lr/vit': staged_training_metrics['lr_vit_used'],
                    'lr/object': staged_training_metrics['lr_object_used'],
                    'lr_next/head': staged_training_metrics['lr_head_next'],
                    'lr_next/vit': staged_training_metrics['lr_vit_next'],
                    'lr_next/object': staged_training_metrics['lr_object_next'],
                    'retry_count': staged_training_metrics['retry_count'],
                    'rollback_count': staged_training_metrics['rollback_count'],
                    'rollback_count_total': staged_training_metrics['rollback_count_total'],
                    'one_time_lr_drop_applied': int(
                        staged_training_metrics['one_time_lr_drop_applied']
                    ),
                    'object_forward_active': int(
                        staged_training_metrics['object_forward_active']
                    ),
                })
            for metric in ("full", "unseen", "seen"):
                if metric in row["on"]:
                    contribution_log[f"module_ablation/full_{metric}"] = row["on"][metric]
                if metric in row["gap"]:
                    contribution_log[f"module_contribution/gate_{metric}"] = row["gap"][metric]
                    contribution_log[f"module_ablation/gate_off_{metric}"] = row["off"][metric]
                for adapter_name, short_name in (
                    ("text_adapter", "text"),
                    ("object_conditioned_adapter", "object"),
                ):
                    value = adapter_contribution.get(adapter_name, {}).get(metric)
                    if value is not None:
                        contribution_log[
                            f"module_contribution/{short_name}_{metric}"
                        ] = value
                        contribution_log[
                            f"module_ablation/{short_name}_off_{metric}"
                        ] = adapter_off[adapter_name][metric]
            if not self._wandb_module_metrics_defined:
                wandb.define_metric("module_contribution/epoch")
                for metric_name in contribution_log:
                    if metric_name != "module_contribution/epoch":
                        wandb.define_metric(
                            metric_name,
                            step_metric="module_contribution/epoch",
                        )
                self._wandb_module_metrics_defined = True
            wandb.log(contribution_log)

        os.makedirs(self.args.output_dir, exist_ok=True)
        json_path = os.path.join(self.args.output_dir, "scene_gate_diagnostics.json")
        rows = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as handle:
                    rows = json.load(handle)
            except (json.JSONDecodeError, OSError):
                rows = []
        rows = [item for item in rows if int(item.get("epoch", -1)) != row["epoch"]]
        rows.append(row)
        rows.sort(key=lambda item: int(item.get("epoch", 0)))
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)

        csv_rows = []
        for item in rows:
            diag = item.get("diagnostics", {})
            flattened = {
                "epoch": item.get("epoch", 0),
                "on_full": item.get("on", {}).get("full", 0.0),
                "on_unseen": item.get("on", {}).get("unseen", 0.0),
                "on_seen": item.get("on", {}).get("seen", 0.0),
                "off_full": item.get("off", {}).get("full", 0.0),
                "off_unseen": item.get("off", {}).get("unseen", 0.0),
                "off_seen": item.get("off", {}).get("seen", 0.0),
                "gap_full": item.get("gap", {}).get("full", 0.0),
                "gap_unseen": item.get("gap", {}).get("unseen", 0.0),
                "gap_seen": item.get("gap", {}).get("seen", 0.0),
            }
            ablation = item.get("adapter_ablation", {})
            adapter_off = ablation.get("off", {})
            adapter_contribution = ablation.get("contribution", {})
            for adapter_name in ("text_adapter", "object_conditioned_adapter"):
                for field in ("full", "unseen", "seen"):
                    flattened[f"{adapter_name}_off_{field}"] = (
                        adapter_off.get(adapter_name, {}).get(field, 0.0)
                    )
                    flattened[f"{adapter_name}_contribution_{field}"] = (
                        adapter_contribution.get(adapter_name, {}).get(field, 0.0)
                    )
            lr_schedule = item.get('text_lr_schedule', {})
            flattened['text_lr_unseen_contribution'] = lr_schedule.get(
                'unseen_contribution', 0.0
            )
            flattened['text_lr_before'] = lr_schedule.get('lr_before', 0.0)
            flattened['text_lr_next_epoch'] = lr_schedule.get(
                'lr_next_epoch', 0.0
            )
            flattened['text_lr_trigger'] = lr_schedule.get('trigger', '')
            gate_lr_schedule = item.get('scene_gate_lr_schedule', {})
            flattened['scene_gate_lr_unseen_contribution'] = (
                gate_lr_schedule.get('unseen_contribution', 0.0)
            )
            flattened['scene_gate_lr_threshold'] = gate_lr_schedule.get(
                'threshold', 0.0
            )
            flattened['scene_gate_lr_before'] = gate_lr_schedule.get(
                'lr_before', 0.0
            )
            flattened['scene_gate_lr_next_epoch'] = gate_lr_schedule.get(
                'lr_next_epoch', 0.0
            )
            flattened['scene_gate_lr_trigger'] = gate_lr_schedule.get(
                'trigger', ''
            )
            joint_policy = item.get('joint_training_policy', {})
            for field in (
                'best_unseen_before',
                'current_unseen',
                'delta_vs_best',
                'significant_delta',
                'new_best',
                'rollback',
                'optimizer_reset',
                'action',
                'object_lr_evaluated_epoch',
                'gate_lr_evaluated_epoch',
                'object_lr_next_epoch',
                'gate_lr_next_epoch',
                'no_best_streak',
                'patience',
                'early_stop',
            ):
                flattened[f'joint_{field}'] = joint_policy.get(field, '')
            staged_metrics = item.get('staged_training_metrics', {})
            for field in (
                'attempt',
                'retry_count',
                'full_mAP',
                'unseen_mAP',
                'seen_mAP',
                'historical_best_before',
                'decline_from_best',
                'allowed_decline',
                'threshold_reached_before',
                'threshold_reached_after',
                'one_time_lr_drop_applied',
                'lr_head_used',
                'lr_vit_used',
                'lr_object_used',
                'lr_head_next',
                'lr_vit_next',
                'lr_object_next',
                'rollback_count',
                'rollback_count_total',
                'object_forward_active',
            ):
                flattened[f'staged_{field}'] = staged_metrics.get(field, '')
            for name in ("raw", "u", "u_rel", "ratio"):
                for field in ("mean", "std", "min", "max", "p50", "p90"):
                    flattened[f"{name}_{field}"] = diag.get(name, {}).get(field, 0.0)
            csv_rows.append(flattened)

        csv_path = os.path.join(self.args.output_dir, "scene_gate_diagnostics.csv")
        if csv_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

        print(
            f"[SceneGate] ON={row['on']} OFF={row['off']} GAP={row['gap']}"
        )
        if adapter_off:
            print(
                "[Adapter Contribution] "
                f"OFF={adapter_off} CONTRIBUTION={adapter_contribution}"
            )
        print(f"[SceneGate] diagnostics saved to {json_path} and {csv_path}")

    def _evaluate_adapter_ablations(self, dataloader):
        """Measure conditional Text/Object Adapter mAP contributions.

        Each pass disables exactly one adapter while leaving every other module
        in its full-model state. Evaluation is sequential and under no_grad, so
        this changes neither weights nor optimizer state.
        """
        if not getattr(self.args, 'adapter_contribution_diagnostics', False):
            return {}

        real_net = self._state.net.module if hasattr(self._state.net, 'module') else self._state.net
        ablations = {}
        candidates = [("text_adapter", "use_text_adapter")]
        if not getattr(
            self.args,
            'skip_object_adapter_contribution',
            False,
        ):
            candidates.append(
                ("object_conditioned_adapter", "use_obj_cond_adapter")
            )
        for output_name, attribute_name in candidates:
            original_value = bool(getattr(real_net, attribute_name, False))
            if not original_value:
                continue

            # tp caches evaluation text features. It must be cleared both before
            # and after toggling Text Adapter, otherwise the OFF pass could reuse
            # ON features and falsely report zero contribution.
            real_net.tp = None
            setattr(real_net, attribute_name, False)
            try:
                ablations[output_name] = self.test_hico(
                    dataloader,
                    self.args,
                    report=False,
                    collect_gate_diagnostics=False,
                )
            finally:
                setattr(real_net, attribute_name, original_value)
                real_net.tp = None

        return ablations

    def _on_end_iteration(self):
        # Print stats in the master process
        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.inputs = to_device(self._state.inputs, self.device)
        self._state.targets = to_device(self._state.targets, self.device)

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data_total = self._state.t_data.sum() / self._world_size
        t_iter_total = self._state.t_iteration.sum() / self._world_size

        t_iter_mean = self._state.t_iteration.mean()
        t_data_mean = self._state.t_data.mean()

        it_sec = t_iter_mean + t_data_mean

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            if getattr(self.args, 'lain_object_staged_retry_policy', False):
                current_iter = (
                    self._state.iteration
                    - self._staged_attempt_iteration_start
                )
            else:
                current_iter = self._state.iteration - num_iter * (self._state.epoch - 1)
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time/iter[Data/Train/Total]: [{:.3f}s/{:.3f}s/{:.3f}s], "
                "Window[Data/Train]: [{:.2f}s/{:.2f}s], "
                "Remain: {}".format(
                    self._state.epoch, self.args.epochs,
                    str(current_iter).zfill(n_d),
                    num_iter, running_loss,
                    t_data_mean, t_iter_mean, it_sec,
                    t_data_total, t_iter_total,
                    datetime.timedelta(seconds=(num_iter - current_iter) * it_sec)
                ))
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_each_iteration(self):
        self._state.net.train()
        with amp.autocast(enabled=self.args.amp):
            loss_dict = self._state.net(
                *self._state.inputs, targets=self._state.targets)

        if (
            self._rank == 0
            and wandb.run is not None
            and getattr(self.args, 'lain_object_staged_retry_policy', False)
            and getattr(self.args, 'staged_resume_existing_run', False)
        ):
            heartbeat_now = time.monotonic()
            if heartbeat_now - self._wandb_resume_heartbeat_last >= 240.0:
                active_lrs = self._staged_group_lrs()
                wandb.log({
                    'runtime/heartbeat_unix': time.time(),
                    'runtime/logical_epoch_in_progress': int(self._state.epoch),
                    'runtime/iteration': int(self._state.iteration),
                    'runtime/retry_count': int(self._staged_retry_count),
                    'runtime/lr_head': active_lrs['head'],
                    'runtime/lr_vit': active_lrs['vit'],
                    'runtime/lr_object': active_lrs['object'],
                })
                self._wandb_resume_heartbeat_last = heartbeat_now
                print(
                    '[W&B Heartbeat] '
                    f'logical_epoch={self._state.epoch} '
                    f'iteration={self._state.iteration} '
                    f"object_lr={active_lrs['object']:.2e}"
                )

        if self._state.iteration % self._print_interval == 0:
            # 提取 interaction_loss 的值
            it_loss = loss_dict['interaction_loss'].item()
            if not np.isfinite(it_loss):
                raise ValueError(f"The HOI loss is non-finite for rank {self._rank}")
            if self._rank == 0:
                print(f"==> Iteration {self._state.iteration} | Interaction Loss: {it_loss:.4f}")
                if not getattr(
                    self.args,
                    'lain_object_staged_retry_policy',
                    False,
                ):
                    wandb.log({
                        "interaction_loss": it_loss,
                        "iteration": self._state.iteration,
                    })

        if self.args.amp:
            self._state.loss = sum(loss for loss in loss_dict.values())
            self._state.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(self._state.loss).backward()
            # GradScaler stores scaled gradients after backward(). Unscale
            # them before clipping so --clip-max-norm is applied to the real
            # gradient norm instead of the temporary AMP-scaled values.
            if self.max_norm > 0:
                self.scaler.unscale_(self._state.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._state.net.parameters(), self.max_norm
                )
            self.scaler.step(self._state.optimizer)
            self.scaler.update()
        else:
            self._state.loss = sum(loss for loss in loss_dict.values())
            self._state.optimizer.zero_grad(set_to_none=True)
            self._state.loss.backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
            self._state.optimizer.step()

    def _on_end_epoch(self):
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()
        self.net.object_class_to_target_class = self.test_loader.dataset.dataset.object_class_to_target_class

        if self.args.dataset == 'vcoco':
            # V-COCO 逻辑保持不变
            ret = self.cache_vcoco(self.test_loader)
            vsrl_annot_file = 'vcoco/data/vcoco/vcoco_test.json'
            coco_file = 'vcoco/data/instances_vcoco_all_2014.json'
            split_file = 'vcoco/data/splits/vcoco_test.ids'
            # 简单兼容处理，防止 import 报错
            try:
                import eval_vcoco
                vcocoeval = eval_vcoco.VCOCOeval(vsrl_annot_file, coco_file, split_file)
                det_file = 'vcoco_cache/cache.pkl'
                b = vcocoeval._do_eval(ret, ovr_thresh=0.5)
                mAPs = {'sc2': b[1]}
                wandb.log(mAPs)
            except ImportError:
                print("Warning: eval_vcoco not found, skipping evaluation.")
            return

        # Evaluate the actual model scores. No class suppression or unseen boost
        # is applied anywhere in this path.
        ap = self.test_hico(self.test_loader, self.args)
        scene_gate_diagnostics = self._last_scene_gate_diagnostics
        ap_gate_off = None

        real_net = self._state.net.module if hasattr(self._state.net, 'module') else self._state.net
        staged_training_metrics = None
        if getattr(self.args, 'lain_object_staged_retry_policy', False):
            staged_training_metrics = self._evaluate_staged_attempt(
                self._ap_summary(ap)
            )
            if self._staged_retry_requested:
                # The rejected weights and optimizer were already replaced by
                # the protected best and a fresh lower-LR AdamW. Do not run
                # contribution passes, save a normal checkpoint, or log W&B.
                self.net.object_class_to_target_class = (
                    self.train_loader.dataset.dataset.object_class_to_target_class
                )
                self.net.tp = None
                return
            # Gate/Text are structurally OFF. Record their conditional OFF
            # baselines as identical to Full without spending extra eval passes.
            ap_gate_off = ap

        if (
            getattr(self.args, 'scene_gate_compare_off', False)
            and getattr(real_net, 'use_scene_gate', False)
        ):
            real_net.tp = None
            real_net.use_scene_gate = False
            try:
                ap_gate_off = self.test_hico(self.test_loader, self.args, report=False)
            finally:
                real_net.use_scene_gate = True
                real_net.tp = None

        adapter_ablations = self._evaluate_adapter_ablations(self.test_loader)
        if getattr(self.args, 'lain_object_staged_retry_policy', False):
            adapter_ablations['text_adapter'] = ap
            if not getattr(real_net, 'use_obj_cond_adapter', False):
                adapter_ablations['object_conditioned_adapter'] = ap
        joint_training_policy = None
        if getattr(self.args, 'joint_best_rollback_policy', False):
            if not self.args.zs:
                raise RuntimeError(
                    'Joint best-rollback policy requires a zero-shot Unseen metric.'
                )
            current_summary = self._ap_summary(ap)
            joint_training_policy = self._apply_joint_best_rollback_policy(
                current_summary['unseen']
            )

        unseen_lr_threshold_policy = None
        if getattr(self.args, 'unseen_lr_threshold_schedule', False):
            current_summary = self._ap_summary(ap)
            unseen_lr_threshold_policy = (
                self._apply_unseen_lr_threshold_schedule(
                    current_summary['unseen']
                )
            )

        if (
            (
                getattr(self.args, 'scene_gate_diagnostics', False)
                and getattr(real_net, 'use_scene_gate', False)
            )
            or adapter_ablations
            or staged_training_metrics
        ):
            self._save_scene_gate_diagnostics(
                ap,
                ap_gate_off,
                scene_gate_diagnostics,
                adapter_ablations=adapter_ablations,
                joint_training_policy=joint_training_policy,
                staged_training_metrics=staged_training_metrics,
            )

        self.net.object_class_to_target_class = self.train_loader.dataset.dataset.object_class_to_target_class
        self.net.tp = None

        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(self.train_loader.dataset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        if self._rank == 0:
            mAPs = {'mAP': ap.mean() * 100,
                    'rare': ap[rare].mean() * 100,
                    'non-rare': ap[non_rare].mean() * 100
                    }

            print(
                f"The mAP is {ap.mean() * 100:.2f},"
                f" rare: {ap[rare].mean() * 100:.2f},"
                f" none-rare: {ap[non_rare].mean() * 100:.2f},"
            )

            if self.args.zs:
                zs_hoi_idx = hico_unseen_index[self.args.zs_type]
                print(f'>>> zero-shot setting({self.args.zs_type}!!)')
                ap_unseen = []
                ap_seen = []
                for i, value in enumerate(ap):
                    if i in zs_hoi_idx:
                        ap_unseen.append(value)
                    else:
                        ap_seen.append(value)

                ap_unseen = torch.as_tensor(ap_unseen).mean()
                ap_seen = torch.as_tensor(ap_seen).mean()

                mAPs.update({"unseen": ap_unseen * 100, "seen": ap_seen * 100})
                print(
                    f"full mAP: {ap.mean() * 100:.2f}",
                    f"unseen: {ap_unseen * 100:.2f}",
                    f"seen: {ap_seen * 100:.2f}",
                )

            if unseen_lr_threshold_policy:
                for name, lr in unseen_lr_threshold_policy['lrs_after'].items():
                    mAPs[f'lr/{name}'] = lr
                mAPs['unseen_lr_threshold/triggered_now'] = int(
                    unseen_lr_threshold_policy['triggered_now']
                )
                mAPs['unseen_lr_threshold/active'] = int(
                    unseen_lr_threshold_policy['triggered']
                )

            self.save_checkpoint(
                unseen_map=float(ap_unseen.item() * 100)
                if self.args.zs else None
            )
            if staged_training_metrics:
                current_unseen = float(ap_unseen.item() * 100)
                if current_unseen > self._staged_best_unseen:
                    self._staged_best_unseen = current_unseen
                state_payload = {
                    'last_accepted_logical_epoch': int(self._state.epoch),
                    'best_unseen': float(self._staged_best_unseen),
                    'best_checkpoint': os.path.abspath(
                        self._staged_best_checkpoint
                    ),
                    'threshold_reached': bool(
                        self._staged_threshold_reached
                    ),
                    'current_lrs': self._staged_group_lrs(),
                    'total_rollbacks': int(self._staged_total_rollbacks),
                    'last_accepted_attempt': staged_training_metrics,
                }
                state_path = os.path.join(
                    self.args.output_dir, 'staged_retry_state.json'
                )
                temporary_state = state_path + '.tmp'
                with open(temporary_state, 'w', encoding='utf-8') as handle:
                    json.dump(
                        state_payload,
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )
                os.replace(temporary_state, state_path)
                mAPs['logical_epoch'] = int(self._state.epoch)
            wandb.log(mAPs)
            if (
                joint_training_policy
                and joint_training_policy.get('early_stop', False)
            ):
                raise EarlyStopTraining(
                    'Object+Gate produced no new best Unseen mAP for '
                    f"{joint_training_policy['no_best_streak']} consecutive "
                    'epochs; protected best_unseen.pt remains unchanged.'
                )

    @torch.no_grad()
    def test_hico(
        self,
        dataloader,
        args=None,
        report=True,
        collect_gate_diagnostics=True,
    ):
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset.dataset
        unseen_ids = hico_unseen_index[args.zs_type] if args.zs else []
        seen_ids = set(range(600)) - set(unseen_ids)
        run_error_analysis = bool(
            getattr(args, 'eval', False)
            or getattr(args, 'hoi_error_analysis', False)
        )
        analyzer = (
            HOIErrorAnalyzer(unseen_ids=unseen_ids, seen_ids=seen_ids)
            if run_error_analysis else None
        )
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(
            np.asarray(dataset.object_n_verb_to_interaction, dtype=float)
        )
        real_net = net.module if hasattr(net, 'module') else net
        collect_scene_diagnostics = (
            collect_gate_diagnostics
            and
            getattr(args, 'scene_gate_diagnostics', False)
            and getattr(real_net, 'use_scene_gate', False)
            and hasattr(real_net, 'scene_gate')
            and hasattr(real_net.scene_gate, 'enable_diagnostics')
        )
        if collect_scene_diagnostics:
            real_net.scene_gate.enable_diagnostics(True)
        meter = DetectionAPMeter(
            600, nproc=1, num_gt=dataset.anno_interaction, algorithm='11P'
        )
        pred_list = []

        for batch in tqdm(dataloader, desc="Evaluating raw model output"):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            outputs = net(inputs, batch[1])
            if outputs is None or len(outputs) == 0:
                if analyzer is not None:
                    for target in batch[-1]:
                        analyzer.update(target['hoi'], None)
                continue

            for output, target in zip(outputs, batch[-1]):
                output = pocket.ops.relocate_to_cpu(output, ignore=True)
                scores = output['scores']
                pairing = output['pairing']
                objects = output['objects']
                verbs = output['labels']

                if real_net.num_classes in (117, 407):
                    interactions = conversion[objects, verbs]
                else:
                    interactions = verbs

                labels = torch.zeros_like(scores)
                matched_top_ids = [None] * len(target['hoi']) if analyzer is not None else None
                if pairing.numel() > 0 and interactions.numel() > 0:
                    boxes = output['boxes']
                    boxes_h, boxes_o = boxes[pairing].unbind(0)
                    gt_bx_h = real_net.recover_boxes(target['boxes_h'], target['size'])
                    gt_bx_o = real_net.recover_boxes(target['boxes_o'], target['size'])

                    if analyzer is not None:
                        pair_iou = torch.minimum(
                            box_ops.box_iou(gt_bx_h, boxes_h),
                            box_ops.box_iou(gt_bx_o, boxes_o),
                        )
                        for gt_idx in range(len(target['hoi'])):
                            matched_predictions = torch.nonzero(pair_iou[gt_idx] >= 0.5).squeeze(1)
                            if len(matched_predictions):
                                best_prediction = matched_predictions[scores[matched_predictions].argmax()]
                                matched_top_ids[gt_idx] = interactions[best_prediction].item()

                    for hoi_idx in interactions.unique():
                        gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                        det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                        if len(gt_idx):
                            labels[det_idx] = associate(
                                (gt_bx_h[gt_idx].view(-1, 4), gt_bx_o[gt_idx].view(-1, 4)),
                                (boxes_h[det_idx].view(-1, 4), boxes_o[det_idx].view(-1, 4)),
                                scores[det_idx].view(-1),
                            )
                if analyzer is not None:
                    analyzer.update_aligned(target['hoi'], matched_top_ids)
                pred_list.append((scores, interactions, labels))

        gathered_pred_list = []
        for rank_predictions in ddp.all_gather(pred_list):
            gathered_pred_list.extend(rank_predictions)
        for prediction in gathered_pred_list:
            meter.append(*prediction)

        merged_analyzer = None
        if analyzer is not None:
            analyzer_states = ddp.all_gather(analyzer.state_dict())
            merged_analyzer = HOIErrorAnalyzer(unseen_ids=unseen_ids, seen_ids=seen_ids)
            for state in analyzer_states:
                partial = HOIErrorAnalyzer(unseen_ids=unseen_ids, seen_ids=seen_ids)
                partial.load_state_dict(state)
                merged_analyzer.merge(partial)

        ap = meter.eval()
        if collect_scene_diagnostics:
            self._last_scene_gate_diagnostics = real_net.scene_gate.diagnostics()
            real_net.scene_gate.enable_diagnostics(False)
            real_net.scene_gate.reset_diagnostics()
        elif collect_gate_diagnostics:
            self._last_scene_gate_diagnostics = {}

        if report and self._rank == 0 and args.zs:
            print("\n" + "=" * 20 + " 原始模型结果（无校准） " + "=" * 20)
            if merged_analyzer is not None:
                merged_analyzer.report()
            unseen_ap = torch.as_tensor([ap[idx] for idx in unseen_ids]).mean()
            print(f"Unseen mAP: {unseen_ap * 100:.2f}")
            print("=" * 60)

        return ap

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        # 保持原有 cache 逻辑不变，仅做设备适配
        net = self._state.net
        net.eval()
        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(dataset.object_n_verb_to_interaction, dtype=float))
        object2int = dataset.object_to_interaction
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = [img.to(self.device) for img in batch[0]]
            output = net(inputs)
            if output is None or len(output) == 0: continue

            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            image_idx = dataset._idx[i]

            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]

            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([ow / w, oh / h, ow / w, oh / h]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num

        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        # 保持原有逻辑
        return []  # 这里简化处理，因为你主要跑 HICO
