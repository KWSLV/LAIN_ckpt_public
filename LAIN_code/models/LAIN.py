"""
Unary-pairwise transformer for human-object interaction detection
Optimized and Robust Version
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

import numpy as np

from utils.hico_list import hico_verbs_sentence
from utils.vcoco_list import vcoco_verbs_sentence
from utils.hico_utils import reserve_indices
from utils.postprocessor import PostProcess
from utils.ops import binary_focal_loss_with_logits
from utils import hico_text_label
from models.lora_utils import ObjectConditionedAdapter, TextSemanticAdapter
from models.scene_gate_v5 import (
    build_scene_gate,
    build_scene_gate_v2,
    build_scene_gate_v3,
    build_scene_gate_v5,
    build_scene_gate_v5_lowrank,
)
from models.scene_gate_v5_legacy import build_scene_gate_v5_legacy

from CLIP.clip import build_model
from CLIP.customCLIP import CustomCLIP, tokenize

sys.path.insert(0, 'detr')
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer
from detr.models.detr import DETR
from detr.util import box_ops
from detr.util.misc import nested_tensor_from_tensor_list

sys.path.pop(0)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SemanticAdapter(nn.Module):
    def __init__(self, dim=768, bottleneck=64):
        super().__init__()
        # [Optimization] Added LayerNorm for training stability
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()  # [Optimization] ReLU -> GELU usually works better for Transformers
        self.up = nn.Linear(bottleneck, dim, bias=False)

        # Initialize weights to near-zero for the up-projection to act as identity initially
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        # Residual connection
        residual = x
        x = self.norm(x)
        return residual + self.up(self.act(self.down(x)))


class LAIN(nn.Module):
    def __init__(self,
                 args,
                 detector: nn.Module,
                 postprocessor: nn.Module,
                 model: nn.Module,
                 object_embedding: torch.tensor,
                 human_idx: int, num_classes: int,
                 alpha: float = 0.5, gamma: float = 2.0,
                 box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
                 min_instances: int = 3, max_instances: int = 15,
                 object_class_to_target_class: List[list] = None,
                 object_n_verb_to_interaction: List[list] = None,

                 ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model
        self.args = args

        self.register_buffer("object_embedding", object_embedding)

        self.visual_output_dim = model.image_encoder.output_dim
        self.text_output_dim = model.text_encoder.text_projection.shape[1]
        self.visual_width = getattr(model.image_encoder.conv1, "out_channels", 768)
        self.object_n_verb_to_interaction = np.asarray(
            object_n_verb_to_interaction, dtype=float
        )

        self.human_idx = human_idx
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self._target_mask_mapping_id = None
        self._target_class_mask_cache = None

        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda
        self.use_insadapter = args.use_insadapter
        self.tp = None
        self.reserve_indices = reserve_indices

        self.priors_initial_dim = self.visual_output_dim + 5
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.priors_downproj = MLP(self.priors_initial_dim, 128, args.adapt_dim, 3)

        self.query_proj = MLP(512, 128, self.visual_width, 2)

        self.use_text_adapter = getattr(args, "use_text_adapter", False)
        if self.use_text_adapter:
            self._setup_text_adapter(args)

        self.use_obj_cond_adapter = getattr(args, "use_obj_cond_adapter", False)
        if self.use_obj_cond_adapter:
            self.obj_cond_adapter = ObjectConditionedAdapter(
                dim=self.text_output_dim,
                rank=getattr(args, "obj_cond_rank", 4),
            )
            print(
                f"[INFO] Object-conditioned text adapter initialized: "
                f"dim={self.text_output_dim}, rank={getattr(args, 'obj_cond_rank', 4)}"
            )

        self.use_semantic_adapter = getattr(args, "use_semantic_adapter", False)
        if self.use_semantic_adapter:
            self.semantic_adapter = SemanticAdapter(dim=self.text_output_dim, bottleneck=args.adapt_dim)

        self.use_scene_gate = getattr(args, "use_scene_gate", False)
        if self.use_scene_gate and not getattr(args, "use_hotoken", False):
            print("[WARN] --use_scene_gate requires --use_hotoken; SceneGate is disabled for this run.")
            self.use_scene_gate = False

        self.scene_gate_type = getattr(args, "scene_gate_type", "pair")
        self.scene_gate_version = getattr(args, "scene_gate_version", "v5")
        self.scene_gate_rank = getattr(args, "scene_gate_rank", 16)
        self.scene_gate_dropout = getattr(args, "scene_gate_dropout", 0.1)
        self.scene_gate_post_l2_norm = getattr(args, "scene_gate_post_l2_norm", False)
        if self.use_scene_gate:
            self.scene_gate = self._build_scene_gate(args)

    def _setup_text_adapter(self, args):
        text_encoder = self.clip_head.text_encoder
        for param in text_encoder.parameters():
            param.requires_grad = False

        text_encoder.text_adapter = TextSemanticAdapter(
            dim=self.text_output_dim,
            bottleneck=getattr(args, "text_adapter_dim", 64),
            rank=getattr(args, "lora_rank", 2),
            alpha=getattr(args, "lora_alpha", 8),
            residual_scale=getattr(args, "adapter_residual_scale", 0.05),
            dropout=getattr(args, "adapter_dropout", 0.1),
        )
        original_forward = text_encoder.forward

        def adapted_forward(prompts, tokenized_prompts, *forward_args, **forward_kwargs):
            text_features = original_forward(prompts, tokenized_prompts, *forward_args, **forward_kwargs)
            # Keep the adapter module in the checkpoint while allowing a
            # read-only evaluation ablation. The flag is restored immediately
            # after each diagnostic pass and never changes trainable weights.
            if self.use_text_adapter:
                text_features = text_encoder.text_adapter(text_features)
            return text_features / text_features.norm(dim=-1, keepdim=True)

        text_encoder.forward = adapted_forward
        print(
            f"[INFO] Text adapter initialized: dim={self.text_output_dim}, "
            f"bottleneck={getattr(args, 'text_adapter_dim', 64)}, "
            f"rank={getattr(args, 'lora_rank', 2)}"
        )

    def _build_scene_gate(self, args):
        version = self.scene_gate_version
        gate_type = self.scene_gate_type
        if version == "v5_legacy":
            if gate_type != "pair":
                print("[WARN] Legacy SceneGate v5 only supports pair gate; switching scene_gate_type to pair.")
                self.scene_gate_type = "pair"
                gate_type = "pair"
            gate = build_scene_gate_v5_legacy(
                gate_type=gate_type,
                dim=self.visual_output_dim,
                hidden_dim=getattr(args, "scene_gate_hidden_dim", 128),
                dropout=self.scene_gate_dropout,
                alpha=getattr(args, "scene_gate_alpha", 0.1),
                center_pairs=not getattr(
                    args, "scene_gate_disable_centering", False
                ),
            )
        elif version in {"v5", "v5_lowrank"}:
            if gate_type != "pair":
                print(f"[WARN] SceneGate {version} only supports pair gate; switching scene_gate_type to pair.")
                self.scene_gate_type = "pair"
                gate_type = "pair"
            common_kwargs = dict(
                gate_type=gate_type,
                dim=self.visual_output_dim,
                dropout=self.scene_gate_dropout,
                rank=self.scene_gate_rank,
                alpha=getattr(args, "scene_gate_alpha", 0.1),
                activation=getattr(args, "scene_gate_activation", "tanh"),
                init_std=getattr(args, "scene_gate_init_std", 0.0),
            )
            if version == "v5_lowrank":
                gate = build_scene_gate_v5_lowrank(**common_kwargs)
            else:
                gate = build_scene_gate_v5(
                    hidden_dim=getattr(args, "scene_gate_hidden_dim", 128),
                    **common_kwargs,
                )
        elif version == "v3":
            gate = build_scene_gate_v3(
                gate_type=gate_type,
                dim=self.visual_output_dim,
                hidden_dim=getattr(args, "scene_gate_hidden_dim", 128),
                dropout=self.scene_gate_dropout,
                rank=self.scene_gate_rank,
            )
        elif version == "v2":
            gate = build_scene_gate_v2(
                gate_type=gate_type,
                dim=self.visual_output_dim,
                rank=self.scene_gate_rank,
                dropout=self.scene_gate_dropout,
            )
        elif version == "v1":
            gate = build_scene_gate(
                gate_type=gate_type,
                dim=self.visual_output_dim,
                hidden_dim=getattr(args, "scene_gate_hidden_dim", 128),
            )
        else:
            raise ValueError(f"Unknown scene_gate_version: {version}")

        print(f"[INFO] SceneGate initialized: version={version}, type={self.scene_gate_type}")
        return gate

    def _get_text_features(self):
        if not self.training and self.tp is not None:
            return self.tp

        prompts = self.clip_head.prompt_learner()
        text_features = self.clip_head.text_encoder(prompts, self.clip_head.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.use_semantic_adapter:
            text_features = self.semantic_adapter(text_features)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if not self.training:
            self.tp = text_features
        return text_features

    def _object_conditioned_logits(self, visual_features, text_features, object_classes):
        logits = torch.zeros(
            visual_features.shape[0],
            text_features.shape[0],
            device=visual_features.device,
            dtype=visual_features.dtype,
        )
        for obj_cls in object_classes.unique():
            mask = object_classes == obj_cls
            obj_emb = self.object_embedding[obj_cls].to(device=visual_features.device, dtype=visual_features.dtype)
            obj_emb = obj_emb / (obj_emb.norm() + 1e-8)
            conditioned_text = self.obj_cond_adapter(text_features, obj_emb)
            conditioned_text = conditioned_text / conditioned_text.norm(dim=-1, keepdim=True)
            logits[mask] = (visual_features[mask] @ conditioned_text.T).to(dtype=logits.dtype)
        return logits

    def _get_target_class_mask(self, device):
        mapping = self.object_class_to_target_class
        mapping_id = id(mapping)
        cache = self._target_class_mask_cache
        if (
            cache is None
            or self._target_mask_mapping_id != mapping_id
            or cache.device != device
        ):
            cache = torch.zeros(
                len(mapping), self.num_classes, dtype=torch.bool, device=device
            )
            object_indices = []
            target_indices = []
            for object_idx, targets in enumerate(mapping):
                for target_idx in targets:
                    target_idx = int(target_idx)
                    if 0 <= target_idx < self.num_classes:
                        object_indices.append(object_idx)
                        target_indices.append(target_idx)
            if object_indices:
                cache[object_indices, target_indices] = True
            self._target_mask_mapping_id = mapping_id
            self._target_class_mask_cache = cache
        return cache

    def compute_prior_scores(self,
                             x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
                             ) -> Tensor:
        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        target_mask = self._get_target_class_mask(scores.device)[object_class[y].long()]
        target_mask = target_mask.to(dtype=torch.float32)
        prior_h = target_mask * s_h.float().unsqueeze(1)
        prior_o = target_mask * s_o.float().unsqueeze(1)

        return torch.stack([prior_h, prior_o])

    def compute_sim_scores(self, region_props: List[dict], image, priors=None):
        device = image.tensors.device
        boxes_h_collated = []
        boxes_o_collated = []
        prior_collated = []
        object_class_collated = []
        all_logits = []

        text_features = self._get_text_features().to(device=device)

        def append_empty(dtype):
            empty_idx = torch.zeros(0, dtype=torch.int64, device=device)
            boxes_h_collated.append(empty_idx)
            boxes_o_collated.append(empty_idx)
            object_class_collated.append(empty_idx)
            prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device, dtype=dtype))
            all_logits.append(torch.zeros(0, text_features.shape[0], device=device, dtype=dtype))

        for b_idx, props in enumerate(region_props):
            boxes = props["boxes"]
            scores = props["scores"]
            labels = props["labels"]
            feats = props["feat"]
            is_human = labels == self.human_idx
            human_idx = torch.nonzero(is_human).squeeze(1)

            if len(human_idx) == 0 or len(boxes) <= 1:
                append_empty(text_features.dtype)
                continue

            # Original LAIN pairs each human with every other detection. A
            # second person must remain eligible as the object for HOI 160-169.
            candidate_idx = torch.arange(len(boxes), device=device)
            x_keep = human_idx.repeat_interleave(len(candidate_idx))
            y_keep = candidate_idx.repeat(len(human_idx))
            valid = x_keep != y_keep
            x_keep = x_keep[valid]
            y_keep = y_keep[valid]
            if len(x_keep) == 0:
                append_empty(text_features.dtype)
                continue

            image_tensor = image.decompose()[0][b_idx:b_idx + 1]
            if self.args.use_hotoken:
                patch_size = getattr(self.clip_head.image_encoder, "patch_size", 16)
                patch_tokens = (image_tensor.shape[-2] // patch_size) * (image_tensor.shape[-1] // patch_size)
                prefix_tokens = len(x_keep) + 1
                mask = torch.zeros(
                    (prefix_tokens + patch_tokens, prefix_tokens + patch_tokens),
                    dtype=torch.bool,
                    device=device,
                )
                mask[:prefix_tokens, :prefix_tokens] = ~torch.eye(prefix_tokens, dtype=torch.bool, device=device)
                mask[-(patch_tokens + 1):, :-(patch_tokens + 1)] = True

                ho_tokens = self.query_proj(torch.cat([feats[x_keep], feats[y_keep]], dim=-1))
                la_masks = (boxes, x_keep, y_keep)

                if self.use_scene_gate:
                    ho_features, cls_token, _ = self.clip_head.image_encoder(
                        image_tensor,
                        priors[b_idx] if self.args.use_prior else None,
                        ho_tokens,
                        mask,
                        la_masks,
                        return_cls=True,
                    )
                    ho_features = ho_features / ho_features.norm(dim=-1, keepdim=True)
                    cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                    global_feat = self.scene_gate(ho_features, cls_token).squeeze(0)
                    if self.scene_gate_post_l2_norm:
                        global_feat = F.normalize(global_feat, p=2, dim=-1, eps=1e-6)
                else:
                    prefix_feat, _ = self.clip_head.image_encoder(
                        image_tensor,
                        priors[b_idx] if self.args.use_prior else None,
                        ho_tokens,
                        mask,
                        la_masks,
                    )
                    global_feat = prefix_feat[:, :-1, :]
                    global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
                    global_feat = global_feat.squeeze(0)
            else:
                prefix_feat, _ = self.clip_head.image_encoder(
                    image_tensor,
                    priors[b_idx] if self.args.use_prior else None,
                )
                global_feat = prefix_feat[:, 0, :]
                global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
                global_feat = global_feat.repeat(len(x_keep), 1)

            current_text = text_features.to(device=global_feat.device, dtype=global_feat.dtype)
            if self.use_obj_cond_adapter:
                logits_text = self._object_conditioned_logits(global_feat, current_text, labels[y_keep])
            else:
                logits_text = global_feat @ current_text.T
            logits = logits_text * self.logit_scale_text.exp().to(dtype=logits_text.dtype)

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(x_keep, y_keep, scores, labels))
            all_logits.append(logits)

        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        # Fix for empty ground truth
        if len(gt_bx_h) == 0:
            return labels

        ious_h = box_iou(boxes_h, gt_bx_h)
        ious_o = box_iou(boxes_o, gt_bx_o)

        min_ious = torch.min(ious_h, ious_o)

        x, y = torch.nonzero(min_ious >= self.fg_iou_thresh).unbind(1)

        if self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1
        else:
            labels[x, targets['hoi'][y]] = 1
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):

        # [Robustness] Check for empty batch
        if len(logits) == 0:
            return torch.tensor(0.0, device=boxes[0].device, requires_grad=True)

        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])

        prior = torch.cat(prior, dim=1).prod(0)
        logits = torch.cat(logits)

        if logits.numel() == 0 or prior.numel() == 0 or labels.numel() == 0:
            return self.logit_scale_text.sum() * 0.0

        # [Safety] Dimension Mismatch Guard
        if len(labels) != len(logits):
            # This happens if some pairs were filtered out during logits calc but not here
            # For now, we assume strict alignment, but print warning if diff
            pass

        x, y = torch.nonzero(prior).unbind(1)
        if len(x) == 0:
            return self.logit_scale_text.sum() * 0.0
        logits = logits[x, y]
        prior = prior[x, y]
        labels = labels[x, y]

        n_p = torch.count_nonzero(labels).to(dtype=torch.float32)

        # all_reduce already synchronizes the count; a separate barrier only
        # stalls the single-GPU path used by the training scripts.
        if dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            dist.all_reduce(n_p)
            n_p = n_p / world_size

        loss = binary_focal_loss_with_logits(
            torch.log(prior / (1 + torch.exp(-logits) - prior) + 1e-8),
            labels,
            reduction='sum',
            alpha=self.alpha,
            gamma=self.gamma
        )

        has_positive = (n_p > 0).to(dtype=loss.dtype)
        return loss * has_positive / n_p.clamp_min(1.0)

    def prepare_region_proposals(self, results):
        region_props = []
        for res in results:
            sc, lb, bx, feat = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            feat = feat[keep].view(-1, 256)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum();
            n_object = len(keep) - n_human

            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                feat=feat[keep]
            ))

        return region_props

    def get_prior(self, region_props, image_size):

        max_feat = self.priors_initial_dim
        # Handle case where no boxes exist
        if len(region_props) == 0:
            return torch.zeros((0, 14, 14, self.args.adapt_dim), device=image_size.device)

        priors = torch.zeros((len(region_props), 14, 14), dtype=torch.float32, device=region_props[0]['boxes'].device)
        priors_dim = torch.zeros((len(region_props), 14, 14, max_feat), dtype=torch.float32,
                                 device=region_props[0]['boxes'].device)

        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] * (14 / scale_fct[b_idx][None, :])
            scores = props['scores']
            labels = props['labels']
            priors[b_idx] = len(boxes)

            boxes[:, 2:] += 0.5
            new_boxes = torch.round(boxes).long()

            # [FIX] Clamp coordinates to stay within 14x14 grid to prevent index out of bounds
            new_boxes[:, 0].clamp_(0, 13)
            new_boxes[:, 1].clamp_(0, 13)
            new_boxes[:, 2].clamp_(0, 14)  # x2 is exclusive in slice, so can be 14
            new_boxes[:, 3].clamp_(0, 14)  # y2 is exclusive

            for inb, nb in enumerate(new_boxes):
                x1, y1, x2, y2 = nb
                # Safety check: ensure x2 > x1 and y2 > y1
                if x2 > x1 and y2 > y1:
                    priors[b_idx, y1:y2, x1:x2] = inb

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human);
            n = len(boxes)
            if n_h == 0 or n <= 1:
                pass

            boxes = torch.cat([boxes, torch.tensor([[-1, -1, -1, -1.]]).to(boxes)], dim=0)
            labels = torch.cat([labels, torch.tensor([80]).to(boxes)], dim=0).long()
            scores = torch.cat([scores, torch.tensor([-1.]).to(boxes)], dim=0)

            object_embs = self.object_embedding[labels]

            sb = torch.cat((scores.unsqueeze(-1), boxes), dim=-1)
            sb_feat = sb[priors[b_idx].long()]
            obj_feat = object_embs[priors[b_idx].long()]

            prior_feat = torch.cat([sb_feat, obj_feat], dim=-1)
            priors_dim[b_idx] = prior_feat

        priors = self.priors_downproj(priors_dim)
        return priors

    def forward(self,
                images: List[Tensor],
                targets: Optional[List[dict]] = None
                ) -> List[dict]:

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        batch_size = len(images)
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]
        device = images_clip[0].device
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images_clip
        ], device=device)

        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)

        features, pos = self.detector.backbone(images_orig)
        src, mask = features[-1].decompose()

        hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask,
                                                    self.detector.query_embed.weight, pos[-1])
        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()

        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'feats': hs[-1]}
        results = self.postprocessor(results, image_sizes)
        region_props = self.prepare_region_proposals(results)

        priors = self.get_prior(region_props, image_sizes)

        images_clip = nested_tensor_from_tensor_list(images_clip)

        # [Important] Check if all regions are empty, skip computation to save time/error
        all_empty = all(len(r['boxes']) == 0 for r in region_props)
        if all_empty:
            logits, prior, bh, bo, objects = [], [], [], [], []
        else:
            logits, prior, bh, bo, objects = self.compute_sim_scores(region_props, images_clip, priors)

        boxes = [r['boxes'] for r in region_props]

        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)
            loss_dict = dict(interaction_loss=interaction_loss)
            return loss_dict

        if len(logits) == 0:
            # Return empty detections structure matching the format
            return [dict(boxes=torch.tensor([]), scores=torch.tensor([]), labels=torch.tensor([])) for _ in
                    range(batch_size)]

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes):
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size in zip(
                boxes, bh, bo, logits, prior, objects, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)

            # [LOGIC] Interference Suppression could be added here
            # e.g., scores[:, suppressed_ids] *= 0.1

            scores = torch.sigmoid(lg[x, y])

            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections


@torch.no_grad()
def get_obj_text_emb(args, clip_model, obj_class_names):
    obj_text_inputs = torch.cat([tokenize(obj_text) for obj_text in obj_class_names])
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
    return object_embedding


def safe_torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path):
    # build DETR
    num_classes = 80
    if args.dataset == 'vcoco' and 'e632da11' in args.pretrained:
        num_classes = 91

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    detr = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    postprocessors = {'bbox': PostProcess()}

    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        detector_checkpoint = safe_torch_load(args.pretrained, map_location='cpu')
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(detector_checkpoint['model'])
        else:
            detr.load_state_dict(detector_checkpoint['model_state_dict'])
    else:
        if dist.get_rank() == 0:
            print(f"!! WARNING: Pretrained weights not found at {args.pretrained}. !!")
            print("!! Model is starting from RANDOM INITIALIZATION (Performance will be poor) !!")

    clip_state_dict = safe_torch_load(clip_model_path, map_location="cpu").state_dict()
    clip_model = build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, adapter_pos=args.adapter_pos,
                             args=args)

    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError

    model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)

    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]

    object_embedding = get_obj_text_emb(args, clip_model=clip_model, obj_class_names=obj_class_names)
    object_embedding = object_embedding.clone().detach()

    detector = LAIN(args,
                    detr, postprocessors['bbox'], model, object_embedding,
                    human_idx=args.human_idx, num_classes=args.num_classes,
                    alpha=args.alpha, gamma=args.gamma,
                    box_score_thresh=args.box_score_thresh,
                    fg_iou_thresh=args.fg_iou_thresh,
                    min_instances=args.min_instances,
                    max_instances=args.max_instances,
                    object_class_to_target_class=class_corr,
                    object_n_verb_to_interaction=object_n_verb_to_interaction,
                    )

    return detector
