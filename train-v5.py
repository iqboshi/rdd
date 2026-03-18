#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_train_val_datasets_by_big_image
from model import LeafInstanceSegModel
from transforms_v2 import get_train_transform, get_val_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train leaf-only direct instance segmentation model.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["data/patches_size512", "data/patches_size768", "data/patches_size1024"],
        help="One or more patch roots.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--num_queries", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--pretrained_backbone_path",
        type=str,
        default="pretrain_riceseg\outputs\exp_20260307_124458\checkpoints\\best_backbone_for_instance.pth",
        help="Optional pretrained backbone checkpoint path (e.g. best_backbone_for_instance.pth).",
    )
    parser.add_argument(
        "--pretrained_backbone_strict",
        action="store_true",
        default=False,
        help="Use strict=True when loading backbone weights.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")

    parser.add_argument("--w_cls", type=float, default=0.5, help="Weight for classification loss.")
    parser.add_argument("--w_mask", type=float, default=2.0, help="Weight for mask BCE loss.")
    parser.add_argument("--w_dice", type=float, default=2.0, help="Weight for dice loss.")
    parser.add_argument("--enable_aux_heads", action="store_true", help="Enable center+offset auxiliary heads (A-v1).")
    parser.add_argument("--w_center", type=float, default=0.30, help="Weight for center heatmap BCE aux loss.")
    parser.add_argument("--w_offset", type=float, default=0.05, help="Weight for offset L1 aux loss.")
    parser.add_argument(
        "--w_vote_consistency",
        type=float,
        default=0.02,
        help="Weight for instance center-vote consistency loss (A-v5).",
    )
    parser.add_argument(
        "--w_separation",
        type=float,
        default=0.10,
        help="Weight for boundary separation supervision loss (A-v6).",
    )
    parser.add_argument(
        "--w_repulsion",
        type=float,
        default=0.02,
        help="Weight for cross-instance vote repulsion loss on touching boundaries (A-v6).",
    )
    parser.add_argument(
        "--w_conflict",
        type=float,
        default=0.0,
        help="Weight for boundary conflict supervision loss (A7).",
    )
    parser.add_argument(
        "--w_affinity",
        type=float,
        default=0.0,
        help="Weight for pixel affinity embedding discriminative loss (A7).",
    )
    parser.add_argument(
        "--w_overlap_excl",
        type=float,
        default=0.0,
        help="Weight for mutual-exclusion overlap penalty on matched instance masks (A7).",
    )
    parser.add_argument(
        "--vote_consistency_min_pixels",
        type=int,
        default=20,
        help="Minimum pixels for an instance to participate in vote consistency loss.",
    )
    parser.add_argument(
        "--vote_consistency_touch_boost",
        type=float,
        default=1.6,
        help="Extra loss weight for instances touching patch borders (occlusion/truncation-prone).",
    )
    parser.add_argument("--center_sigma", type=float, default=4.0, help="Sigma for center heatmap targets.")
    parser.add_argument("--offset_clip", type=float, default=64.0, help="Clip absolute target offset values.")
    parser.add_argument(
        "--separation_dilate",
        type=int,
        default=1,
        help="Dilate radius (px) for GT separation target map.",
    )
    parser.add_argument(
        "--separation_pos_weight",
        type=float,
        default=3.0,
        help="Positive class weight for separation BCE.",
    )
    parser.add_argument(
        "--separation_dice_weight",
        type=float,
        default=1.0,
        help="Dice term multiplier inside separation loss.",
    )
    parser.add_argument(
        "--repulsion_margin",
        type=float,
        default=6.0,
        help="Minimum center-vote distance between touching different instances.",
    )
    parser.add_argument(
        "--repulsion_max_pairs",
        type=int,
        default=6000,
        help="Maximum sampled touching pixel pairs per batch element for repulsion loss.",
    )
    parser.add_argument(
        "--conflict_dilate",
        type=int,
        default=2,
        help="Dilate radius (px) for conflict target band.",
    )
    parser.add_argument(
        "--conflict_pos_weight",
        type=float,
        default=2.5,
        help="Positive class weight for conflict BCE.",
    )
    parser.add_argument(
        "--conflict_dice_weight",
        type=float,
        default=1.0,
        help="Dice term multiplier inside conflict loss.",
    )
    parser.add_argument(
        "--affinity_dim",
        type=int,
        default=16,
        help="Channel dimension for affinity embedding head.",
    )
    parser.add_argument(
        "--affinity_min_pixels",
        type=int,
        default=24,
        help="Minimum pixels per instance for affinity supervision.",
    )
    parser.add_argument(
        "--affinity_margin_var",
        type=float,
        default=0.5,
        help="Intra-instance embedding compactness margin.",
    )
    parser.add_argument(
        "--affinity_margin_dist",
        type=float,
        default=1.5,
        help="Inter-instance embedding separation margin.",
    )
    parser.add_argument(
        "--affinity_dist_weight",
        type=float,
        default=1.0,
        help="Weight for inter-instance term inside affinity loss.",
    )
    parser.add_argument(
        "--affinity_reg_weight",
        type=float,
        default=0.001,
        help="Weight for embedding norm regularization inside affinity loss.",
    )
    parser.add_argument(
        "--affinity_max_instances",
        type=int,
        default=64,
        help="Maximum number of instances sampled per image for affinity loss.",
    )
    parser.add_argument(
        "--overlap_bg_margin",
        type=float,
        default=0.15,
        help="Allowed summed mask probability margin on background pixels.",
    )
    parser.add_argument(
        "--overlap_bg_weight",
        type=float,
        default=0.2,
        help="Background term weight inside mutual-exclusion overlap loss.",
    )
    parser.add_argument("--match_cls", type=float, default=1.0)
    parser.add_argument("--match_mask", type=float, default=1.0)
    parser.add_argument("--match_dice", type=float, default=1.0)
    parser.add_argument("--match_size", type=int, default=128)

    parser.add_argument("--vis_every", type=int, default=1, help="Save train/val visualization every N epochs.")
    parser.add_argument("--max_train_steps", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max_val_steps", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--num_workers_pin_memory", action="store_true", default=True)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--enable_patch_scale_weighting",
        action="store_true",
        help="Use weighted sampling to increase exposure of larger patch scales during training.",
    )
    parser.add_argument("--patch_scale_weight_512", type=float, default=1.0, help="Sampling weight for 512 patches.")
    parser.add_argument("--patch_scale_weight_768", type=float, default=1.5, help="Sampling weight for 768 patches.")
    parser.add_argument("--patch_scale_weight_1024", type=float, default=2.5, help="Sampling weight for 1024 patches.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested (`{device_arg}`) but CUDA is not available. "
                "Please check your CUDA/PyTorch installation or pass --device cpu explicitly."
            )
        return torch.device(device_arg)
    return torch.device("cpu")


def leaf_collate_fn(batch: List[Dict]):
    images = torch.stack([b["image"] for b in batch], dim=0)
    semantic_masks = torch.stack([b["semantic_mask"] for b in batch], dim=0)
    instance_maps = torch.stack([b["instance_map"] for b in batch], dim=0)
    gt_masks = [b["gt_masks"] for b in batch]
    gt_labels = [b["gt_labels"] for b in batch]
    metas = [b["meta"] for b in batch]
    return {
        "image": images,
        "semantic_mask": semantic_masks,
        "instance_map": instance_maps,
        "gt_masks": gt_masks,
        "gt_labels": gt_labels,
        "meta": metas,
    }


def dice_loss_from_logits(pred_logits: torch.Tensor, target_masks: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    pred_probs = pred_logits.sigmoid()
    pred_flat = pred_probs.flatten(1)
    target_flat = target_masks.flatten(1)
    numerator = 2.0 * (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    loss = 1.0 - (numerator + eps) / (denominator + eps)
    return loss.mean()


class HungarianMatcher:
    def __init__(self, cls_weight: float = 1.0, mask_weight: float = 1.0, dice_weight: float = 1.0, match_size: int = 128):
        self.cls_weight = float(cls_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)
        self.match_size = int(match_size)

    @torch.no_grad()
    def __call__(self, pred_logits: torch.Tensor, pred_masks: torch.Tensor, targets: List[Dict[str, torch.Tensor]]):
        device = pred_logits.device
        bsz, num_queries, _ = pred_logits.shape
        out_prob = pred_logits.softmax(dim=-1)
        matches = []

        for b in range(bsz):
            tgt_labels = targets[b]["labels"]
            tgt_masks = targets[b]["masks"]
            num_gt = int(tgt_labels.numel())

            if num_gt == 0:
                empty = torch.zeros((0,), dtype=torch.long, device=device)
                matches.append((empty, empty))
                continue

            cost_class = -out_prob[b][:, tgt_labels]  # [Q, N]

            pred_m = pred_masks[b]  # [Q, H, W]
            tgt_m = tgt_masks.float()  # [N, H, W]

            if self.match_size > 0:
                pred_m = F.interpolate(
                    pred_m.unsqueeze(1),
                    size=(self.match_size, self.match_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                tgt_m = F.interpolate(
                    tgt_m.unsqueeze(1),
                    size=(self.match_size, self.match_size),
                    mode="nearest",
                ).squeeze(1)

            qn_pred = pred_m.unsqueeze(1).expand(-1, num_gt, -1, -1)
            qn_tgt = tgt_m.unsqueeze(0).expand(num_queries, -1, -1, -1)

            cost_mask = F.binary_cross_entropy_with_logits(qn_pred, qn_tgt, reduction="none").mean(dim=(2, 3))

            pred_prob = qn_pred.sigmoid()
            numerator = 2.0 * (pred_prob * qn_tgt).sum(dim=(2, 3))
            denominator = pred_prob.sum(dim=(2, 3)) + qn_tgt.sum(dim=(2, 3))
            cost_dice = 1.0 - (numerator + 1.0) / (denominator + 1.0)

            total_cost = (
                self.cls_weight * cost_class
                + self.mask_weight * cost_mask
                + self.dice_weight * cost_dice
            )

            row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
            row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=device)
            col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=device)
            matches.append((row_ind, col_ind))

        return matches


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    matcher: HungarianMatcher,
    no_object_class: int = 1,
    w_cls: float = 1.0,
    w_mask: float = 1.0,
    w_dice: float = 1.0,
):
    pred_logits = outputs["pred_logits"]  # [B, Q, C]
    pred_masks = outputs["pred_masks"]  # [B, Q, H, W]
    device = pred_logits.device
    bsz, num_queries, _ = pred_logits.shape

    matches = matcher(pred_logits, pred_masks, targets)

    cls_loss_sum = torch.tensor(0.0, device=device)
    mask_loss_sum = torch.tensor(0.0, device=device)
    dice_loss_sum = torch.tensor(0.0, device=device)
    matched_count = 0

    for b in range(bsz):
        row_ind, col_ind = matches[b]
        target_classes = torch.full(
            (num_queries,),
            fill_value=int(no_object_class),
            dtype=torch.long,
            device=device,
        )
        if col_ind.numel() > 0:
            target_classes[row_ind] = targets[b]["labels"][col_ind]

        cls_loss_sum = cls_loss_sum + F.cross_entropy(pred_logits[b], target_classes)

        if col_ind.numel() > 0:
            pred_m = pred_masks[b, row_ind]
            tgt_m = targets[b]["masks"][col_ind].float()
            bce = F.binary_cross_entropy_with_logits(pred_m, tgt_m, reduction="mean")
            dice = dice_loss_from_logits(pred_m, tgt_m)

            n_match = int(col_ind.numel())
            mask_loss_sum = mask_loss_sum + bce * n_match
            dice_loss_sum = dice_loss_sum + dice * n_match
            matched_count += n_match

    cls_loss = cls_loss_sum / max(bsz, 1)
    if matched_count > 0:
        mask_loss = mask_loss_sum / matched_count
        dice_loss = dice_loss_sum / matched_count
    else:
        mask_loss = torch.tensor(0.0, device=device)
        dice_loss = torch.tensor(0.0, device=device)

    total_loss = w_cls * cls_loss + w_mask * mask_loss + w_dice * dice_loss
    return {
        "loss": total_loss,
        "loss_cls": cls_loss,
        "loss_mask": mask_loss,
        "loss_dice": dice_loss,
        "matches": matches,
    }


def prepare_targets(batch: Dict, device: torch.device):
    targets = []
    for masks, labels in zip(batch["gt_masks"], batch["gt_labels"]):
        targets.append(
            {
                "masks": masks.to(device, non_blocking=True).float(),
                "labels": labels.to(device, non_blocking=True).long(),
            }
        )
    return targets


def build_center_offset_targets_for_batch(
    instance_maps: torch.Tensor,
    device: torch.device,
    center_sigma: float = 4.0,
    offset_clip: float = 64.0,
) -> Dict[str, torch.Tensor]:
    inst = instance_maps.to(device, non_blocking=True).long()
    bsz, h, w = inst.shape
    center_targets = torch.zeros((bsz, h, w), dtype=torch.float32, device=device)
    offset_targets = torch.zeros((bsz, 2, h, w), dtype=torch.float32, device=device)
    fg_mask = (inst > 0).float()

    sigma = float(center_sigma)
    radius = max(1, int(round(3.0 * sigma))) if sigma > 0 else 0

    for b in range(bsz):
        ids = torch.unique(inst[b])
        ids = ids[ids > 0]
        for ins_id in ids.tolist():
            ys, xs = torch.where(inst[b] == int(ins_id))
            if ys.numel() == 0:
                continue

            ys_f = ys.float()
            xs_f = xs.float()
            cy = ys_f.mean()
            cx = xs_f.mean()

            offset_targets[b, 0, ys, xs] = cy - ys_f
            offset_targets[b, 1, ys, xs] = cx - xs_f

            if sigma <= 0:
                x = int(round(float(cx)))
                y = int(round(float(cy)))
                if 0 <= x < w and 0 <= y < h:
                    center_targets[b, y, x] = torch.maximum(
                        center_targets[b, y, x], torch.tensor(1.0, device=device)
                    )
                continue

            x0 = max(0, int(torch.floor(cx).item()) - radius)
            y0 = max(0, int(torch.floor(cy).item()) - radius)
            x1 = min(w, int(torch.floor(cx).item()) + radius + 1)
            y1 = min(h, int(torch.floor(cy).item()) + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue

            grid_x = torch.arange(x0, x1, device=device, dtype=torch.float32)[None, :]
            grid_y = torch.arange(y0, y1, device=device, dtype=torch.float32)[:, None]
            g = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2.0 * sigma * sigma))
            center_targets[b, y0:y1, x0:x1] = torch.maximum(center_targets[b, y0:y1, x0:x1], g)

    if offset_clip > 0:
        offset_targets = torch.clamp(offset_targets, min=-float(offset_clip), max=float(offset_clip))

    return {
        "center": center_targets,
        "offset": offset_targets,
        "fg_mask": fg_mask,
    }


def build_separation_targets_for_batch(
    instance_maps: torch.Tensor,
    device: torch.device,
    dilate_radius: int = 1,
) -> torch.Tensor:
    """
    Build a binary target map for boundaries between different leaf instances.
    Only leaf-vs-leaf touching boundaries are treated as positives.
    """
    inst = instance_maps.to(device, non_blocking=True).long()
    bsz, h, w = inst.shape
    boundary = torch.zeros((bsz, h, w), dtype=torch.bool, device=device)

    if h > 1:
        top = inst[:, :-1, :]
        bottom = inst[:, 1:, :]
        m = (top > 0) & (bottom > 0) & (top != bottom)
        boundary[:, :-1, :] |= m
        boundary[:, 1:, :] |= m

    if w > 1:
        left = inst[:, :, :-1]
        right = inst[:, :, 1:]
        m = (left > 0) & (right > 0) & (left != right)
        boundary[:, :, :-1] |= m
        boundary[:, :, 1:] |= m

    r = max(0, int(dilate_radius))
    if r > 0:
        k = 2 * r + 1
        boundary_f = boundary.float().unsqueeze(1)
        boundary_f = F.max_pool2d(boundary_f, kernel_size=k, stride=1, padding=r)
        boundary = boundary_f.squeeze(1) > 0.5

    return boundary.float()


def build_conflict_targets_for_batch(
    instance_maps: torch.Tensor,
    device: torch.device,
    dilate_radius: int = 2,
) -> torch.Tensor:
    """
    Build ambiguity/conflict bands around boundaries where labels change.
    This includes leaf-vs-leaf and leaf-vs-background boundaries.
    """
    inst = instance_maps.to(device, non_blocking=True).long()
    bsz, h, w = inst.shape
    boundary = torch.zeros((bsz, h, w), dtype=torch.bool, device=device)

    if h > 1:
        top = inst[:, :-1, :]
        bottom = inst[:, 1:, :]
        m = (top != bottom) & ((top > 0) | (bottom > 0))
        boundary[:, :-1, :] |= m
        boundary[:, 1:, :] |= m

    if w > 1:
        left = inst[:, :, :-1]
        right = inst[:, :, 1:]
        m = (left != right) & ((left > 0) | (right > 0))
        boundary[:, :, :-1] |= m
        boundary[:, :, 1:] |= m

    r = max(0, int(dilate_radius))
    if r > 0:
        k = 2 * r + 1
        boundary_f = boundary.float().unsqueeze(1)
        boundary_f = F.max_pool2d(boundary_f, kernel_size=k, stride=1, padding=r)
        boundary = boundary_f.squeeze(1) > 0.5

    return boundary.float()


def compute_boundary_repulsion_loss(
    pred_offset: torch.Tensor,
    instance_maps: torch.Tensor,
    margin: float = 6.0,
    max_pairs: int = 6000,
) -> torch.Tensor:
    """
    For adjacent pixels that belong to different GT instances, encourage their
    offset-voted centers to stay separated by at least `margin` pixels.
    """
    if pred_offset.ndim != 4 or pred_offset.shape[1] != 2:
        raise ValueError(f"pred_offset should be [B,2,H,W], got {tuple(pred_offset.shape)}")

    inst = instance_maps.to(pred_offset.device, non_blocking=True).long()
    bsz, _, h, w = pred_offset.shape
    margin = float(max(0.0, margin))
    max_pairs = int(max(1, max_pairs))

    total_loss = pred_offset.new_tensor(0.0)
    valid_batches = 0

    for b in range(bsz):
        y0_parts = []
        x0_parts = []
        y1_parts = []
        x1_parts = []

        if h > 1:
            top = inst[b, :-1, :]
            bottom = inst[b, 1:, :]
            m = (top > 0) & (bottom > 0) & (top != bottom)
            yy, xx = torch.where(m)
            if yy.numel() > 0:
                y0_parts.append(yy)
                x0_parts.append(xx)
                y1_parts.append(yy + 1)
                x1_parts.append(xx)

        if w > 1:
            left = inst[b, :, :-1]
            right = inst[b, :, 1:]
            m = (left > 0) & (right > 0) & (left != right)
            yy, xx = torch.where(m)
            if yy.numel() > 0:
                y0_parts.append(yy)
                x0_parts.append(xx)
                y1_parts.append(yy)
                x1_parts.append(xx + 1)

        if len(y0_parts) == 0:
            continue

        y0 = torch.cat(y0_parts, dim=0)
        x0 = torch.cat(x0_parts, dim=0)
        y1 = torch.cat(y1_parts, dim=0)
        x1 = torch.cat(x1_parts, dim=0)

        n_pairs = int(y0.numel())
        if n_pairs > max_pairs:
            perm = torch.randperm(n_pairs, device=pred_offset.device)[:max_pairs]
            y0 = y0[perm]
            x0 = x0[perm]
            y1 = y1[perm]
            x1 = x1[perm]

        dy0 = pred_offset[b, 0, y0, x0]
        dx0 = pred_offset[b, 1, y0, x0]
        dy1 = pred_offset[b, 0, y1, x1]
        dx1 = pred_offset[b, 1, y1, x1]

        vote_y0 = y0.float() + dy0
        vote_x0 = x0.float() + dx0
        vote_y1 = y1.float() + dy1
        vote_x1 = x1.float() + dx1
        dist = torch.sqrt((vote_y0 - vote_y1) ** 2 + (vote_x0 - vote_x1) ** 2 + 1e-6)
        loss_b = F.relu(margin - dist).mean()

        total_loss = total_loss + loss_b
        valid_batches += 1

    if valid_batches == 0:
        return pred_offset.new_tensor(0.0)
    return total_loss / float(valid_batches)


def compute_affinity_embedding_loss(
    pred_affinity: torch.Tensor,
    instance_maps: torch.Tensor,
    min_pixels: int = 24,
    margin_var: float = 0.5,
    margin_dist: float = 1.5,
    dist_weight: float = 1.0,
    reg_weight: float = 0.001,
    max_instances: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Discriminative embedding loss:
    - pull pixels of the same instance together (variance term)
    - push different instance centers apart (distance term)
    """
    if pred_affinity.ndim != 4:
        raise ValueError(f"pred_affinity should be [B,D,H,W], got {tuple(pred_affinity.shape)}")

    inst = instance_maps.to(pred_affinity.device, non_blocking=True).long()
    bsz, _, _, _ = pred_affinity.shape
    min_pixels = max(1, int(min_pixels))
    max_instances = max(1, int(max_instances))

    var_sum = pred_affinity.new_tensor(0.0)
    dist_sum = pred_affinity.new_tensor(0.0)
    reg_sum = pred_affinity.new_tensor(0.0)
    valid_batches = 0
    dist_batches = 0

    for b in range(bsz):
        emb = pred_affinity[b].permute(1, 2, 0).contiguous()  # [H, W, D]
        ids = torch.unique(inst[b])
        ids = ids[ids > 0]
        if ids.numel() == 0:
            continue
        if ids.numel() > max_instances:
            perm = torch.randperm(ids.numel(), device=ids.device)[:max_instances]
            ids = ids[perm]

        means = []
        for ins_id in ids.tolist():
            ys, xs = torch.where(inst[b] == int(ins_id))
            if ys.numel() < min_pixels:
                continue
            vecs = emb[ys, xs]  # [N, D]
            mu = vecs.mean(dim=0)
            means.append(mu)

            intra = torch.norm(vecs - mu.unsqueeze(0), dim=1)
            var_term = F.relu(intra - float(margin_var)) ** 2
            var_sum = var_sum + var_term.mean()

        if len(means) == 0:
            continue

        centers = torch.stack(means, dim=0)  # [K, D]
        reg_sum = reg_sum + centers.norm(dim=1).mean()
        valid_batches += 1

        if centers.shape[0] > 1:
            dist_mat = torch.cdist(centers, centers, p=2)
            eye = torch.eye(dist_mat.shape[0], dtype=torch.bool, device=dist_mat.device)
            off_diag = dist_mat[~eye]
            if off_diag.numel() > 0:
                dist_term = F.relu(float(margin_dist) - off_diag) ** 2
                dist_sum = dist_sum + dist_term.mean()
                dist_batches += 1

    if valid_batches == 0:
        zero = pred_affinity.new_tensor(0.0)
        return {"total": zero, "var": zero, "dist": zero, "reg": zero}

    var = var_sum / float(valid_batches)
    reg = reg_sum / float(valid_batches)
    if dist_batches > 0:
        dist = dist_sum / float(dist_batches)
    else:
        dist = pred_affinity.new_tensor(0.0)

    total = var + float(dist_weight) * dist + float(reg_weight) * reg
    return {"total": total, "var": var, "dist": dist, "reg": reg}


def compute_mutual_exclusion_overlap_loss(
    pred_masks: torch.Tensor,
    matches: List[Tuple[torch.Tensor, torch.Tensor]],
    instance_maps: torch.Tensor,
    bg_margin: float = 0.15,
    bg_weight: float = 0.2,
) -> torch.Tensor:
    """
    Penalize excessive overlap among matched instance masks.
    This directly targets mixed-color/impure instance regions.
    """
    if pred_masks.ndim != 4:
        raise ValueError(f"pred_masks should be [B,Q,H,W], got {tuple(pred_masks.shape)}")

    inst = instance_maps.to(pred_masks.device, non_blocking=True).long()
    bsz, _, _, _ = pred_masks.shape
    total = pred_masks.new_tensor(0.0)
    valid = 0

    for b in range(bsz):
        row_ind, _ = matches[b]
        if row_ind.numel() == 0:
            continue

        probs = pred_masks[b, row_ind].sigmoid()  # [N, H, W]
        sum_probs = probs.sum(dim=0)
        leaf = (inst[b] > 0).float()
        bg = 1.0 - leaf

        leaf_denom = leaf.sum().clamp(min=1.0)
        loss_leaf = (F.relu(sum_probs - 1.0) * leaf).sum() / leaf_denom

        loss_bg = pred_masks.new_tensor(0.0)
        bg_denom = bg.sum()
        if float(bg_denom.item()) > 0:
            loss_bg = (F.relu(sum_probs - float(bg_margin)) * bg).sum() / bg_denom

        total = total + loss_leaf + float(bg_weight) * loss_bg
        valid += 1

    if valid == 0:
        return pred_masks.new_tensor(0.0)
    return total / float(valid)


def compute_vote_consistency_loss(
    pred_offset: torch.Tensor,
    instance_maps: torch.Tensor,
    min_pixels: int = 20,
    touch_boost: float = 1.6,
) -> torch.Tensor:
    """
    Encourage all visible parts of one GT instance to vote for a compact center.
    This is designed for occluded/truncated long leaves that tend to split.
    """
    if pred_offset.ndim != 4 or pred_offset.shape[1] != 2:
        raise ValueError(f"pred_offset should be [B,2,H,W], got {tuple(pred_offset.shape)}")

    inst = instance_maps.to(pred_offset.device, non_blocking=True).long()
    bsz, _, h, w = pred_offset.shape
    min_pixels = max(1, int(min_pixels))
    touch_boost = float(max(1.0, touch_boost))

    weighted_loss = pred_offset.new_tensor(0.0)
    weight_sum = pred_offset.new_tensor(0.0)

    for b in range(bsz):
        ids = torch.unique(inst[b])
        ids = ids[ids > 0]
        for ins_id in ids.tolist():
            ys, xs = torch.where(inst[b] == int(ins_id))
            if ys.numel() < min_pixels:
                continue

            dy = pred_offset[b, 0, ys, xs]
            dx = pred_offset[b, 1, ys, xs]
            vote_y = ys.float() + dy
            vote_x = xs.float() + dx

            mean_y = vote_y.mean()
            mean_x = vote_x.mean()
            dist = torch.sqrt((vote_y - mean_y) ** 2 + (vote_x - mean_x) ** 2 + 1e-6)
            inst_loss = dist.mean()

            touching_border = bool(
                (int(ys.min().item()) == 0)
                or (int(ys.max().item()) == h - 1)
                or (int(xs.min().item()) == 0)
                or (int(xs.max().item()) == w - 1)
            )
            w_inst = touch_boost if touching_border else 1.0
            weighted_loss = weighted_loss + float(w_inst) * inst_loss
            weight_sum = weight_sum + float(w_inst)

    if float(weight_sum.item()) <= 0:
        return pred_offset.new_tensor(0.0)
    return weighted_loss / weight_sum


def tensor_image_to_rgb_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().float().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def color_for_id(idx: int):
    if idx <= 0:
        return (0, 0, 0)
    b = (37 * idx + 23) % 256
    g = (17 * idx + 91) % 256
    r = (97 * idx + 53) % 256
    return int(b), int(g), int(r)


def colorize_instance_map(instance_map: np.ndarray) -> np.ndarray:
    h, w = instance_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(instance_map)
    for idx in ids:
        i = int(idx)
        if i <= 0:
            continue
        out[instance_map == i] = color_for_id(i)
    return out


def build_gt_instance_map_from_masks(gt_masks: torch.Tensor) -> np.ndarray:
    if gt_masks.numel() == 0:
        return np.zeros((int(gt_masks.shape[-2]), int(gt_masks.shape[-1])), dtype=np.int32)
    n, h, w = gt_masks.shape
    out = np.zeros((h, w), dtype=np.int32)
    for i in range(n):
        m = (gt_masks[i].detach().cpu().numpy() > 0.5)
        out[m] = i + 1
    return out


def build_pred_instance_map(outputs: Dict[str, torch.Tensor], match: Tuple[torch.Tensor, torch.Tensor], sample_idx: int = 0, threshold: float = 0.5):
    pred_logits = outputs["pred_logits"][sample_idx].detach()
    pred_masks = outputs["pred_masks"][sample_idx].detach()
    row_ind, col_ind = match
    h, w = int(pred_masks.shape[-2]), int(pred_masks.shape[-1])
    out = np.zeros((h, w), dtype=np.int32)
    if row_ind.numel() == 0:
        return out

    probs = pred_logits.softmax(dim=-1)
    leaf_scores = probs[row_ind, 0]
    order = torch.argsort(leaf_scores, descending=True)
    new_id = 1
    for k in order.tolist():
        q_idx = int(row_ind[k].item())
        m = (pred_masks[q_idx].sigmoid().cpu().numpy() > threshold)
        if m.any():
            out[m] = new_id
            new_id += 1
    return out


def save_visualization(image_tensor: torch.Tensor, gt_masks: torch.Tensor, pred_instance_map: np.ndarray, save_path: Path):
    image_rgb = tensor_image_to_rgb_uint8(image_tensor)
    gt_instance_map = build_gt_instance_map_from_masks(gt_masks)
    gt_color = colorize_instance_map(gt_instance_map)
    pred_color = colorize_instance_map(pred_instance_map)
    overlay = cv2.addWeighted(image_rgb, 0.65, pred_color, 0.35, 0.0)

    top = np.concatenate([image_rgb, gt_color], axis=1)
    bottom = np.concatenate([pred_color, overlay], axis=1)
    panel = np.concatenate([top, bottom], axis=0)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)

    cv2.putText(panel_bgr, "Input", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel_bgr, "GT Instance", (panel_bgr.shape[1] // 2 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel_bgr, "Pred Instance", (8, panel_bgr.shape[0] // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel_bgr, "Pred Overlay", (panel_bgr.shape[1] // 2 + 8, panel_bgr.shape[0] // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), panel_bgr)


def init_csv_logger(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_total",
                    "train_cls",
                    "train_mask",
                    "train_dice",
                    "train_center",
                    "train_offset",
                    "train_vote_consistency",
                    "train_separation",
                    "train_repulsion",
                    "train_conflict",
                    "train_affinity",
                    "train_overlap_excl",
                    "val_total",
                    "val_cls",
                    "val_mask",
                    "val_dice",
                    "val_center",
                    "val_offset",
                    "val_vote_consistency",
                    "val_separation",
                    "val_repulsion",
                    "val_conflict",
                    "val_affinity",
                    "val_overlap_excl",
                    "lr",
                ]
            )


def append_csv_log(csv_path: Path, row: List):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_checkpoint(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def _extract_state_dict_from_checkpoint(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format: expected dict-like checkpoint.")


def _normalize_backbone_state_dict_keys(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for k, v in raw_state.items():
        key = str(k)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone.backbone."):
            new_key = key
        elif key.startswith("backbone."):
            # from SwinBackbone.state_dict(): backbone.patch_embed...
            new_key = "backbone." + key
        else:
            # from pure timm backbone keys: patch_embed...
            new_key = "backbone.backbone." + key
        normalized[new_key] = v
    return normalized


def load_pretrained_backbone_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    strict: bool = False,
):
    if not ckpt_path:
        return

    p = Path(ckpt_path)
    if not p.is_file():
        raise FileNotFoundError(f"pretrained_backbone_path does not exist: {p}")

    ckpt_obj = torch.load(str(p), map_location="cpu")
    raw_state = _extract_state_dict_from_checkpoint(ckpt_obj)
    normalized_state = _normalize_backbone_state_dict_keys(raw_state)

    model_keys = set(model.state_dict().keys())
    backbone_state = {k: v for k, v in normalized_state.items() if k in model_keys and k.startswith("backbone.")}
    if len(backbone_state) == 0:
        raise RuntimeError(
            f"No compatible backbone keys found in checkpoint: {p}. "
            f"Please check key prefixes (expected backbone.*)."
        )

    missing, unexpected = model.load_state_dict(backbone_state, strict=bool(strict))
    print(f"[INFO] Loaded pretrained backbone from: {p}")
    print(f"[INFO] Loaded backbone tensors: {len(backbone_state)}")
    print(f"[INFO] load_state_dict strict={bool(strict)} | missing={len(missing)} | unexpected={len(unexpected)}")


def make_experiment_dir(base_dir: Path, prefix: str = "exp") -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"{prefix}_{stamp}"
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    idx = 1
    while True:
        alt = base_dir / f"{prefix}_{stamp}_{idx:02d}"
        if not alt.exists():
            alt.mkdir(parents=True, exist_ok=False)
            return alt
        idx += 1


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    matcher: HungarianMatcher,
    device: torch.device,
    train_mode: bool,
    args,
    epoch: int,
    vis_dir: Path,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_mask = 0.0
    total_dice = 0.0
    total_center = 0.0
    total_offset = 0.0
    total_vote_consistency = 0.0
    total_separation = 0.0
    total_repulsion = 0.0
    total_conflict = 0.0
    total_affinity = 0.0
    total_overlap_excl = 0.0
    steps = 0
    vis_saved = False
    autocast_enabled = bool(args.amp and device.type == "cuda")

    max_steps = args.max_train_steps if train_mode else args.max_val_steps
    total_steps = len(loader)
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)

    phase = "train" if train_mode else "val"
    progress = tqdm(
        enumerate(loader),
        total=total_steps,
        desc=f"Epoch {epoch:03d} [{phase}]",
        dynamic_ncols=True,
        leave=False,
        disable=bool(args.no_tqdm),
    )

    for step, batch in progress:
        if max_steps > 0 and step >= max_steps:
            break

        images = batch["image"].to(device, non_blocking=True)
        targets = prepare_targets(batch, device)

        with torch.set_grad_enabled(train_mode):
            amp_context = torch.amp.autocast(device_type="cuda", enabled=True) if autocast_enabled else nullcontext()
            with amp_context:
                outputs = model(images)
                loss_dict = compute_losses(
                    outputs=outputs,
                    targets=targets,
                    matcher=matcher,
                    no_object_class=1,
                    w_cls=args.w_cls,
                    w_mask=args.w_mask,
                    w_dice=args.w_dice,
                )
                loss = loss_dict["loss"]
                loss_center = torch.tensor(0.0, device=device)
                loss_offset = torch.tensor(0.0, device=device)
                loss_vote_consistency = torch.tensor(0.0, device=device)
                loss_separation = torch.tensor(0.0, device=device)
                loss_repulsion = torch.tensor(0.0, device=device)
                loss_conflict = torch.tensor(0.0, device=device)
                loss_affinity = torch.tensor(0.0, device=device)
                loss_overlap_excl = torch.tensor(0.0, device=device)

                if bool(args.enable_aux_heads):
                    if "pred_center" not in outputs or "pred_offset" not in outputs:
                        raise RuntimeError("Aux heads enabled but model outputs missing pred_center/pred_offset.")
                    aux_targets = build_center_offset_targets_for_batch(
                        instance_maps=batch["instance_map"],
                        device=device,
                        center_sigma=args.center_sigma,
                        offset_clip=args.offset_clip,
                    )
                    pred_center = outputs["pred_center"].squeeze(1)
                    loss_center = F.binary_cross_entropy_with_logits(pred_center, aux_targets["center"])

                    fg = aux_targets["fg_mask"].unsqueeze(1)
                    offset_l1 = torch.abs(outputs["pred_offset"] - aux_targets["offset"])
                    denom = fg.sum().clamp(min=1.0)
                    loss_offset = (offset_l1 * fg).sum() / denom
                    loss_vote_consistency = compute_vote_consistency_loss(
                        pred_offset=outputs["pred_offset"],
                        instance_maps=batch["instance_map"],
                        min_pixels=args.vote_consistency_min_pixels,
                        touch_boost=args.vote_consistency_touch_boost,
                    )

                    if float(args.w_separation) > 0:
                        if "pred_separation" not in outputs:
                            raise RuntimeError(
                                "w_separation > 0 but model outputs missing pred_separation. "
                                "Please ensure model aux head includes separation branch."
                            )
                        sep_target = build_separation_targets_for_batch(
                            instance_maps=batch["instance_map"],
                            device=device,
                            dilate_radius=args.separation_dilate,
                        )
                        pred_sep = outputs["pred_separation"].squeeze(1)
                        pos_w = torch.tensor(float(args.separation_pos_weight), device=device)
                        sep_bce = F.binary_cross_entropy_with_logits(pred_sep, sep_target, pos_weight=pos_w)
                        sep_dice = dice_loss_from_logits(pred_sep, sep_target)
                        loss_separation = sep_bce + float(args.separation_dice_weight) * sep_dice

                    if float(args.w_repulsion) > 0:
                        loss_repulsion = compute_boundary_repulsion_loss(
                            pred_offset=outputs["pred_offset"],
                            instance_maps=batch["instance_map"],
                            margin=args.repulsion_margin,
                            max_pairs=args.repulsion_max_pairs,
                        )

                    if float(args.w_conflict) > 0:
                        if "pred_conflict" not in outputs:
                            raise RuntimeError(
                                "w_conflict > 0 but model outputs missing pred_conflict. "
                                "Please ensure model aux head includes conflict branch."
                            )
                        conflict_target = build_conflict_targets_for_batch(
                            instance_maps=batch["instance_map"],
                            device=device,
                            dilate_radius=args.conflict_dilate,
                        )
                        pred_conflict = outputs["pred_conflict"].squeeze(1)
                        pos_w = torch.tensor(float(args.conflict_pos_weight), device=device)
                        conflict_bce = F.binary_cross_entropy_with_logits(
                            pred_conflict, conflict_target, pos_weight=pos_w
                        )
                        conflict_dice = dice_loss_from_logits(pred_conflict, conflict_target)
                        loss_conflict = conflict_bce + float(args.conflict_dice_weight) * conflict_dice

                    if float(args.w_affinity) > 0:
                        if "pred_affinity" not in outputs:
                            raise RuntimeError(
                                "w_affinity > 0 but model outputs missing pred_affinity. "
                                "Please ensure model aux head includes affinity branch."
                            )
                        aff = compute_affinity_embedding_loss(
                            pred_affinity=outputs["pred_affinity"],
                            instance_maps=batch["instance_map"],
                            min_pixels=args.affinity_min_pixels,
                            margin_var=args.affinity_margin_var,
                            margin_dist=args.affinity_margin_dist,
                            dist_weight=args.affinity_dist_weight,
                            reg_weight=args.affinity_reg_weight,
                            max_instances=args.affinity_max_instances,
                        )
                        loss_affinity = aff["total"]

                    loss = (
                        loss
                        + float(args.w_center) * loss_center
                        + float(args.w_offset) * loss_offset
                        + float(args.w_vote_consistency) * loss_vote_consistency
                        + float(args.w_separation) * loss_separation
                        + float(args.w_repulsion) * loss_repulsion
                        + float(args.w_conflict) * loss_conflict
                        + float(args.w_affinity) * loss_affinity
                    )

                if float(args.w_overlap_excl) > 0:
                    loss_overlap_excl = compute_mutual_exclusion_overlap_loss(
                        pred_masks=outputs["pred_masks"],
                        matches=loss_dict["matches"],
                        instance_maps=batch["instance_map"],
                        bg_margin=args.overlap_bg_margin,
                        bg_weight=args.overlap_bg_weight,
                    )
                    loss = loss + float(args.w_overlap_excl) * loss_overlap_excl

                loss_dict["loss_center"] = loss_center
                loss_dict["loss_offset"] = loss_offset
                loss_dict["loss_vote_consistency"] = loss_vote_consistency
                loss_dict["loss_separation"] = loss_separation
                loss_dict["loss_repulsion"] = loss_repulsion
                loss_dict["loss_conflict"] = loss_conflict
                loss_dict["loss_affinity"] = loss_affinity
                loss_dict["loss_overlap_excl"] = loss_overlap_excl
                loss_dict["loss"] = loss

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss_dict["loss"].detach().cpu().item())
        total_cls += float(loss_dict["loss_cls"].detach().cpu().item())
        total_mask += float(loss_dict["loss_mask"].detach().cpu().item())
        total_dice += float(loss_dict["loss_dice"].detach().cpu().item())
        total_center += float(loss_dict["loss_center"].detach().cpu().item())
        total_offset += float(loss_dict["loss_offset"].detach().cpu().item())
        total_vote_consistency += float(loss_dict["loss_vote_consistency"].detach().cpu().item())
        total_separation += float(loss_dict["loss_separation"].detach().cpu().item())
        total_repulsion += float(loss_dict["loss_repulsion"].detach().cpu().item())
        total_conflict += float(loss_dict["loss_conflict"].detach().cpu().item())
        total_affinity += float(loss_dict["loss_affinity"].detach().cpu().item())
        total_overlap_excl += float(loss_dict["loss_overlap_excl"].detach().cpu().item())
        steps += 1

        progress.set_postfix(
            loss=f"{loss_dict['loss'].detach().cpu().item():.4f}",
            cls=f"{loss_dict['loss_cls'].detach().cpu().item():.4f}",
            mask=f"{loss_dict['loss_mask'].detach().cpu().item():.4f}",
            dice=f"{loss_dict['loss_dice'].detach().cpu().item():.4f}",
            ctr=f"{loss_dict['loss_center'].detach().cpu().item():.4f}",
            off=f"{loss_dict['loss_offset'].detach().cpu().item():.4f}",
            vcon=f"{loss_dict['loss_vote_consistency'].detach().cpu().item():.4f}",
            sep=f"{loss_dict['loss_separation'].detach().cpu().item():.4f}",
            rep=f"{loss_dict['loss_repulsion'].detach().cpu().item():.4f}",
            cfl=f"{loss_dict['loss_conflict'].detach().cpu().item():.4f}",
            aff=f"{loss_dict['loss_affinity'].detach().cpu().item():.4f}",
            mex=f"{loss_dict['loss_overlap_excl'].detach().cpu().item():.4f}",
        )

        if (not vis_saved) and (args.vis_every > 0) and (epoch % args.vis_every == 0):
            match0 = loss_dict["matches"][0]
            pred_map = build_pred_instance_map(outputs, match0, sample_idx=0, threshold=0.5)
            image0 = batch["image"][0]
            gt_masks0 = batch["gt_masks"][0]
            img_name = str(batch["meta"][0].get("image_name", f"epoch{epoch}_step{step}.png"))
            vis_path = vis_dir / f"epoch{epoch:03d}_{img_name}.png"
            save_visualization(image0, gt_masks0, pred_map, vis_path)
            vis_saved = True

    progress.close()

    if steps == 0:
        return {
            "loss": 0.0,
            "loss_cls": 0.0,
            "loss_mask": 0.0,
            "loss_dice": 0.0,
            "loss_center": 0.0,
            "loss_offset": 0.0,
            "loss_vote_consistency": 0.0,
            "loss_separation": 0.0,
            "loss_repulsion": 0.0,
            "loss_conflict": 0.0,
            "loss_affinity": 0.0,
            "loss_overlap_excl": 0.0,
            "steps": 0,
        }

    return {
        "loss": total_loss / steps,
        "loss_cls": total_cls / steps,
        "loss_mask": total_mask / steps,
        "loss_dice": total_dice / steps,
        "loss_center": total_center / steps,
        "loss_offset": total_offset / steps,
        "loss_vote_consistency": total_vote_consistency / steps,
        "loss_separation": total_separation / steps,
        "loss_repulsion": total_repulsion / steps,
        "loss_conflict": total_conflict / steps,
        "loss_affinity": total_affinity / steps,
        "loss_overlap_excl": total_overlap_excl / steps,
        "steps": steps,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    base_save_dir = Path(args.save_dir)
    save_dir = make_experiment_dir(base_save_dir, prefix="exp")
    ckpt_dir = save_dir / "checkpoints"
    log_dir = save_dir / "logs"
    vis_train_dir = save_dir / "vis" / "train"
    vis_val_dir = save_dir / "vis" / "val"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    vis_train_dir.mkdir(parents=True, exist_ok=True)
    vis_val_dir.mkdir(parents=True, exist_ok=True)

    config_path = log_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    csv_log_path = log_dir / "train_log.csv"
    init_csv_logger(csv_log_path)

    device = select_device(args.device)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Roots: {args.roots}")
    print(f"[INFO] Experiment dir: {save_dir}")
    print(f"[INFO] Loss weights: w_cls={args.w_cls}, w_mask={args.w_mask}, w_dice={args.w_dice}")
    print(
        f"[INFO] Aux heads: enable={bool(args.enable_aux_heads)}, "
        f"w_center={args.w_center}, w_offset={args.w_offset}, "
        f"w_vote_consistency={args.w_vote_consistency}, "
        f"w_separation={args.w_separation}, w_repulsion={args.w_repulsion}, "
        f"w_conflict={args.w_conflict}, w_affinity={args.w_affinity}, "
        f"w_overlap_excl={args.w_overlap_excl}, "
        f"vote_min_pixels={args.vote_consistency_min_pixels}, "
        f"vote_touch_boost={args.vote_consistency_touch_boost}, "
        f"center_sigma={args.center_sigma}, offset_clip={args.offset_clip}, "
        f"sep_dilate={args.separation_dilate}, sep_pos_weight={args.separation_pos_weight}, "
        f"sep_dice_weight={args.separation_dice_weight}, repulsion_margin={args.repulsion_margin}, "
        f"repulsion_max_pairs={args.repulsion_max_pairs}, "
        f"conflict_dilate={args.conflict_dilate}, conflict_pos_weight={args.conflict_pos_weight}, "
        f"affinity_dim={args.affinity_dim}, affinity_min_pixels={args.affinity_min_pixels}, "
        f"affinity_margin_var={args.affinity_margin_var}, affinity_margin_dist={args.affinity_margin_dist}, "
        f"overlap_bg_margin={args.overlap_bg_margin}, overlap_bg_weight={args.overlap_bg_weight}"
    )
    print(
        f"[INFO] Patch-scale weighting: enable={bool(args.enable_patch_scale_weighting)}, "
        f"w512={args.patch_scale_weight_512}, w768={args.patch_scale_weight_768}, w1024={args.patch_scale_weight_1024}"
    )

    train_transform = get_train_transform(target_size=args.input_size)
    val_transform = get_val_transform(target_size=args.input_size)

    train_ds, val_ds = build_train_val_datasets_by_big_image(
        root_dirs=args.roots,
        train_transform=train_transform,
        val_transform=val_transform,
        seed=args.seed,
        remap_instance_ids=False,
        skip_empty=False,
        enable_patch_scale_weighting=bool(args.enable_patch_scale_weighting),
        patch_scale_weights={
            512: float(args.patch_scale_weight_512),
            768: float(args.patch_scale_weight_768),
            1024: float(args.patch_scale_weight_1024),
        },
    )
    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples: {len(val_ds)}")

    train_sampler = None
    train_shuffle = True
    if bool(args.enable_patch_scale_weighting):
        train_sampler = train_ds.build_weighted_sampler(num_samples=len(train_ds), replacement=True)
        train_shuffle = False
        sampling_summary = train_ds.get_patch_scale_sampling_summary()
        print(f"[INFO] Patch-scale sampling summary: {sampling_summary}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=bool(args.num_workers_pin_memory),
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=False,
        collate_fn=leaf_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.num_workers_pin_memory),
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=False,
        collate_fn=leaf_collate_fn,
    )

    model = LeafInstanceSegModel(
        num_queries=args.num_queries,
        hidden_dim=256,
        num_classes=2,
        mask_embed_dim=256,
        pretrained=args.pretrained,
        input_size=args.input_size,
        upsample_masks_to_input=True,
        enable_aux_heads=bool(args.enable_aux_heads),
        aux_affinity_dim=args.affinity_dim,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Params total={total_params:,}, trainable={trainable_params:,} (backbone not frozen)")

    # Optional backbone initialization from RiceSEG pretraining before optimizer construction.
    if args.pretrained_backbone_path:
        load_pretrained_backbone_weights(
            model=model,
            ckpt_path=args.pretrained_backbone_path,
            strict=args.pretrained_backbone_strict,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    matcher = HungarianMatcher(
        cls_weight=args.match_cls,
        mask_weight=args.match_mask,
        dice_weight=args.match_dice,
        match_size=args.match_size,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            ckpt = torch.load(str(resume_path), map_location=device)
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=not bool(args.enable_aux_heads))
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            print(
                f"[INFO] Resumed from {resume_path}, start_epoch={start_epoch}, best_val={best_val:.6f}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        else:
            print(f"[WARN] Resume checkpoint not found: {resume_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            matcher=matcher,
            device=device,
            train_mode=True,
            args=args,
            epoch=epoch,
            vis_dir=vis_train_dir,
        )
        val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            matcher=matcher,
            device=device,
            train_mode=False,
            args=args,
            epoch=epoch,
            vis_dir=vis_val_dir,
        )

        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - t0

        msg = (
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_total={train_stats['loss']:.4f} "
            f"(cls={train_stats['loss_cls']:.4f}, mask={train_stats['loss_mask']:.4f}, dice={train_stats['loss_dice']:.4f}, "
            f"center={train_stats['loss_center']:.4f}, offset={train_stats['loss_offset']:.4f}, "
            f"vote={train_stats['loss_vote_consistency']:.4f}, sep={train_stats['loss_separation']:.4f}, "
            f"rep={train_stats['loss_repulsion']:.4f}, cfl={train_stats['loss_conflict']:.4f}, "
            f"aff={train_stats['loss_affinity']:.4f}, mex={train_stats['loss_overlap_excl']:.4f}) | "
            f"val_total={val_stats['loss']:.4f} "
            f"(cls={val_stats['loss_cls']:.4f}, mask={val_stats['loss_mask']:.4f}, dice={val_stats['loss_dice']:.4f}, "
            f"center={val_stats['loss_center']:.4f}, offset={val_stats['loss_offset']:.4f}, "
            f"vote={val_stats['loss_vote_consistency']:.4f}, sep={val_stats['loss_separation']:.4f}, "
            f"rep={val_stats['loss_repulsion']:.4f}, cfl={val_stats['loss_conflict']:.4f}, "
            f"aff={val_stats['loss_affinity']:.4f}, mex={val_stats['loss_overlap_excl']:.4f}) | "
            f"lr={lr_now:.6e} | time={elapsed:.1f}s"
        )
        print(msg)

        append_csv_log(
            csv_log_path,
            [
                epoch,
                train_stats["loss"],
                train_stats["loss_cls"],
                train_stats["loss_mask"],
                train_stats["loss_dice"],
                train_stats["loss_center"],
                train_stats["loss_offset"],
                train_stats["loss_vote_consistency"],
                train_stats["loss_separation"],
                train_stats["loss_repulsion"],
                train_stats["loss_conflict"],
                train_stats["loss_affinity"],
                train_stats["loss_overlap_excl"],
                val_stats["loss"],
                val_stats["loss_cls"],
                val_stats["loss_mask"],
                val_stats["loss_dice"],
                val_stats["loss_center"],
                val_stats["loss_offset"],
                val_stats["loss_vote_consistency"],
                val_stats["loss_separation"],
                val_stats["loss_repulsion"],
                val_stats["loss_conflict"],
                val_stats["loss_affinity"],
                val_stats["loss_overlap_excl"],
                lr_now,
            ],
        )

        is_new_best = val_stats["loss"] < best_val
        if is_new_best:
            best_val = val_stats["loss"]

        latest_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val": best_val,
            "args": vars(args),
        }
        save_checkpoint(ckpt_dir / "latest.pth", latest_ckpt)

        if is_new_best:
            best_ckpt = dict(latest_ckpt)
            save_checkpoint(ckpt_dir / "best.pth", best_ckpt)
            print(f"[INFO] New best checkpoint saved. best_val={best_val:.6f}")

    print(f"[INFO] Training completed. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
