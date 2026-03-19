#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def add_loss_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register all loss-related CLI args in one place."""
    parser.add_argument("--w_cls", type=float, default=0.5, help="Weight for classification loss.")
    parser.add_argument("--w_mask", type=float, default=2.0, help="Weight for mask BCE loss.")
    parser.add_argument("--w_dice", type=float, default=2.0, help="Weight for dice loss.")
    parser.add_argument(
        "--enable_aux_heads",
        action="store_true",
        default=True,
        help="Enable center+offset auxiliary heads (A-v1).",
    )
    parser.add_argument("--no-enable_aux_heads", dest="enable_aux_heads", action="store_false")
    parser.add_argument("--w_center", type=float, default=0.15, help="Weight for center heatmap BCE aux loss.")
    parser.add_argument("--w_offset", type=float, default=0.01, help="Weight for offset L1 aux loss.")
    parser.add_argument(
        "--w_vote_consistency",
        type=float,
        default=0.005,
        help="Weight for instance center-vote consistency loss (A-v5).",
    )
    parser.add_argument(
        "--w_separation",
        type=float,
        default=0.08,
        help="Weight for boundary separation supervision loss (A-v6).",
    )
    parser.add_argument(
        "--w_repulsion",
        type=float,
        default=0.015,
        help="Weight for cross-instance vote repulsion loss on touching boundaries (A-v6).",
    )
    parser.add_argument(
        "--w_conflict",
        type=float,
        default=0.05,
        help="Weight for boundary conflict supervision loss (A7).",
    )
    parser.add_argument(
        "--w_affinity",
        type=float,
        default=0.03,
        help="Weight for pixel affinity embedding discriminative loss (A7).",
    )
    parser.add_argument(
        "--w_overlap_excl",
        type=float,
        default=0.02,
        help="Weight for mutual-exclusion overlap penalty on matched instance masks (A7).",
    )
    parser.add_argument(
        "--w_order",
        type=float,
        default=0.0,
        help="Weight for topology order consistency loss (LTO).",
    )
    parser.add_argument(
        "--order_min_dy",
        type=float,
        default=4.0,
        help="Minimum GT centroid-y gap (px) to form a valid order pair.",
    )
    parser.add_argument(
        "--order_pair_max_dist",
        type=float,
        default=180.0,
        help="Maximum centroid distance (px) to form a local topology order pair. <=0 disables the limit.",
    )
    parser.add_argument(
        "--order_max_pairs",
        type=int,
        default=4096,
        help="Maximum sampled order pairs per image (nearest pairs kept first).",
    )
    parser.add_argument(
        "--vote_consistency_min_pixels",
        type=int,
        default=24,
        help="Minimum pixels for an instance to participate in vote consistency loss.",
    )
    parser.add_argument(
        "--vote_consistency_touch_boost",
        type=float,
        default=1.8,
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
    return parser


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
        bsz, _, _ = pred_logits.shape
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

            pred_flat = pred_m.flatten(1).float()  # [Q, P]
            tgt_flat = tgt_m.flatten(1).float()  # [N, P]
            num_pix = float(max(pred_flat.shape[1], 1))

            # BCEWithLogits(x, y) = softplus(x) - x * y, computed as pairwise matrix ops.
            softplus_mean = F.softplus(pred_flat).mean(dim=1, keepdim=True)  # [Q, 1]
            xy_mean = (pred_flat @ tgt_flat.t()) / num_pix  # [Q, N]
            cost_mask = softplus_mean - xy_mean

            pred_prob = pred_flat.sigmoid()
            numerator = 2.0 * (pred_prob @ tgt_flat.t())
            denominator = pred_prob.sum(dim=1, keepdim=True) + tgt_flat.sum(dim=1).unsqueeze(0)
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


@torch.no_grad()
def build_order_targets_for_batch(
    matches: List[Tuple[torch.Tensor, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    min_dy: float = 4.0,
    pair_max_dist: float = 180.0,
    max_pairs: int = 4096,
) -> List[Dict[str, torch.Tensor]]:
    """
    Build pairwise query order targets from GT centroid ordering.
    We only keep local pairs to avoid noisy long-range ranking constraints.
    """
    pair_targets: List[Dict[str, torch.Tensor]] = []
    min_dy = float(max(0.0, min_dy))
    pair_max_dist = float(pair_max_dist)
    max_pairs = max(1, int(max_pairs))

    for b, (row_ind, col_ind) in enumerate(matches):
        device = row_ind.device
        if col_ind.numel() < 2:
            empty = torch.zeros((0,), dtype=torch.long, device=device)
            empty_f = torch.zeros((0,), dtype=torch.float32, device=device)
            pair_targets.append({"q_i": empty, "q_j": empty, "target_sign": empty_f, "pair_weight": empty_f})
            continue

        gt_masks = targets[b]["masks"][col_ind].float()  # [M, H, W]
        binary_masks = (gt_masks > 0.5).float()
        pixel_counts = binary_masks.sum(dim=(1, 2))
        valid_local_ids_t = torch.where(pixel_counts > 0)[0]

        if valid_local_ids_t.numel() < 2:
            empty = torch.zeros((0,), dtype=torch.long, device=device)
            empty_f = torch.zeros((0,), dtype=torch.float32, device=device)
            pair_targets.append({"q_i": empty, "q_j": empty, "target_sign": empty_f, "pair_weight": empty_f})
            continue

        masks_valid = binary_masks[valid_local_ids_t]  # [K, H, W]
        _, h, w = masks_valid.shape
        grid_y = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1)
        grid_x = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w)
        counts_f = pixel_counts[valid_local_ids_t].clamp(min=1.0)
        cy = (masks_valid * grid_y).sum(dim=(1, 2)) / counts_f
        cx = (masks_valid * grid_x).sum(dim=(1, 2)) / counts_f
        centers = torch.stack([cy, cx], dim=1)  # [K, 2], (cy, cx)
        query_ids = row_ind[valid_local_ids_t]  # [K]

        dcy = centers[:, 0][:, None] - centers[:, 0][None, :]
        dcx = centers[:, 1][:, None] - centers[:, 1][None, :]
        dist = torch.sqrt(dcy ** 2 + dcx ** 2 + 1e-6)

        upper = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
        valid_pair = upper & (torch.abs(dcy) >= min_dy)
        if pair_max_dist > 0:
            valid_pair = valid_pair & (dist <= pair_max_dist)

        pi, pj = torch.where(valid_pair)
        if pi.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=device)
            empty_f = torch.zeros((0,), dtype=torch.float32, device=device)
            pair_targets.append({"q_i": empty, "q_j": empty, "target_sign": empty_f, "pair_weight": empty_f})
            continue

        pair_dist = dist[pi, pj]
        if int(pi.numel()) > max_pairs:
            keep = torch.argsort(pair_dist)[:max_pairs]
            pi = pi[keep]
            pj = pj[keep]
            pair_dist = pair_dist[keep]

        signs = torch.where(dcy[pi, pj] > 0, torch.ones_like(pair_dist), -torch.ones_like(pair_dist))
        if pair_max_dist > 0:
            weights = (1.0 - pair_dist / max(pair_max_dist, 1e-6)).clamp(min=0.05, max=1.0)
        else:
            weights = torch.ones_like(pair_dist)

        pair_targets.append(
            {
                "q_i": query_ids[pi],
                "q_j": query_ids[pj],
                "target_sign": signs,
                "pair_weight": weights,
            }
        )

    return pair_targets


def compute_order_consistency_loss(
    pred_order: torch.Tensor,
    matches: List[Tuple[torch.Tensor, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    min_dy: float = 4.0,
    pair_max_dist: float = 180.0,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """
    Pairwise ranking loss on query order logits:
    GT instance with larger centroid-y should get larger order logit.
    """
    if pred_order.ndim != 2:
        raise ValueError(f"pred_order should be [B,Q], got {tuple(pred_order.shape)}")

    pair_targets = build_order_targets_for_batch(
        matches=matches,
        targets=targets,
        min_dy=min_dy,
        pair_max_dist=pair_max_dist,
        max_pairs=max_pairs,
    )

    total = pred_order.new_tensor(0.0)
    weight_sum = pred_order.new_tensor(0.0)
    for b, pair in enumerate(pair_targets):
        q_i = pair["q_i"]
        if q_i.numel() == 0:
            continue
        q_j = pair["q_j"]
        sign = pair["target_sign"]
        w = pair["pair_weight"]

        diff = pred_order[b, q_i] - pred_order[b, q_j]
        rank_loss = F.softplus(-sign * diff)
        total = total + (rank_loss * w).sum()
        weight_sum = weight_sum + w.sum()

    return total / torch.clamp(weight_sum, min=1e-12)


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
        emb = pred_affinity[b].permute(1, 2, 0).reshape(-1, pred_affinity.shape[1])  # [HW, D]
        labels = inst[b].reshape(-1)  # [HW]

        fg = labels > 0
        labels_fg = labels[fg]
        if labels_fg.numel() == 0:
            continue
        emb_fg = emb[fg]

        ids = torch.unique(labels_fg)
        if ids.numel() > max_instances:
            perm = torch.randperm(ids.numel(), device=ids.device)[:max_instances]
            ids = ids[perm]

        sampled = torch.isin(labels_fg, ids)
        labels_sel = labels_fg[sampled]
        emb_sel = emb_fg[sampled]
        if labels_sel.numel() == 0:
            continue

        _, inverse, counts = torch.unique(
            labels_sel, sorted=False, return_inverse=True, return_counts=True
        )
        valid_inst = counts >= min_pixels
        if torch.where(valid_inst)[0].numel() == 0:
            continue

        pixel_keep = valid_inst[inverse]
        emb_keep = emb_sel[pixel_keep]
        inv_keep_old = inverse[pixel_keep]

        remap = torch.cumsum(valid_inst.to(torch.long), dim=0) - 1
        inv_keep = remap[inv_keep_old]
        counts_keep = counts[valid_inst].to(dtype=emb_keep.dtype)
        n_inst = int(counts_keep.numel())

        centers_sum = emb_keep.new_zeros((n_inst, emb_keep.shape[1]))
        centers_sum.index_add_(0, inv_keep, emb_keep)
        centers = centers_sum / counts_keep.unsqueeze(1)  # [K, D]

        intra = torch.norm(emb_keep - centers[inv_keep], dim=1)
        var_term = F.relu(intra - float(margin_var)) ** 2
        var_per_inst_sum = var_term.new_zeros((n_inst,))
        var_per_inst_sum.index_add_(0, inv_keep, var_term)
        var_sum = var_sum + (var_per_inst_sum / counts_keep).sum()

        reg_sum = reg_sum + centers.norm(dim=1).mean()
        valid_batches += 1

        if centers.shape[0] > 1:
            dist_mat = torch.cdist(centers, centers, p=2)
            eye = torch.eye(dist_mat.shape[0], dtype=torch.bool, device=dist_mat.device)
            off_diag = dist_mat[~eye]
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

        bg_denom = bg.sum().clamp(min=1.0)
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
