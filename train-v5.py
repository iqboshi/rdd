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
    parser.add_argument("--center_sigma", type=float, default=4.0, help="Sigma for center heatmap targets.")
    parser.add_argument("--offset_clip", type=float, default=64.0, help="Clip absolute target offset values.")
    parser.add_argument("--match_cls", type=float, default=1.0)
    parser.add_argument("--match_mask", type=float, default=1.0)
    parser.add_argument("--match_dice", type=float, default=1.0)
    parser.add_argument("--match_size", type=int, default=128)

    parser.add_argument("--vis_every", type=int, default=1, help="Save train/val visualization every N epochs.")
    parser.add_argument("--max_train_steps", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max_val_steps", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--num_workers_pin_memory", action="store_true", default=True)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
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
                    "val_total",
                    "val_cls",
                    "val_mask",
                    "val_dice",
                    "val_center",
                    "val_offset",
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
                    loss = loss + float(args.w_center) * loss_center + float(args.w_offset) * loss_offset

                loss_dict["loss_center"] = loss_center
                loss_dict["loss_offset"] = loss_offset
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
        steps += 1

        progress.set_postfix(
            loss=f"{loss_dict['loss'].detach().cpu().item():.4f}",
            cls=f"{loss_dict['loss_cls'].detach().cpu().item():.4f}",
            mask=f"{loss_dict['loss_mask'].detach().cpu().item():.4f}",
            dice=f"{loss_dict['loss_dice'].detach().cpu().item():.4f}",
            ctr=f"{loss_dict['loss_center'].detach().cpu().item():.4f}",
            off=f"{loss_dict['loss_offset'].detach().cpu().item():.4f}",
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
            "steps": 0,
        }

    return {
        "loss": total_loss / steps,
        "loss_cls": total_cls / steps,
        "loss_mask": total_mask / steps,
        "loss_dice": total_dice / steps,
        "loss_center": total_center / steps,
        "loss_offset": total_offset / steps,
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
        f"center_sigma={args.center_sigma}, offset_clip={args.offset_clip}"
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
    )
    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
            f"center={train_stats['loss_center']:.4f}, offset={train_stats['loss_offset']:.4f}) | "
            f"val_total={val_stats['loss']:.4f} "
            f"(cls={val_stats['loss_cls']:.4f}, mask={val_stats['loss_mask']:.4f}, dice={val_stats['loss_dice']:.4f}, "
            f"center={val_stats['loss_center']:.4f}, offset={val_stats['loss_offset']:.4f}) | "
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
                val_stats["loss"],
                val_stats["loss_cls"],
                val_stats["loss_mask"],
                val_stats["loss_dice"],
                val_stats["loss_center"],
                val_stats["loss_offset"],
                lr_now,
            ],
        )

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

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_ckpt = dict(latest_ckpt)
            best_ckpt["best_val"] = best_val
            save_checkpoint(ckpt_dir / "best.pth", best_ckpt)
            print(f"[INFO] New best checkpoint saved. best_val={best_val:.6f}")

    print(f"[INFO] Training completed. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
