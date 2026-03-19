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
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_train_val_datasets_by_big_image
from losses_v5 import (
    HungarianMatcher,
    add_loss_args,
    build_center_offset_targets_for_batch,
    build_conflict_targets_for_batch,
    build_separation_targets_for_batch,
    compute_affinity_embedding_loss,
    compute_boundary_repulsion_loss,
    compute_losses,
    compute_mutual_exclusion_overlap_loss,
    compute_order_consistency_loss,
    compute_vote_consistency_loss,
    dice_loss_from_logits,
)
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
        "--reset_best_on_resume",
        action="store_true",
        help="When resuming, reset best_val to +inf so this run can write a new best.pth independently.",
    )
    parser.add_argument(
        "--pretrained_backbone_path",
        type=str,
        default="pretrain_riceseg\\outputs\\exp_20260307_124458\\checkpoints\\best_backbone_for_instance.pth",
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

    # Centralized loss/matcher args are defined in losses_v5.py.
    add_loss_args(parser)

    parser.add_argument("--vis_every", type=int, default=1, help="Save train/val visualization every N epochs.")
    parser.add_argument("--max_train_steps", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max_val_steps", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--num_workers_pin_memory", action="store_true", default=True)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        default=True,
        help="Enable cuDNN autotune for fixed input sizes.",
    )
    parser.add_argument("--no-cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help="Allow TF32 matmul/conv on Ampere+ GPU for faster training.",
    )
    parser.add_argument("--no-allow_tf32", dest="allow_tf32", action="store_false")
    parser.add_argument(
        "--tqdm_postfix_interval",
        type=int,
        default=20,
        help="Update tqdm postfix every N steps to reduce GPU-CPU sync overhead.",
    )
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


def configure_cuda_runtime(device: torch.device, args) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)


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
                    "train_order",
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
                    "val_order",
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

    total_loss = torch.zeros((), device=device)
    total_cls = torch.zeros((), device=device)
    total_mask = torch.zeros((), device=device)
    total_dice = torch.zeros((), device=device)
    total_center = torch.zeros((), device=device)
    total_offset = torch.zeros((), device=device)
    total_vote_consistency = torch.zeros((), device=device)
    total_separation = torch.zeros((), device=device)
    total_repulsion = torch.zeros((), device=device)
    total_conflict = torch.zeros((), device=device)
    total_affinity = torch.zeros((), device=device)
    total_order = torch.zeros((), device=device)
    total_overlap_excl = torch.zeros((), device=device)
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
    postfix_interval = max(1, int(getattr(args, "tqdm_postfix_interval", 20)))

    for step, batch in progress:
        if max_steps > 0 and step >= max_steps:
            break

        images = batch["image"].to(device, non_blocking=True)
        targets = prepare_targets(batch, device)

        grad_context = torch.enable_grad() if train_mode else torch.inference_mode()
        with grad_context:
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
                loss_order = torch.tensor(0.0, device=device)
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

                if float(args.w_order) > 0:
                    if "pred_order" not in outputs:
                        raise RuntimeError(
                            "w_order > 0 but model outputs missing pred_order. "
                            "Please ensure query head includes order branch."
                        )
                    loss_order = compute_order_consistency_loss(
                        pred_order=outputs["pred_order"],
                        matches=loss_dict["matches"],
                        targets=targets,
                        min_dy=args.order_min_dy,
                        pair_max_dist=args.order_pair_max_dist,
                        max_pairs=args.order_max_pairs,
                    )
                    loss = loss + float(args.w_order) * loss_order

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
                loss_dict["loss_order"] = loss_order
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

        total_loss = total_loss + loss_dict["loss"].detach()
        total_cls = total_cls + loss_dict["loss_cls"].detach()
        total_mask = total_mask + loss_dict["loss_mask"].detach()
        total_dice = total_dice + loss_dict["loss_dice"].detach()
        total_center = total_center + loss_dict["loss_center"].detach()
        total_offset = total_offset + loss_dict["loss_offset"].detach()
        total_vote_consistency = total_vote_consistency + loss_dict["loss_vote_consistency"].detach()
        total_separation = total_separation + loss_dict["loss_separation"].detach()
        total_repulsion = total_repulsion + loss_dict["loss_repulsion"].detach()
        total_conflict = total_conflict + loss_dict["loss_conflict"].detach()
        total_affinity = total_affinity + loss_dict["loss_affinity"].detach()
        total_order = total_order + loss_dict["loss_order"].detach()
        total_overlap_excl = total_overlap_excl + loss_dict["loss_overlap_excl"].detach()
        steps += 1

        should_postfix = (not bool(args.no_tqdm)) and (
            (step % postfix_interval == 0) or (step + 1 >= total_steps)
        )
        if should_postfix:
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
                ord=f"{loss_dict['loss_order'].detach().cpu().item():.4f}",
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
            "loss_order": 0.0,
            "loss_overlap_excl": 0.0,
            "steps": 0,
        }

    return {
        "loss": float((total_loss / steps).detach().cpu().item()),
        "loss_cls": float((total_cls / steps).detach().cpu().item()),
        "loss_mask": float((total_mask / steps).detach().cpu().item()),
        "loss_dice": float((total_dice / steps).detach().cpu().item()),
        "loss_center": float((total_center / steps).detach().cpu().item()),
        "loss_offset": float((total_offset / steps).detach().cpu().item()),
        "loss_vote_consistency": float((total_vote_consistency / steps).detach().cpu().item()),
        "loss_separation": float((total_separation / steps).detach().cpu().item()),
        "loss_repulsion": float((total_repulsion / steps).detach().cpu().item()),
        "loss_conflict": float((total_conflict / steps).detach().cpu().item()),
        "loss_affinity": float((total_affinity / steps).detach().cpu().item()),
        "loss_order": float((total_order / steps).detach().cpu().item()),
        "loss_overlap_excl": float((total_overlap_excl / steps).detach().cpu().item()),
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
    configure_cuda_runtime(device, args)
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(
            f"[INFO] CUDA runtime: cudnn_benchmark={torch.backends.cudnn.benchmark}, "
            f"allow_tf32_matmul={torch.backends.cuda.matmul.allow_tf32}, "
            f"allow_tf32_cudnn={torch.backends.cudnn.allow_tf32}"
        )
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
        f"[INFO] LTO order loss: w_order={args.w_order}, order_min_dy={args.order_min_dy}, "
        f"order_pair_max_dist={args.order_pair_max_dist}, order_max_pairs={args.order_max_pairs}"
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

    adamw_kwargs = {
        "params": model.parameters(),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(fused=True, **adamw_kwargs)
            print("[INFO] Optimizer: AdamW(fused=True)")
        except (TypeError, RuntimeError):
            optimizer = torch.optim.AdamW(**adamw_kwargs)
            print("[INFO] Optimizer: AdamW(fused=False, fallback)")
    else:
        optimizer = torch.optim.AdamW(**adamw_kwargs)
        print("[INFO] Optimizer: AdamW(cpu)")
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
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            if bool(args.reset_best_on_resume):
                best_val = float("inf")
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=not bool(args.enable_aux_heads))

            optim_loaded = False
            sched_loaded = False
            scaler_loaded = False
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    optim_loaded = True
                except ValueError as e:
                    # Common case when architecture adds/removes parameters (e.g. LTO order head).
                    print(
                        f"[WARN] Resume optimizer state incompatible with current model param groups: {e}. "
                        "Falling back to fresh optimizer state while keeping model weights."
                    )
            if optim_loaded and "scheduler" in ckpt and ckpt["scheduler"] is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                    sched_loaded = True
                except Exception as e:
                    print(f"[WARN] Failed to load scheduler state from resume checkpoint: {e}")
            if optim_loaded and "scaler" in ckpt and ckpt["scaler"] is not None:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                    scaler_loaded = True
                except Exception as e:
                    print(f"[WARN] Failed to load AMP scaler state from resume checkpoint: {e}")

            if (not sched_loaded) and start_epoch > 1:
                # Keep LR schedule roughly aligned when optimizer/scheduler state cannot be resumed.
                try:
                    scheduler.step(start_epoch - 1)
                except TypeError:
                    for _ in range(start_epoch - 1):
                        scheduler.step()

            print(
                f"[INFO] Resumed from {resume_path}, start_epoch={start_epoch}, best_val={best_val:.6f}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"optimizer_loaded={optim_loaded}, scheduler_loaded={sched_loaded}, scaler_loaded={scaler_loaded}"
            )
            if bool(args.reset_best_on_resume):
                print("[INFO] reset_best_on_resume=True: best_val has been reset for this run.")
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
            f"aff={train_stats['loss_affinity']:.4f}, ord={train_stats['loss_order']:.4f}, "
            f"mex={train_stats['loss_overlap_excl']:.4f}) | "
            f"val_total={val_stats['loss']:.4f} "
            f"(cls={val_stats['loss_cls']:.4f}, mask={val_stats['loss_mask']:.4f}, dice={val_stats['loss_dice']:.4f}, "
            f"center={val_stats['loss_center']:.4f}, offset={val_stats['loss_offset']:.4f}, "
            f"vote={val_stats['loss_vote_consistency']:.4f}, sep={val_stats['loss_separation']:.4f}, "
            f"rep={val_stats['loss_repulsion']:.4f}, cfl={val_stats['loss_conflict']:.4f}, "
            f"aff={val_stats['loss_affinity']:.4f}, ord={val_stats['loss_order']:.4f}, "
            f"mex={val_stats['loss_overlap_excl']:.4f}) | "
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
                train_stats["loss_order"],
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
                val_stats["loss_order"],
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
