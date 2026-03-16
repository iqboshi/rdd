#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import (  # noqa: E402
    DEFAULT_LEAF_CLASS_IDS,
    RICESEG_CLASS_NAMES,
    build_train_val_datasets,
    save_mapping_preview,
)
from model import RiceSegPretrainModel, extract_backbone_state_dict_for_instance  # noqa: E402
from transforms import get_train_transform, get_val_transform  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="RiceSEG leaf vs non-leaf semantic pretraining.")
    parser.add_argument("--data_root", type=str, default="data/external_data")
    parser.add_argument("--save_dir", type=str, default="pretrain_riceseg/outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--leaf_class_ids", nargs="+", type=int, default=list(DEFAULT_LEAF_CLASS_IDS))

    parser.add_argument("--w_ce", type=float, default=1.0)
    parser.add_argument("--w_dice", type=float, default=1.0)

    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--save_vis_every", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_val_steps", type=int, default=0)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--save_data_preview", action="store_true", default=False)
    parser.add_argument("--preview_num", type=int, default=12)
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


def make_experiment_dir(base_dir: Path, prefix: str = "exp") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{prefix}_{stamp}"
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=False)
        return exp_dir
    i = 1
    while True:
        c = base_dir / f"{prefix}_{stamp}_{i:02d}"
        if not c.exists():
            c.mkdir(parents=True, exist_ok=False)
            return c
        i += 1


def collate_fn(batch: List[Dict]):
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return {"image": images, "mask": masks, "meta": metas}


def binary_dice_loss_from_logits(seg_logits: torch.Tensor, target_mask: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs_leaf = torch.softmax(seg_logits, dim=1)[:, 1]
    target = (target_mask == 1).float()
    inter = (probs_leaf * target).sum(dim=(1, 2))
    den = probs_leaf.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = 1.0 - (2.0 * inter + eps) / (den + eps)
    return dice.mean()


@torch.no_grad()
def binary_iou_from_logits(seg_logits: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> float:
    pred = torch.argmax(seg_logits, dim=1)
    pred_leaf = pred == 1
    gt_leaf = target_mask == 1
    inter = (pred_leaf & gt_leaf).sum().item()
    union = (pred_leaf | gt_leaf).sum().item()
    return float(inter) / float(union + eps)


def compute_loss(seg_logits: torch.Tensor, target_mask: torch.Tensor, w_ce: float, w_dice: float):
    ce = F.cross_entropy(seg_logits, target_mask.long())
    dice = binary_dice_loss_from_logits(seg_logits, target_mask)
    total = w_ce * ce + w_dice * dice
    return {"loss": total, "loss_ce": ce, "loss_dice": dice}


def tensor_image_to_rgb_uint8(image_t: torch.Tensor) -> np.ndarray:
    arr = image_t.detach().cpu().float().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def colorize_binary_mask(mask_2d: np.ndarray) -> np.ndarray:
    m = (mask_2d > 0).astype(np.uint8) * 255
    colored = cv2.applyColorMap(m, cv2.COLORMAP_SUMMER)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def save_vis_sample(image_t: torch.Tensor, mask_t: torch.Tensor, seg_logits: torch.Tensor, save_path: Path):
    image = tensor_image_to_rgb_uint8(image_t)
    gt = mask_t.detach().cpu().numpy().astype(np.uint8)
    pred = torch.argmax(seg_logits, dim=0).detach().cpu().numpy().astype(np.uint8)

    gt_color = colorize_binary_mask(gt)
    pred_color = colorize_binary_mask(pred)
    overlay = cv2.addWeighted(image, 0.65, pred_color, 0.35, 0.0)

    panel = np.concatenate([image, gt_color, pred_color, overlay], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.putText(panel_bgr, "Image", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    w = image.shape[1]
    cv2.putText(panel_bgr, "GT Leaf", (w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel_bgr, "Pred Leaf", (w * 2 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel_bgr, "Overlay", (w * 3 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), panel_bgr)


def init_csv(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_ce",
                    "train_dice",
                    "train_iou",
                    "val_loss",
                    "val_ce",
                    "val_dice",
                    "val_iou",
                    "lr",
                ]
            )


def append_csv(csv_path: Path, row: List):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_checkpoint(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    train_mode: bool,
    args,
    epoch: int,
    vis_dir: Path,
):
    model.train(train_mode)
    max_steps = args.max_train_steps if train_mode else args.max_val_steps
    total_steps = len(loader) if max_steps <= 0 else min(len(loader), max_steps)
    phase = "train" if train_mode else "val"

    progress = tqdm(
        enumerate(loader),
        total=total_steps,
        desc=f"Epoch {epoch:03d} [{phase}]",
        leave=False,
        dynamic_ncols=True,
        disable=bool(args.no_tqdm),
    )

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    steps = 0
    vis_saved = False
    autocast_enabled = bool(args.amp and device.type == "cuda")

    for step, batch in progress:
        if max_steps > 0 and step >= max_steps:
            break
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True).long()

        with torch.set_grad_enabled(train_mode):
            amp_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if autocast_enabled else nullcontext()
            with amp_ctx:
                out = model(image)
                seg_logits = out["seg_logits"]
                losses = compute_loss(seg_logits, mask, w_ce=args.w_ce, w_dice=args.w_dice)
                loss = losses["loss"]

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        iou = binary_iou_from_logits(seg_logits.detach(), mask.detach())
        total_loss += float(losses["loss"].detach().cpu().item())
        total_ce += float(losses["loss_ce"].detach().cpu().item())
        total_dice += float(losses["loss_dice"].detach().cpu().item())
        total_iou += float(iou)
        steps += 1

        progress.set_postfix(
            loss=f"{losses['loss'].detach().cpu().item():.4f}",
            ce=f"{losses['loss_ce'].detach().cpu().item():.4f}",
            dice=f"{losses['loss_dice'].detach().cpu().item():.4f}",
            iou=f"{iou:.4f}",
        )

        if (not vis_saved) and (args.save_vis_every > 0) and (epoch % args.save_vis_every == 0):
            vis_name = str(batch["meta"][0].get("image_name", f"e{epoch}_s{step}.png"))
            save_vis_sample(
                image_t=batch["image"][0],
                mask_t=batch["mask"][0],
                seg_logits=seg_logits[0].detach().cpu(),
                save_path=vis_dir / f"epoch{epoch:03d}_{vis_name}.png",
            )
            vis_saved = True

    progress.close()

    if steps == 0:
        return {"loss": 0.0, "ce": 0.0, "dice": 0.0, "iou": 0.0}

    return {
        "loss": total_loss / steps,
        "ce": total_ce / steps,
        "dice": total_dice / steps,
        "iou": total_iou / steps,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    exp_dir = make_experiment_dir(Path(args.save_dir), prefix="exp")
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    vis_train_dir = exp_dir / "vis" / "train"
    vis_val_dir = exp_dir / "vis" / "val"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    vis_train_dir.mkdir(parents=True, exist_ok=True)
    vis_val_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data root: {args.data_root}")
    print(f"[INFO] Experiment dir: {exp_dir}")
    print(f"[INFO] RiceSEG classes: {RICESEG_CLASS_NAMES}")
    print(f"[INFO] Leaf class ids: {args.leaf_class_ids}")

    with (log_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    csv_path = log_dir / "train_log.csv"
    init_csv(csv_path)

    if args.save_data_preview:
        save_mapping_preview(
            data_root=args.data_root,
            save_dir=exp_dir / "data_preview",
            num_samples=args.preview_num,
            leaf_class_ids=args.leaf_class_ids,
            seed=args.seed,
        )

    train_tf = get_train_transform(target_size=args.input_size)
    val_tf = get_val_transform(target_size=args.input_size)
    train_ds, val_ds = build_train_val_datasets(
        data_root=args.data_root,
        train_transform=train_tf,
        val_transform=val_tf,
        val_ratio=args.val_ratio,
        seed=args.seed,
        leaf_class_ids=args.leaf_class_ids,
    )
    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    model = RiceSegPretrainModel(
        num_classes=2,
        hidden_dim=256,
        pretrained=args.pretrained,
        input_size=args.input_size,
        upsample_to_input=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        rp = Path(args.resume)
        if rp.is_file():
            ckpt = torch.load(str(rp), map_location=device)
            model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            print(f"[INFO] Resumed from {rp}, start_epoch={start_epoch}, best_val={best_val:.6f}")
        else:
            print(f"[WARN] Resume checkpoint not found: {rp}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
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
            device=device,
            train_mode=False,
            args=args,
            epoch=epoch,
            vis_dir=vis_val_dir,
        )
        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_stats['loss']:.4f} (ce={train_stats['ce']:.4f}, dice={train_stats['dice']:.4f}, iou={train_stats['iou']:.4f}) | "
            f"val_loss={val_stats['loss']:.4f} (ce={val_stats['ce']:.4f}, dice={val_stats['dice']:.4f}, iou={val_stats['iou']:.4f}) | "
            f"lr={lr_now:.6e} | time={dt:.1f}s"
        )

        append_csv(
            csv_path,
            [
                epoch,
                train_stats["loss"],
                train_stats["ce"],
                train_stats["dice"],
                train_stats["iou"],
                val_stats["loss"],
                val_stats["ce"],
                val_stats["dice"],
                val_stats["iou"],
                lr_now,
            ],
        )

        latest = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val": best_val,
            "args": vars(args),
        }
        save_checkpoint(ckpt_dir / "latest.pth", latest)

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best = dict(latest)
            best["best_val"] = best_val
            save_checkpoint(ckpt_dir / "best.pth", best)

            full_state = model.state_dict()
            backbone_for_instance = extract_backbone_state_dict_for_instance(full_state)
            save_checkpoint(
                ckpt_dir / "best_backbone_for_instance.pth",
                {
                    "state_dict": backbone_for_instance,
                    "best_val": best_val,
                    "epoch": epoch,
                },
            )
            save_checkpoint(
                ckpt_dir / "best_backbone_module.pth",
                {
                    "state_dict": model.backbone.state_dict(),
                    "best_val": best_val,
                    "epoch": epoch,
                },
            )
            print(f"[INFO] New best checkpoint saved. best_val={best_val:.6f}")

    print(f"[INFO] Pretraining completed. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
