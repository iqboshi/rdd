#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import collect_samples, load_image, load_instance_map, split_samples_by_big_image
from model import LeafInstanceSegModel


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-level leaf-only instance segmentation inference (v1).")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["data/patches_size512", "data/patches_size768", "data/patches_size1024"],
        help="One or more patch roots containing images/instance subfolders.",
    )
    parser.add_argument("--checkpoint", type=str, default="outputs/exp_20260308_155448/checkpoints/latest.pth", help="Checkpoint path.")
    parser.add_argument("--input_size", type=int, default=512, help="Inference input size.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu.")
    parser.add_argument("--save_dir", type=str, default="outputs/predict-v1", help="Output directory.")
    parser.add_argument("--split", type=str, default="val", choices=["all", "train", "val"], help="Predict all samples, or only train/val split by big image.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio used when --split is train/val.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for big-image split.")
    parser.add_argument("--score_thresh", type=float, default=0.35, help="Score threshold.")
    parser.add_argument("--mask_thresh", type=float, default=0.40, help="Mask threshold.")
    parser.add_argument("--nms_iou_thresh", type=float, default=0.65, help="Mask IoU NMS threshold.")
    parser.add_argument("--contain_thresh", type=float, default=0.75, help="Containment threshold.")
    parser.add_argument("--min_area", type=int, default=25, help="Minimum mask area.")
    parser.add_argument("--max_images", type=int, default=2, help="Max images to infer, 0 means all.")
    parser.add_argument("--save_gt_panel", action="store_true", default=True, help="If GT exists, save panel with GT.")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm.")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def load_transform_v2_get_val_transform():
    base_dir = Path(__file__).resolve().parent
    candidates = [base_dir / "transform-v2.py", base_dir / "transforms_v2.py"]
    file_path = None
    for p in candidates:
        if p.exists():
            file_path = p
            break
    if file_path is None:
        raise FileNotFoundError(f"Cannot find transform-v2.py or transforms_v2.py in: {base_dir}")

    spec = importlib.util.spec_from_file_location("transform_v2_module", str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load transform module from: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_val_transform"):
        raise ImportError(f"Module does not provide get_val_transform: {file_path}")
    return mod.get_val_transform


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def infer_model_hparams_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    num_queries = int(state.get("mask_head.query_embed.weight", torch.empty(50, 256)).shape[0])
    hidden_dim = int(state.get("mask_head.query_embed.weight", torch.empty(50, 256)).shape[1])
    num_classes = int(state.get("mask_head.class_embed.weight", torch.empty(2, hidden_dim)).shape[0])
    mask_embed_dim = int(state.get("mask_head.pixel_embed.weight", torch.empty(256, hidden_dim, 1, 1)).shape[0])
    return {
        "num_queries": num_queries,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "mask_embed_dim": mask_embed_dim,
    }


def build_model_from_checkpoint(checkpoint_path: Path, input_size: int, device: torch.device):
    state = load_checkpoint_state(checkpoint_path)
    hp = infer_model_hparams_from_state(state)
    model = LeafInstanceSegModel(
        num_queries=hp["num_queries"],
        hidden_dim=hp["hidden_dim"],
        num_classes=hp["num_classes"],
        mask_embed_dim=hp["mask_embed_dim"],
        pretrained=False,
        input_size=input_size,
        upsample_masks_to_input=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(
        f"[INFO] Model hparams from ckpt: num_queries={hp['num_queries']}, hidden_dim={hp['hidden_dim']}, "
        f"num_classes={hp['num_classes']}, mask_embed_dim={hp['mask_embed_dim']}"
    )
    print(f"[INFO] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum(dtype=np.int64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.int64)
    return float(inter) / float(union + 1e-6)


def run_postprocess(
    pred_logits_qc: torch.Tensor,
    pred_masks_qhw: torch.Tensor,
    score_thresh: float,
    mask_thresh: float,
    nms_iou_thresh: float,
    contain_thresh: float,
    min_area: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    # leaf class id = 0, no-object class id = 1 (consistent with training setup)
    probs = F.softmax(pred_logits_qc, dim=-1)[:, 0].detach().cpu().numpy()
    masks_prob = pred_masks_qhw.sigmoid().detach().cpu().numpy()
    num_queries = int(probs.shape[0])

    counts = {"q_total": num_queries}

    # 1) score threshold
    kept = []
    for i in range(num_queries):
        s = float(probs[i])
        if s >= score_thresh:
            kept.append({"query_idx": i, "score": s, "mask_prob": masks_prob[i]})
    counts["after_score"] = len(kept)

    # 2) mask threshold
    for item in kept:
        m = item["mask_prob"] >= float(mask_thresh)
        item["mask_bin"] = m
        item["area"] = int(m.sum())
    kept = [x for x in kept if x["area"] > 0]
    counts["after_mask"] = len(kept)

    # 3) mask NMS by score desc
    kept.sort(key=lambda x: x["score"], reverse=True)
    nms_kept = []
    for cur in kept:
        keep_flag = True
        for prev in nms_kept:
            if mask_iou(cur["mask_bin"], prev["mask_bin"]) > float(nms_iou_thresh):
                keep_flag = False
                break
        if keep_flag:
            nms_kept.append(cur)
    kept = nms_kept
    counts["after_nms"] = len(kept)

    # 4) containment filter: remove small masks heavily contained by larger masks
    keep_flags = [True] * len(kept)
    for i in range(len(kept)):
        if not keep_flags[i]:
            continue
        area_i = max(1, kept[i]["area"])
        for j in range(len(kept)):
            if i == j or not keep_flags[i]:
                continue
            area_j = kept[j]["area"]
            if area_j <= area_i:
                continue
            inter = np.logical_and(kept[i]["mask_bin"], kept[j]["mask_bin"]).sum(dtype=np.int64)
            contain_ratio = float(inter) / float(area_i)
            if contain_ratio >= float(contain_thresh):
                keep_flags[i] = False
    kept = [k for k, f in zip(kept, keep_flags) if f]
    counts["after_contain"] = len(kept)

    # 5) min area
    kept = [x for x in kept if x["area"] >= int(min_area)]
    counts["after_min_area"] = len(kept)

    return kept, counts


def build_instance_map(final_instances: List[Dict], h: int, w: int) -> np.ndarray:
    inst_map = np.zeros((h, w), dtype=np.int32)
    # low score first, high score overwrite
    final_instances = sorted(final_instances, key=lambda x: x["score"])
    cur_id = 1
    for item in final_instances:
        inst_map[item["mask_bin"]] = cur_id
        cur_id += 1
    return inst_map


def id_to_color(idx: int):
    if idx <= 0:
        return (0, 0, 0)
    b = (37 * idx + 23) % 256
    g = (17 * idx + 91) % 256
    r = (97 * idx + 53) % 256
    return int(b), int(g), int(r)


def colorize_instance_map(inst_map: np.ndarray) -> np.ndarray:
    h, w = inst_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in np.unique(inst_map):
        i = int(idx)
        if i <= 0:
            continue
        out[inst_map == i] = id_to_color(i)
    return out


def make_panel(image_rgb: np.ndarray, pred_inst_map: np.ndarray, gt_inst_map: np.ndarray = None):
    pred_color = colorize_instance_map(pred_inst_map)
    pred_overlay = cv2.addWeighted(image_rgb, 0.65, pred_color, 0.35, 0.0)

    if gt_inst_map is None:
        panel = np.concatenate([image_rgb, pred_color, pred_overlay], axis=1)
        return panel

    gt_color = colorize_instance_map(gt_inst_map)
    panel = np.concatenate([image_rgb, pred_color, pred_overlay, gt_color], axis=1)
    return panel


def main():
    args = parse_args()
    device = select_device(args.device)

    save_dir = Path(args.save_dir)
    vis_dir = save_dir / "vis"
    raw_dir = save_dir / "raw"
    vis_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    get_val_transform = load_transform_v2_get_val_transform()
    val_transform = get_val_transform(target_size=args.input_size)
    model = build_model_from_checkpoint(checkpoint_path, input_size=args.input_size, device=device)

    all_samples = collect_samples(args.roots, skip_empty=False)
    if len(all_samples) == 0:
        raise RuntimeError(f"No samples found from roots: {args.roots}")

    if args.split == "all":
        samples = all_samples
    else:
        train_samples, val_samples = split_samples_by_big_image(
            samples=all_samples,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        samples = train_samples if args.split == "train" else val_samples

    if len(samples) == 0:
        raise RuntimeError(f"No samples left after split='{args.split}' from roots: {args.roots}")

    if args.max_images > 0:
        samples = samples[: args.max_images]

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Split: {args.split}")
    if args.split != "all":
        print(f"[INFO] Split config: val_ratio={args.val_ratio}, seed={args.seed}")
        print(f"[INFO] Total collected samples: {len(all_samples)}")
    print(f"[INFO] Infer samples: {len(samples)}")
    print(
        f"[INFO] Thresholds: score={args.score_thresh}, mask={args.mask_thresh}, "
        f"nms_iou={args.nms_iou_thresh}, contain={args.contain_thresh}, min_area={args.min_area}"
    )

    summary = []
    progress = tqdm(samples, disable=bool(args.no_tqdm), dynamic_ncols=True, desc="Predict-v1")
    with torch.no_grad():
        for sample in progress:
            image_path = Path(sample["image_path"])
            stem = image_path.stem
            image_rgb = load_image(image_path)
            h0, w0 = image_rgb.shape[:2]

            # Use transform-v2 get_val_transform (expects image + masks)
            dummy_mask = np.zeros((h0, w0), dtype=np.uint8)
            transformed = val_transform(image_rgb, dummy_mask, dummy_mask)
            image_t = transformed["image"].unsqueeze(0).to(device)

            outputs = model(image_t)
            pred_logits = outputs["pred_logits"][0]  # [Q, C]
            pred_masks = outputs["pred_masks"][0]  # [Q, H, W] (upsampled to input)

            final_instances, counts = run_postprocess(
                pred_logits_qc=pred_logits,
                pred_masks_qhw=pred_masks,
                score_thresh=args.score_thresh,
                mask_thresh=args.mask_thresh,
                nms_iou_thresh=args.nms_iou_thresh,
                contain_thresh=args.contain_thresh,
                min_area=args.min_area,
            )

            # Resize binary masks back to original resolution for visualization if input_size changed.
            if pred_masks.shape[-2:] != (h0, w0):
                for item in final_instances:
                    m = item["mask_bin"].astype(np.uint8) * 255
                    m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    item["mask_bin"] = m > 0
                    item["area"] = int(item["mask_bin"].sum())

            pred_inst_map = build_instance_map(final_instances, h=h0, w=w0)

            gt_inst_map = None
            if args.save_gt_panel:
                inst_path = Path(sample.get("instance_path", ""))
                if inst_path.exists():
                    try:
                        gt_inst_map = load_instance_map(inst_path)
                    except Exception:
                        gt_inst_map = None

            panel = make_panel(image_rgb=image_rgb, pred_inst_map=pred_inst_map, gt_inst_map=gt_inst_map)
            panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_dir / f"{stem}_panel.png"), panel_bgr)
            cv2.imwrite(str(raw_dir / f"{stem}_pred_instance.png"), pred_inst_map.astype(np.uint16))

            print(
                f"[POST] {image_path.name} | total={counts['q_total']} -> "
                f"score={counts['after_score']} -> mask={counts['after_mask']} -> "
                f"nms={counts['after_nms']} -> contain={counts['after_contain']} -> "
                f"min_area={counts['after_min_area']}"
            )

            summary.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "q_total": counts["q_total"],
                    "after_score": counts["after_score"],
                    "after_mask": counts["after_mask"],
                    "after_nms": counts["after_nms"],
                    "after_contain": counts["after_contain"],
                    "after_min_area": counts["after_min_area"],
                    "pred_instances": int(np.max(pred_inst_map)),
                }
            )

    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
