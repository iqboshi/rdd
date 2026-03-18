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
    parser.add_argument("--enable_center_stitching", action="store_true", help="Merge split masks via offset-predicted centers.")
    parser.add_argument("--center_merge_dist", type=float, default=26.0, help="Center distance threshold for stitching.")
    parser.add_argument("--edge_gap_dist", type=float, default=22.0, help="Max edge distance between masks for stitching.")
    parser.add_argument("--center_sample_min", type=int, default=30, help="Min pixels in a mask to estimate offset center.")
    parser.add_argument("--center_spread_max", type=float, default=14.0, help="Max median center-vote spread; larger means less reliable center.")
    parser.add_argument("--stitch_use_center_peak", action="store_true", help="Require center-peak consistency when center head exists (A-v3).")
    parser.add_argument("--center_peak_thresh", type=float, default=0.35, help="Center heatmap peak threshold for peak-consistency stitching.")
    parser.add_argument("--center_peak_min_dist", type=int, default=6, help="Minimum distance (px) between center peaks.")
    parser.add_argument("--peak_assign_dist", type=float, default=24.0, help="Max distance from instance center-vote to assigned center peak.")
    parser.add_argument("--axis_sim_thresh", type=float, default=0.45, help="Min cosine similarity of mask major axes for stitching.")
    parser.add_argument("--connect_align_thresh", type=float, default=0.35, help="Min alignment between major axis and pair-connection direction.")
    parser.add_argument("--enable_center_split", action="store_true", help="Split potentially merged instances using center peaks + offset votes.")
    parser.add_argument("--split_peak_thresh", type=float, default=0.42, help="Center peak threshold for merge-splitting.")
    parser.add_argument("--split_peak_min_dist", type=int, default=10, help="Minimum center-peak distance for merge-splitting.")
    parser.add_argument("--split_vote_radius", type=float, default=12.0, help="Vote radius used to validate split peaks.")
    parser.add_argument("--split_peak_min_votes", type=int, default=90, help="Minimum vote support per split peak.")
    parser.add_argument("--split_assign_radius", type=float, default=28.0, help="Vote-distance radius for assigning pixels to peaks.")
    parser.add_argument("--split_min_pixels", type=int, default=110, help="Minimum pixels per split part.")
    parser.add_argument("--split_second_min_pixels", type=int, default=70, help="Second-largest split part must exceed this.")
    parser.add_argument("--split_second_support_ratio", type=float, default=0.18, help="Second-largest peak support ratio threshold.")
    parser.add_argument("--split_min_peak_separation", type=float, default=20.0, help="Minimum separation between kept peaks.")
    parser.add_argument(
        "--split_max_elongation",
        type=float,
        default=2.8,
        help="Only split instances with elongation <= this value (to avoid splitting long single leaves).",
    )
    parser.add_argument("--split_max_parts", type=int, default=4, help="Maximum number of parts kept when splitting one instance.")
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


def checkpoint_has_aux_heads(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(k).startswith("aux_head.") for k in state.keys())


def build_model_from_checkpoint(checkpoint_path: Path, input_size: int, device: torch.device):
    state = load_checkpoint_state(checkpoint_path)
    hp = infer_model_hparams_from_state(state)
    has_aux = checkpoint_has_aux_heads(state)
    model = LeafInstanceSegModel(
        num_queries=hp["num_queries"],
        hidden_dim=hp["hidden_dim"],
        num_classes=hp["num_classes"],
        mask_embed_dim=hp["mask_embed_dim"],
        pretrained=False,
        input_size=input_size,
        upsample_masks_to_input=True,
        enable_aux_heads=has_aux,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(
        f"[INFO] Model hparams from ckpt: num_queries={hp['num_queries']}, hidden_dim={hp['hidden_dim']}, "
        f"num_classes={hp['num_classes']}, mask_embed_dim={hp['mask_embed_dim']}"
    )
    print(f"[INFO] Auxiliary heads in ckpt: {has_aux}")
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


def min_edge_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if np.logical_and(mask_a, mask_b).any():
        return 0.0
    inv = (~mask_a).astype(np.uint8)
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    vals = dist[mask_b]
    if vals.size == 0:
        return float("inf")
    return float(vals.min())


def estimate_center_from_offset(mask_bin: np.ndarray, pred_offset_hw2: np.ndarray, center_sample_min: int = 30):
    ys, xs = np.where(mask_bin)
    if ys.size < int(center_sample_min):
        return None
    dy = pred_offset_hw2[ys, xs, 0].astype(np.float32)
    dx = pred_offset_hw2[ys, xs, 1].astype(np.float32)
    cy_votes = ys.astype(np.float32) + dy
    cx_votes = xs.astype(np.float32) + dx
    cx = float(np.median(cx_votes))
    cy = float(np.median(cy_votes))
    spread = float(np.median(np.sqrt((cx_votes - cx) ** 2 + (cy_votes - cy) ** 2)))
    return {
        "center": (cx, cy),
        "spread": spread,
        "num_votes": int(cx_votes.size),
    }


def extract_mask_axis(mask_bin: np.ndarray):
    ys, xs = np.where(mask_bin)
    if ys.size < 8:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    centroid = pts.mean(axis=0)
    centered = pts - centroid[None, :]
    denom = max(int(centered.shape[0] - 1), 1)
    cov = (centered.T @ centered) / float(denom)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major_val = float(eigvals[order[0]])
    minor_val = float(eigvals[order[1]])
    axis = eigvecs[:, order[0]].astype(np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return None
    axis = axis / axis_norm
    elongation = float(np.sqrt((major_val + 1e-6) / (minor_val + 1e-6)))
    return {
        "centroid": (float(centroid[0]), float(centroid[1])),
        "axis": (float(axis[0]), float(axis[1])),
        "elongation": elongation,
    }


def detect_center_peaks(center_prob_hw: np.ndarray, peak_thresh: float = 0.35, min_dist: int = 6, max_peaks: int = 128):
    if center_prob_hw is None:
        return []
    center_prob_hw = np.asarray(center_prob_hw, dtype=np.float32)
    if center_prob_hw.ndim != 2:
        return []

    h, w = center_prob_hw.shape
    min_dist = max(1, int(min_dist))
    k = int(2 * min_dist + 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    local_max = cv2.dilate(center_prob_hw, kernel) == center_prob_hw
    cand = np.logical_and(local_max, center_prob_hw >= float(peak_thresh))
    ys, xs = np.where(cand)
    if ys.size == 0:
        return []

    order = np.argsort(center_prob_hw[ys, xs])[::-1]
    occupied = np.zeros((h, w), dtype=np.uint8)
    peaks = []
    for idx in order.tolist():
        y = int(ys[idx])
        x = int(xs[idx])
        if occupied[y, x] != 0:
            continue
        peaks.append((float(x), float(y), float(center_prob_hw[y, x])))
        cv2.circle(occupied, (x, y), int(min_dist), color=1, thickness=-1)
        if len(peaks) >= int(max_peaks):
            break
    return peaks


def assign_center_peak(center_xy, peaks, assign_dist: float = 24.0):
    if center_xy is None or len(peaks) == 0:
        return -1, float("inf")
    cx, cy = center_xy
    best_id = -1
    best_dist = float("inf")
    for i, (px, py, _score) in enumerate(peaks):
        d = float(np.hypot(cx - px, cy - py))
        if d < best_dist:
            best_dist = d
            best_id = int(i)
    if best_dist <= float(assign_dist):
        return best_id, best_dist
    return -1, best_dist


def extract_largest_component(mask_bin: np.ndarray) -> np.ndarray:
    m = mask_bin.astype(np.uint8)
    if int(m.sum()) == 0:
        return mask_bin.astype(bool)
    num_labels, labels = cv2.connectedComponents(m, connectivity=8)
    if num_labels <= 2:
        return mask_bin.astype(bool)
    best_label = 1
    best_area = 0
    for lid in range(1, int(num_labels)):
        area = int((labels == lid).sum())
        if area > best_area:
            best_area = area
            best_label = lid
    return labels == int(best_label)


def split_instances_by_center_peaks(
    final_instances: List[Dict],
    pred_offset_hw2: np.ndarray,
    pred_center_hw: np.ndarray,
    split_peak_thresh: float = 0.42,
    split_peak_min_dist: int = 10,
    split_vote_radius: float = 12.0,
    split_peak_min_votes: int = 90,
    split_assign_radius: float = 28.0,
    split_min_pixels: int = 110,
    split_second_min_pixels: int = 70,
    split_second_support_ratio: float = 0.18,
    split_min_peak_separation: float = 20.0,
    split_max_elongation: float = 2.8,
    split_max_parts: int = 4,
) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {"split_masks": 0, "split_added_instances": 0}
    if len(final_instances) == 0:
        return final_instances, stats
    if pred_offset_hw2 is None or pred_center_hw is None:
        return final_instances, stats

    all_peaks = detect_center_peaks(
        center_prob_hw=pred_center_hw,
        peak_thresh=split_peak_thresh,
        min_dist=split_peak_min_dist,
        max_peaks=512,
    )
    if len(all_peaks) < 2:
        return final_instances, stats

    split_min_pixels = int(split_min_pixels)
    split_second_min_pixels = int(split_second_min_pixels)
    split_peak_min_votes = int(split_peak_min_votes)
    split_max_parts = max(2, int(split_max_parts))

    out_instances = []
    for item in final_instances:
        mask_bin = item["mask_bin"]
        ys, xs = np.where(mask_bin)
        if ys.size < max(split_min_pixels * 2, split_peak_min_votes):
            out_instances.append(item)
            continue

        if float(split_max_elongation) > 0:
            axis_info = extract_mask_axis(mask_bin)
            if axis_info is not None and float(axis_info["elongation"]) > float(split_max_elongation):
                out_instances.append(item)
                continue

        peaks_in_mask = []
        h, w = mask_bin.shape
        for px, py, ps in all_peaks:
            xi = int(round(px))
            yi = int(round(py))
            if yi < 0 or yi >= h or xi < 0 or xi >= w:
                continue
            if mask_bin[yi, xi]:
                peaks_in_mask.append((float(px), float(py), float(ps)))
        if len(peaks_in_mask) < 2:
            out_instances.append(item)
            continue

        peak_xy = np.asarray([[p[0], p[1]] for p in peaks_in_mask], dtype=np.float32)
        dy = pred_offset_hw2[ys, xs, 0].astype(np.float32)
        dx = pred_offset_hw2[ys, xs, 1].astype(np.float32)
        vote_xy = np.stack([xs.astype(np.float32) + dx, ys.astype(np.float32) + dy], axis=1)

        dist_vote_to_peak = np.linalg.norm(vote_xy[:, None, :] - peak_xy[None, :, :], axis=2)
        support = (dist_vote_to_peak <= float(split_vote_radius)).sum(axis=0).astype(np.int32)
        valid = np.where(support >= split_peak_min_votes)[0]
        if valid.size < 2:
            out_instances.append(item)
            continue

        order = np.argsort(support[valid])[::-1]
        keep = valid[order[:split_max_parts]]
        peak_xy = peak_xy[keep]
        support = support[keep]
        if peak_xy.shape[0] < 2:
            out_instances.append(item)
            continue

        if float(split_min_peak_separation) > 0:
            sep_ok = False
            for i in range(int(peak_xy.shape[0])):
                for j in range(i + 1, int(peak_xy.shape[0])):
                    d = float(np.hypot(peak_xy[i, 0] - peak_xy[j, 0], peak_xy[i, 1] - peak_xy[j, 1]))
                    if d >= float(split_min_peak_separation):
                        sep_ok = True
                        break
                if sep_ok:
                    break
            if not sep_ok:
                out_instances.append(item)
                continue

        sorted_support = np.sort(support)
        if int(sorted_support[-2]) < split_second_min_pixels:
            out_instances.append(item)
            continue
        second_ratio = float(sorted_support[-2]) / float(max(1, ys.size))
        if second_ratio < float(split_second_support_ratio):
            out_instances.append(item)
            continue

        dist_keep = np.linalg.norm(vote_xy[:, None, :] - peak_xy[None, :, :], axis=2)
        assign = dist_keep.argmin(axis=1).astype(np.int32)
        nearest = dist_keep[np.arange(dist_keep.shape[0]), assign]

        far = nearest > float(split_assign_radius)
        if np.any(far):
            pix_xy = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
            dist_pix = np.linalg.norm(pix_xy[far, None, :] - peak_xy[None, :, :], axis=2)
            assign[far] = dist_pix.argmin(axis=1).astype(np.int32)

        parts = []
        for k in range(int(peak_xy.shape[0])):
            sel = assign == k
            if int(sel.sum()) < split_min_pixels:
                continue
            part_mask = np.zeros_like(mask_bin, dtype=bool)
            part_mask[ys[sel], xs[sel]] = True
            part_mask = extract_largest_component(part_mask)
            area = int(part_mask.sum())
            if area < split_min_pixels:
                continue
            parts.append(
                {
                    "query_idx": int(item["query_idx"]),
                    "score": float(item["score"]) * 0.995,
                    "mask_prob": None,
                    "mask_bin": part_mask,
                    "area": area,
                }
            )

        if len(parts) < 2:
            out_instances.append(item)
            continue

        areas = sorted([int(p["area"]) for p in parts], reverse=True)
        if int(areas[1]) < split_second_min_pixels:
            out_instances.append(item)
            continue

        stats["split_masks"] += 1
        stats["split_added_instances"] += (len(parts) - 1)
        out_instances.extend(parts)

    return out_instances, stats


def stitch_instances_by_center(
    final_instances: List[Dict],
    pred_offset_hw2: np.ndarray,
    pred_center_hw: np.ndarray = None,
    center_merge_dist: float = 26.0,
    edge_gap_dist: float = 22.0,
    center_sample_min: int = 30,
    center_spread_max: float = 14.0,
    stitch_use_center_peak: bool = False,
    center_peak_thresh: float = 0.35,
    center_peak_min_dist: int = 6,
    peak_assign_dist: float = 24.0,
    axis_sim_thresh: float = 0.45,
    connect_align_thresh: float = 0.35,
) -> Tuple[List[Dict], int]:
    if len(final_instances) <= 1:
        return final_instances, 0

    center_infos = [
        estimate_center_from_offset(
            item["mask_bin"],
            pred_offset_hw2=pred_offset_hw2,
            center_sample_min=center_sample_min,
        )
        for item in final_instances
    ]
    axis_infos = [extract_mask_axis(item["mask_bin"]) for item in final_instances]

    peaks = []
    peak_ids = [-1] * len(final_instances)
    use_peak_consistency = bool(stitch_use_center_peak and pred_center_hw is not None)
    if use_peak_consistency:
        peaks = detect_center_peaks(
            center_prob_hw=pred_center_hw,
            peak_thresh=center_peak_thresh,
            min_dist=center_peak_min_dist,
        )
        for i, info in enumerate(center_infos):
            if info is None:
                continue
            if float(center_spread_max) > 0 and float(info["spread"]) > float(center_spread_max):
                continue
            peak_id, _ = assign_center_peak(info["center"], peaks, assign_dist=peak_assign_dist)
            peak_ids[i] = int(peak_id)

    parent = list(range(len(final_instances)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(final_instances)):
        info_i = center_infos[i]
        if info_i is None:
            continue
        if float(center_spread_max) > 0 and float(info_i["spread"]) > float(center_spread_max):
            continue
        center_i = info_i["center"]
        axis_i = axis_infos[i]
        for j in range(i + 1, len(final_instances)):
            info_j = center_infos[j]
            if info_j is None:
                continue
            if float(center_spread_max) > 0 and float(info_j["spread"]) > float(center_spread_max):
                continue
            center_j = info_j["center"]
            axis_j = axis_infos[j]

            d_center = np.hypot(center_i[0] - center_j[0], center_i[1] - center_j[1])
            if d_center > float(center_merge_dist):
                continue
            d_edge = min_edge_distance(final_instances[i]["mask_bin"], final_instances[j]["mask_bin"])
            if d_edge > float(edge_gap_dist):
                continue

            if use_peak_consistency:
                peak_i = peak_ids[i]
                peak_j = peak_ids[j]
                if peak_i >= 0 or peak_j >= 0:
                    if peak_i != peak_j:
                        continue

            if float(axis_sim_thresh) > 0 and axis_i is not None and axis_j is not None:
                vec_i = np.asarray(axis_i["axis"], dtype=np.float32)
                vec_j = np.asarray(axis_j["axis"], dtype=np.float32)
                axis_sim = float(abs(np.dot(vec_i, vec_j)))
                if axis_sim < float(axis_sim_thresh):
                    continue

            if float(connect_align_thresh) > 0 and (axis_i is not None or axis_j is not None):
                pi = np.asarray(axis_i["centroid"] if axis_i is not None else center_i, dtype=np.float32)
                pj = np.asarray(axis_j["centroid"] if axis_j is not None else center_j, dtype=np.float32)
                link = pj - pi
                norm = float(np.linalg.norm(link))
                if norm > 1e-6:
                    link = link / norm
                    aligns = []
                    if axis_i is not None:
                        aligns.append(float(abs(np.dot(np.asarray(axis_i["axis"], dtype=np.float32), link))))
                    if axis_j is not None:
                        aligns.append(float(abs(np.dot(np.asarray(axis_j["axis"], dtype=np.float32), link))))
                    if len(aligns) > 0 and max(aligns) < float(connect_align_thresh):
                        continue
            union(i, j)

    groups = {}
    for idx in range(len(final_instances)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    merged = []
    stitched_pairs = 0
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(final_instances[idxs[0]])
            continue
        stitched_pairs += len(idxs) - 1
        mask_union = np.zeros_like(final_instances[idxs[0]]["mask_bin"], dtype=bool)
        score = 0.0
        qidx = -1
        for k in idxs:
            mask_union = np.logical_or(mask_union, final_instances[k]["mask_bin"])
            if float(final_instances[k]["score"]) >= score:
                score = float(final_instances[k]["score"])
                qidx = int(final_instances[k]["query_idx"])
        merged.append(
            {
                "query_idx": qidx,
                "score": score,
                "mask_prob": None,
                "mask_bin": mask_union,
                "area": int(mask_union.sum()),
            }
        )
    return merged, stitched_pairs


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
    print(
        f"[INFO] Center stitching: enable={bool(args.enable_center_stitching)}, "
        f"center_merge_dist={args.center_merge_dist}, edge_gap_dist={args.edge_gap_dist}, "
        f"center_sample_min={args.center_sample_min}, center_spread_max={args.center_spread_max}"
    )
    print(
        f"[INFO] A3 stitching constraints: peak={bool(args.stitch_use_center_peak)}, "
        f"peak_thresh={args.center_peak_thresh}, peak_min_dist={args.center_peak_min_dist}, "
        f"peak_assign_dist={args.peak_assign_dist}, axis_sim_thresh={args.axis_sim_thresh}, "
        f"connect_align_thresh={args.connect_align_thresh}"
    )
    print(
        f"[INFO] Center split: enable={bool(args.enable_center_split)}, "
        f"peak_thresh={args.split_peak_thresh}, peak_min_dist={args.split_peak_min_dist}, "
        f"vote_radius={args.split_vote_radius}, peak_min_votes={args.split_peak_min_votes}, "
        f"assign_radius={args.split_assign_radius}, split_min_pixels={args.split_min_pixels}, "
        f"split_second_min_pixels={args.split_second_min_pixels}, "
        f"split_second_support_ratio={args.split_second_support_ratio}, "
        f"split_min_peak_separation={args.split_min_peak_separation}, "
        f"split_max_elongation={args.split_max_elongation}, "
        f"split_max_parts={args.split_max_parts}"
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
            pred_offset = None
            pred_center = None
            if "pred_offset" in outputs:
                pred_offset = outputs["pred_offset"][0].detach().cpu().numpy().transpose(1, 2, 0)
            if "pred_center" in outputs:
                pred_center = outputs["pred_center"][0, 0].sigmoid().detach().cpu().numpy()

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
                if pred_offset is not None:
                    pred_offset = cv2.resize(pred_offset, (w0, h0), interpolation=cv2.INTER_LINEAR)
                if pred_center is not None:
                    pred_center = cv2.resize(pred_center, (w0, h0), interpolation=cv2.INTER_LINEAR)

            stitched_pairs = 0
            split_stats = {"split_masks": 0, "split_added_instances": 0}
            if bool(args.enable_center_split):
                if pred_offset is None or pred_center is None:
                    print("[WARN] Center split requested but checkpoint lacks pred_offset/pred_center head. Skipped.")
                else:
                    final_instances, split_stats = split_instances_by_center_peaks(
                        final_instances=final_instances,
                        pred_offset_hw2=pred_offset,
                        pred_center_hw=pred_center,
                        split_peak_thresh=args.split_peak_thresh,
                        split_peak_min_dist=args.split_peak_min_dist,
                        split_vote_radius=args.split_vote_radius,
                        split_peak_min_votes=args.split_peak_min_votes,
                        split_assign_radius=args.split_assign_radius,
                        split_min_pixels=args.split_min_pixels,
                        split_second_min_pixels=args.split_second_min_pixels,
                        split_second_support_ratio=args.split_second_support_ratio,
                        split_min_peak_separation=args.split_min_peak_separation,
                        split_max_elongation=args.split_max_elongation,
                        split_max_parts=args.split_max_parts,
                    )
            if bool(args.enable_center_stitching):
                if pred_offset is None:
                    print("[WARN] Center stitching requested but checkpoint has no pred_offset head. Skipped.")
                else:
                    final_instances, stitched_pairs = stitch_instances_by_center(
                        final_instances,
                        pred_offset_hw2=pred_offset,
                        pred_center_hw=pred_center,
                        center_merge_dist=args.center_merge_dist,
                        edge_gap_dist=args.edge_gap_dist,
                        center_sample_min=args.center_sample_min,
                        center_spread_max=args.center_spread_max,
                        stitch_use_center_peak=bool(args.stitch_use_center_peak),
                        center_peak_thresh=args.center_peak_thresh,
                        center_peak_min_dist=args.center_peak_min_dist,
                        peak_assign_dist=args.peak_assign_dist,
                        axis_sim_thresh=args.axis_sim_thresh,
                        connect_align_thresh=args.connect_align_thresh,
                    )

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
                f"min_area={counts['after_min_area']} | split_add={split_stats['split_added_instances']} "
                f"| stitched_pairs={stitched_pairs}"
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
                    "split_masks": int(split_stats["split_masks"]),
                    "split_added_instances": int(split_stats["split_added_instances"]),
                    "stitched_pairs": int(stitched_pairs),
                    "pred_instances": int(np.max(pred_inst_map)),
                }
            )

    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
