#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import collect_samples, load_image, load_instance_map, split_samples_by_big_image  # noqa: E402
from model import LeafInstanceSegModel  # noqa: E402
from transforms_v2 import get_val_transform  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build baseline review pack for human-in-the-loop visual evaluation.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["data/patches_size512", "data/patches_size768", "data/patches_size1024"],
        help="Patch roots.",
    )
    parser.add_argument("--checkpoint", type=str, default="outputs/exp_20260308_171807/checkpoints/best.pth")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="outputs/review_pack_baseline")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=24)
    parser.add_argument("--score_thresh", type=float, default=0.35)
    parser.add_argument("--mask_thresh", type=float, default=0.40)
    parser.add_argument("--nms_iou_thresh", type=float, default=0.65)
    parser.add_argument("--contain_thresh", type=float, default=0.75)
    parser.add_argument("--min_area", type=int, default=25)
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
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


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
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"[INFO] Auxiliary heads in ckpt: {has_aux}")
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
    probs = F.softmax(pred_logits_qc, dim=-1)[:, 0].detach().cpu().numpy()
    masks_prob = pred_masks_qhw.sigmoid().detach().cpu().numpy()
    num_queries = int(probs.shape[0])

    counts = {"q_total": num_queries}
    kept = []
    for i in range(num_queries):
        score = float(probs[i])
        if score >= score_thresh:
            kept.append({"query_idx": i, "score": score, "mask_prob": masks_prob[i]})
    counts["after_score"] = len(kept)

    for item in kept:
        mask_bin = item["mask_prob"] >= float(mask_thresh)
        item["mask_bin"] = mask_bin
        item["area"] = int(mask_bin.sum())
    kept = [x for x in kept if x["area"] > 0]
    counts["after_mask"] = len(kept)

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


def make_panel(image_rgb: np.ndarray, pred_inst_map: np.ndarray, gt_inst_map: np.ndarray):
    gt_color = colorize_instance_map(gt_inst_map)
    pred_color = colorize_instance_map(pred_inst_map)
    overlay = cv2.addWeighted(image_rgb, 0.65, pred_color, 0.35, 0.0)
    panel = np.concatenate([image_rgb, gt_color, pred_color, overlay], axis=1)
    return panel


def compute_sample_difficulty(sample: Dict) -> Dict:
    image_rgb = load_image(sample["image_path"])
    inst_map = load_instance_map(sample["instance_path"])
    h, w = inst_map.shape

    ids = np.unique(inst_map)
    ids = ids[ids > 0]
    inst_count = int(len(ids))
    fg_ratio = float((inst_map > 0).mean())

    touch_count = 0
    for iid in ids.tolist():
        m = inst_map == int(iid)
        if m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any():
            touch_count += 1
    touch_ratio = float(touch_count / max(1, inst_count))

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    _ = h_ch
    highlight = np.logical_and(v_ch >= 210, s_ch <= 55)
    highlight_ratio = float(highlight.mean())

    return {
        "height": int(h),
        "width": int(w),
        "inst_count": inst_count,
        "fg_ratio": fg_ratio,
        "touch_ratio": touch_ratio,
        "highlight_ratio": highlight_ratio,
    }


def select_review_samples(samples: List[Dict], metric_rows: List[Dict], total: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    if total <= 0:
        raise ValueError("num_samples must be > 0")
    if total > len(samples):
        total = len(samples)

    for sample, metrics in zip(samples, metric_rows):
        sample["difficulty"] = metrics

    selected = []
    selected_paths = set()

    def try_add(candidates: List[Dict], bucket: str, n: int):
        added = 0
        for s in candidates:
            p = str(s["image_path"])
            if p in selected_paths:
                continue
            new_item = dict(s)
            new_item["bucket"] = bucket
            selected.append(new_item)
            selected_paths.add(p)
            added += 1
            if added >= n:
                break

    for scale in [512, 768, 1024]:
        scale_pool = [s for s in samples if int(s.get("patch_scale", -1)) == scale]
        scale_pool.sort(
            key=lambda x: (
                -x["difficulty"]["inst_count"],
                -x["difficulty"]["touch_ratio"],
                -x["difficulty"]["highlight_ratio"],
                str(x["image_path"]),
            )
        )
        try_add(scale_pool, f"scale_anchor_{scale}", 2)

    remain = max(0, total - len(selected))
    each = remain // 3
    rem_last = remain - each * 2

    dense = sorted(
        samples,
        key=lambda x: (
            -x["difficulty"]["inst_count"],
            -x["difficulty"]["touch_ratio"],
            str(x["image_path"]),
        ),
    )
    boundary = sorted(
        samples,
        key=lambda x: (
            -x["difficulty"]["touch_ratio"],
            -x["difficulty"]["inst_count"],
            str(x["image_path"]),
        ),
    )
    highlight = sorted(
        samples,
        key=lambda x: (
            -x["difficulty"]["highlight_ratio"],
            -x["difficulty"]["inst_count"],
            str(x["image_path"]),
        ),
    )

    try_add(dense, "dense", each)
    try_add(boundary, "boundary", each)
    try_add(highlight, "highlight", rem_last)

    if len(selected) < total:
        remained = [s for s in samples if str(s["image_path"]) not in selected_paths]
        rng.shuffle(remained)
        try_add(remained, "random_fill", total - len(selected))

    selected = selected[:total]
    selected.sort(key=lambda x: (x["bucket"], str(x["image_path"])))
    return selected


def write_manifest_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "bucket"])
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scoring_template(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "image_name", "bucket", "score_1to5", "winner_A_or_B", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "index": r["index"],
                    "image_name": r["image_name"],
                    "bucket": r["bucket"],
                    "score_1to5": "",
                    "winner_A_or_B": "",
                    "notes": "",
                }
            )


def main():
    args = parse_args()
    device = select_device(args.device)

    save_dir = Path(args.save_dir)
    panel_dir = save_dir / "panels"
    raw_dir = save_dir / "pred_instance"
    save_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
        raise RuntimeError(f"No samples left after split={args.split}")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Total candidates: {len(samples)}")
    print(f"[INFO] Building difficulty stats...")
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

    metric_rows = []
    metric_iter = tqdm(samples, disable=bool(args.no_tqdm), dynamic_ncols=True, desc="Compute difficulty")
    for sample in metric_iter:
        metric_rows.append(compute_sample_difficulty(sample))

    selected = select_review_samples(samples, metric_rows, total=args.num_samples, seed=args.seed)
    print(f"[INFO] Selected review samples: {len(selected)}")

    model = build_model_from_checkpoint(checkpoint_path, input_size=args.input_size, device=device)
    val_transform = get_val_transform(target_size=args.input_size)

    manifest_rows = []
    summary_rows = []

    infer_iter = tqdm(selected, disable=bool(args.no_tqdm), dynamic_ncols=True, desc="Build review panels")
    with torch.no_grad():
        for idx, sample in enumerate(infer_iter, start=1):
            image_path = Path(sample["image_path"])
            stem = image_path.stem
            bucket = str(sample["bucket"])

            image_rgb = load_image(image_path)
            gt_inst_map = load_instance_map(sample["instance_path"])
            h0, w0 = image_rgb.shape[:2]

            dummy_mask = np.zeros((h0, w0), dtype=np.uint8)
            transformed = val_transform(image_rgb, dummy_mask, dummy_mask)
            image_t = transformed["image"].unsqueeze(0).to(device)

            outputs = model(image_t)
            pred_logits = outputs["pred_logits"][0]
            pred_masks = outputs["pred_masks"][0]
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
            panel = make_panel(image_rgb=image_rgb, pred_inst_map=pred_inst_map, gt_inst_map=gt_inst_map)
            panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)

            d = sample["difficulty"]
            text = (
                f"#{idx:02d} [{bucket}] {image_path.name} | scale={sample['patch_scale']} "
                f"| gt_inst={d['inst_count']} | touch={d['touch_ratio']:.2f} "
                f"| highlight={d['highlight_ratio']:.3f} | pred_inst={int(np.max(pred_inst_map))} "
                f"| stitched={stitched_pairs}"
            )
            cv2.rectangle(panel_bgr, (0, 0), (panel_bgr.shape[1], 32), (18, 18, 18), thickness=-1)
            cv2.putText(panel_bgr, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

            out_name = f"{idx:02d}_{bucket}_{stem}_panel.png"
            cv2.imwrite(str(panel_dir / out_name), panel_bgr)
            cv2.imwrite(str(raw_dir / f"{idx:02d}_{bucket}_{stem}_pred_instance.png"), pred_inst_map.astype(np.uint16))

            manifest_rows.append(
                {
                    "index": idx,
                    "bucket": bucket,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "instance_path": str(sample["instance_path"]),
                    "patch_scale": int(sample["patch_scale"]),
                    "gt_inst_count": int(d["inst_count"]),
                    "gt_fg_ratio": float(d["fg_ratio"]),
                    "gt_touch_ratio": float(d["touch_ratio"]),
                    "highlight_ratio": float(d["highlight_ratio"]),
                }
            )

            summary_rows.append(
                {
                    "index": idx,
                    "bucket": bucket,
                    "image_name": image_path.name,
                    "patch_scale": int(sample["patch_scale"]),
                    "gt_inst_count": int(d["inst_count"]),
                    "pred_inst_count": int(np.max(pred_inst_map)),
                    "q_total": counts["q_total"],
                    "after_score": counts["after_score"],
                    "after_mask": counts["after_mask"],
                    "after_nms": counts["after_nms"],
                    "after_contain": counts["after_contain"],
                    "after_min_area": counts["after_min_area"],
                    "stitched_pairs": int(stitched_pairs),
                }
            )

    write_manifest_csv(save_dir / "review_manifest.csv", manifest_rows)
    write_manifest_csv(save_dir / "review_summary.csv", summary_rows)
    write_scoring_template(save_dir / "review_scoring_template.csv", manifest_rows)

    with (save_dir / "review_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, ensure_ascii=False, indent=2)
    with (save_dir / "review_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    protocol = {
        "date": "2026-03-16",
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "thresholds": {
            "score_thresh": args.score_thresh,
            "mask_thresh": args.mask_thresh,
            "nms_iou_thresh": args.nms_iou_thresh,
            "contain_thresh": args.contain_thresh,
            "min_area": args.min_area,
            "enable_center_stitching": bool(args.enable_center_stitching),
            "center_merge_dist": args.center_merge_dist,
            "edge_gap_dist": args.edge_gap_dist,
            "center_sample_min": args.center_sample_min,
            "center_spread_max": args.center_spread_max,
            "stitch_use_center_peak": bool(args.stitch_use_center_peak),
            "center_peak_thresh": args.center_peak_thresh,
            "center_peak_min_dist": args.center_peak_min_dist,
            "peak_assign_dist": args.peak_assign_dist,
            "axis_sim_thresh": args.axis_sim_thresh,
            "connect_align_thresh": args.connect_align_thresh,
        },
        "selection": {
            "num_samples": len(manifest_rows),
            "seed": args.seed,
            "buckets": sorted(list({row["bucket"] for row in manifest_rows})),
        },
        "panel_layout": "Input | GT Instance | Pred Instance | Pred Overlay",
    }
    with (save_dir / "review_protocol.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Review pack saved to: {save_dir}")
    print(f"[INFO] Panels: {panel_dir}")
    print(f"[INFO] Manifest CSV: {save_dir / 'review_manifest.csv'}")
    print(f"[INFO] Scoring template: {save_dir / 'review_scoring_template.csv'}")


if __name__ == "__main__":
    main()
