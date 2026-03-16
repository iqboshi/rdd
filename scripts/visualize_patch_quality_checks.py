#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
NUMPY_EXTS = [".npy"]
ANY_EXTS = NUMPY_EXTS + IMAGE_EXTS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample patch data and save 6 quality-check visualizations for each sample."
    )
    parser.add_argument("--patch_root", type=str, default="data/patches_size512", help="Patch root base directory")
    parser.add_argument("--output_dir", type=str, default="visual/patches_vis_checks", help="Output base directory")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size suffix, e.g. _size512")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of random patches to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--offset_order",
        type=str,
        default="dy_dx",
        choices=["dy_dx", "dx_dy"],
        help="Offset channel order in stored offset patch",
    )
    parser.add_argument("--center_rel_thresh", type=float, default=0.30, help="Relative threshold for center peaks")
    parser.add_argument("--center_abs_thresh", type=float, default=0.05, help="Absolute threshold for center peaks")
    parser.add_argument("--arrow_step", type=int, default=32, help="Grid step for offset arrow drawing")
    parser.add_argument("--arrow_scale", type=float, default=1.0, help="Offset arrow scale")
    parser.add_argument("--font_scale", type=float, default=0.55, help="Annotation font scale")
    return parser.parse_args()


def find_file_by_stem(directory: Path, stem: str, exts):
    for ext in exts:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    candidates = sorted(directory.glob(f"{stem}.*"))
    for p in candidates:
        if p.suffix.lower() in exts:
            return p
    return None


def load_array(path: Path):
    if path.suffix.lower() == ".npy":
        return np.load(str(path))
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to read: {path}")
    return arr


def to_uint8_gray(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if np.isclose(vmax, vmin):
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - vmin) / (vmax - vmin + 1e-12)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def to_bgr_image(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        g = to_uint8_gray(arr)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            g = to_uint8_gray(arr[:, :, 0])
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if arr.dtype == np.uint8:
            return arr.copy()
        arrf = arr.astype(np.float32)
        vmin = float(np.min(arrf))
        vmax = float(np.max(arrf))
        if np.isclose(vmax, vmin):
            return np.zeros(arr.shape, dtype=np.uint8)
        out = (arrf - vmin) / (vmax - vmin + 1e-12)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported array ndim={arr.ndim}")


def id_to_color(idx):
    if idx <= 0:
        return (0, 0, 0)
    b = (37 * int(idx) + 23) % 256
    g = (17 * int(idx) + 91) % 256
    r = (97 * int(idx) + 53) % 256
    return int(b), int(g), int(r)


def colorize_label_map(label_map):
    label_map = np.asarray(label_map)
    if label_map.ndim == 3:
        label_map = label_map[:, :, 0]
    label_map = label_map.astype(np.int64)
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_map)
    for idx in ids:
        if idx <= 0:
            continue
        out[label_map == idx] = id_to_color(idx)
    return out


def blend(base_bgr, overlay_bgr, alpha=0.45):
    return cv2.addWeighted(base_bgr, 1.0 - alpha, overlay_bgr, alpha, 0.0)


def draw_text_lines(img, lines, start_xy=(8, 22), font_scale=0.55, color=(255, 255, 255)):
    x, y = start_xy
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += int(22 * font_scale / 0.55)
    return img


def add_panel_title(img, title, font_scale=0.62):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (25, 25, 25), thickness=-1)
    cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def semantic_edge_mask(semantic):
    sem = np.asarray(semantic)
    if sem.ndim == 3:
        sem = sem[:, :, 0]
    sem = sem.astype(np.int32)
    fg = (sem > 0).astype(np.uint8) * 255
    edge = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
    return edge > 0


def to_heatmap(arr2d):
    g = to_uint8_gray(arr2d)
    return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)


def to_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, 0]
    raise ValueError(f"Unsupported ndim for 2D conversion: {arr.ndim}")


def to_offset_hw2(offset_arr):
    arr = np.asarray(offset_arr)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 2:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Offset must be HxWx2 or 2xHxW, got shape={arr.shape}")


def detect_center_peaks(center_map, rel_thresh=0.3, abs_thresh=0.05, topk=80):
    c = to_2d(center_map).astype(np.float32)
    if c.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    cmax = float(np.max(c))
    if cmax <= 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    thr = max(abs_thresh, rel_thresh * cmax)

    local_max = (c == cv2.dilate(c, np.ones((3, 3), dtype=np.float32)))
    keep = np.logical_and(local_max, c >= thr)
    ys, xs = np.where(keep)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    vals = c[ys, xs]
    order = np.argsort(-vals)
    if order.size > topk:
        order = order[:topk]
    ys = ys[order]
    xs = xs[order]
    vals = vals[order]
    coords = np.stack([xs, ys], axis=1).astype(np.int32)
    return coords, vals.astype(np.float32)


def get_instance_centers(instance_map):
    inst = to_2d(instance_map).astype(np.int32)
    centers = {}
    for iid in np.unique(inst):
        if iid <= 0:
            continue
        ys, xs = np.where(inst == iid)
        if ys.size == 0:
            continue
        centers[int(iid)] = (float(np.mean(xs)), float(np.mean(ys)))
    return centers


def safe_cosine(a, b):
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an <= 1e-8 or bn <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def panel_image_semantic(image_bgr, semantic, font_scale):
    sem = np.asarray(semantic)
    if sem.ndim == 2:
        sem_color = colorize_label_map(sem)
    else:
        sem_color = to_bgr_image(sem)

    overlay = blend(image_bgr, sem_color, alpha=0.42)
    edge = semantic_edge_mask(semantic)
    vis = overlay.copy()
    vis[edge] = (0, 255, 0)
    sem2d = to_2d(semantic).astype(np.int32)
    sem_ids = np.unique(sem2d)
    sem_fg = sem_ids[sem_ids > 0]
    lines = [
        f"semantic classes: {len(sem_fg)}",
        f"fg pixel ratio: {float((sem2d > 0).mean()):.3f}",
        "green edge should follow leaf boundary",
    ]
    return add_panel_title(draw_text_lines(vis, lines, font_scale=font_scale), "1) image vs semantic alignment")


def panel_instance_check(image_bgr, instance, font_scale):
    inst = to_2d(instance).astype(np.int32)
    inst_color = colorize_label_map(inst)
    vis = blend(image_bgr, inst_color, alpha=0.45)

    uniq = np.unique(inst)
    nonzero = uniq[uniq > 0]
    bg_zero_ok = bool(0 in uniq)
    contiguous = bool(nonzero.size == 0 or np.array_equal(nonzero, np.arange(nonzero.min(), nonzero.max() + 1)))
    inst_num = int(nonzero.size)
    lines = [
        f"bg contains 0: {bg_zero_ok}",
        f"instance count: {inst_num}",
        f"id contiguous: {contiguous}",
    ]
    if inst_num > 0:
        lines.append(f"id min/max: {int(nonzero.min())}/{int(nonzero.max())}")
    return add_panel_title(draw_text_lines(vis, lines, font_scale=font_scale), "2) instance id sanity")


def panel_center_check(image_bgr, center, instance, rel_thresh, abs_thresh, font_scale):
    center2d = to_2d(center).astype(np.float32)
    peaks_xy, peak_vals = detect_center_peaks(center2d, rel_thresh=rel_thresh, abs_thresh=abs_thresh, topk=120)
    inst = to_2d(instance).astype(np.int32)

    heat = to_heatmap(center2d)
    vis = blend(image_bgr, heat, alpha=0.35)
    hit = 0
    for (x, y), val in zip(peaks_xy, peak_vals):
        in_leaf = 0 <= y < inst.shape[0] and 0 <= x < inst.shape[1] and inst[y, x] > 0
        if in_leaf:
            hit += 1
        col = (0, 255, 0) if in_leaf else (0, 0, 255)
        cv2.circle(vis, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)
        if val > peak_vals[0] * 0.6:
            cv2.putText(vis, f"{val:.2f}", (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    total = int(len(peaks_xy))
    ratio = float(hit / total) if total > 0 else 0.0
    lines = [
        f"peaks: {total}",
        f"peak on instance ratio: {ratio:.3f}",
        "green=on leaf, red=off leaf",
    ]
    return add_panel_title(draw_text_lines(vis, lines, font_scale=font_scale), "3) center peak on leaf")


def panel_offset_check(image_bgr, offset, instance, offset_order, arrow_step, arrow_scale, font_scale):
    off = to_offset_hw2(offset).astype(np.float32)
    inst = to_2d(instance).astype(np.int32)

    if offset_order == "dy_dx":
        dy = off[:, :, 0]
        dx = off[:, :, 1]
    else:
        dx = off[:, :, 0]
        dy = off[:, :, 1]

    vis = image_bgr.copy()
    h, w = inst.shape
    centers = get_instance_centers(inst)
    cos_list = []
    used_points = 0

    step = max(4, int(arrow_step))
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            iid = int(inst[y, x])
            if iid <= 0:
                continue
            v = np.array([dx[y, x], dy[y, x]], dtype=np.float32)
            if np.linalg.norm(v) < 1e-6:
                continue
            cxy = centers.get(iid, None)
            if cxy is not None:
                gt = np.array([cxy[0] - float(x), cxy[1] - float(y)], dtype=np.float32)
                cos_list.append(safe_cosine(v, gt))
            used_points += 1

            end_x = int(round(x + arrow_scale * float(v[0])))
            end_y = int(round(y + arrow_scale * float(v[1])))
            end_x = int(np.clip(end_x, 0, w - 1))
            end_y = int(np.clip(end_y, 0, h - 1))
            cv2.arrowedLine(vis, (x, y), (end_x, end_y), (255, 180, 0), 1, cv2.LINE_AA, tipLength=0.25)

    mean_cos = float(np.mean(cos_list)) if len(cos_list) > 0 else 0.0
    pos_ratio = float(np.mean(np.asarray(cos_list) > 0.0)) if len(cos_list) > 0 else 0.0
    lines = [
        f"arrow points: {used_points}",
        f"mean cosine to inst-center: {mean_cos:.3f}",
        f"cos>0 ratio: {pos_ratio:.3f}",
    ]
    return add_panel_title(draw_text_lines(vis, lines, font_scale=font_scale), "4) offset points to center"), {
        "offset_points": used_points,
        "offset_mean_cosine": mean_cos,
        "offset_positive_cos_ratio": pos_ratio,
    }


def panel_density_check(density, instance, font_scale):
    den = to_2d(density).astype(np.float32)
    inst = to_2d(instance).astype(np.int32)
    vis = to_heatmap(den)

    inst_ids = np.unique(inst)
    inst_ids = inst_ids[inst_ids > 0]
    contribs = []
    for iid in inst_ids:
        m = (inst == iid)
        c = float(den[m].sum()) if np.any(m) else 0.0
        contribs.append(c)
    contribs = np.asarray(contribs, dtype=np.float32)

    den_sum = float(den.sum())
    inst_num = int(len(inst_ids))
    frac_lt_099 = float(np.mean(contribs < 0.99)) if contribs.size > 0 else 0.0
    frac_close_1 = float(np.mean(np.abs(contribs - 1.0) <= 0.05)) if contribs.size > 0 else 0.0
    mean_c = float(np.mean(contribs)) if contribs.size > 0 else 0.0

    lines = [
        f"density sum: {den_sum:.3f}",
        f"instance count in patch: {inst_num}",
        f"mean per-inst contrib: {mean_c:.3f}",
        f"frac contrib < 0.99: {frac_lt_099:.3f}",
        f"frac contrib ~1: {frac_close_1:.3f}",
    ]
    return add_panel_title(draw_text_lines(vis, lines, font_scale=font_scale), "5) density global-definition check"), {
        "density_sum": den_sum,
        "density_instance_count": inst_num,
        "density_mean_per_instance_contrib": mean_c,
        "density_frac_contrib_lt_0_99": frac_lt_099,
        "density_frac_contrib_close_1": frac_close_1,
    }


def panel_boundary_truncation(image_bgr, instance, font_scale):
    inst = to_2d(instance).astype(np.int32)
    h, w = inst.shape
    vis = image_bgr.copy()

    boundary_ids = set()
    for row in (0, h - 1):
        vals = np.unique(inst[row, :])
        boundary_ids.update([int(v) for v in vals if v > 0])
    for col in (0, w - 1):
        vals = np.unique(inst[:, col])
        boundary_ids.update([int(v) for v in vals if v > 0])

    boundary_ids = sorted(boundary_ids)
    trunc_mask = np.zeros(inst.shape, dtype=np.uint8)
    for iid in boundary_ids:
        trunc_mask[inst == iid] = 255

    overlay = vis.copy()
    overlay[trunc_mask > 0] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0.0)

    contours, _ = cv2.findContours(trunc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    total_inst = int(np.sum(np.unique(inst) > 0))
    trunc_cnt = int(len(boundary_ids))
    ratio = float(trunc_cnt / total_inst) if total_inst > 0 else 0.0
    lines = [
        f"boundary-touching instances: {trunc_cnt}",
        f"total instances: {total_inst}",
        f"touch ratio: {ratio:.3f}",
        "red region = truncated-by-patch objects",
    ]
    return add_panel_title(draw_text_lines(vis, lines, font_scale=font_scale), "6) boundary truncation check"), {
        "boundary_touching_instances": trunc_cnt,
        "total_instances": total_inst,
        "boundary_touch_ratio": ratio,
    }


def build_grid_2x3(panels):
    if len(panels) != 6:
        raise ValueError("Need exactly 6 panels for 2x3 grid.")
    h, w = panels[0].shape[:2]
    rows = []
    for i in range(0, 6, 3):
        row = []
        for j in range(3):
            p = panels[i + j]
            if p.shape[:2] != (h, w):
                p = cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA)
            row.append(p)
        rows.append(np.hstack(row))
    return np.vstack(rows)


def gather_patch_samples(patch_root: Path):
    dirs = {
        "images": patch_root / "images",
        "semantic": patch_root / "semantic",
        "instance": patch_root / "instance",
        "center": patch_root / "center",
        "offset": patch_root / "offset",
        "density": patch_root / "density",
    }
    for name, d in dirs.items():
        if not d.exists():
            raise FileNotFoundError(f"Missing directory for {name}: {d}")

    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(sorted(dirs["images"].glob(f"*{ext}")))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise RuntimeError(f"No patch images found in: {dirs['images']}")

    samples = []
    for img_path in image_paths:
        stem = img_path.stem
        sem = find_file_by_stem(dirs["semantic"], stem, ANY_EXTS)
        ins = find_file_by_stem(dirs["instance"], stem, ANY_EXTS)
        cen = find_file_by_stem(dirs["center"], stem, ANY_EXTS)
        off = find_file_by_stem(dirs["offset"], stem, ANY_EXTS)
        den = find_file_by_stem(dirs["density"], stem, ANY_EXTS)
        if any(x is None for x in [sem, ins, cen, off, den]):
            continue
        samples.append(
            {
                "stem": stem,
                "image": img_path,
                "semantic": sem,
                "instance": ins,
                "center": cen,
                "offset": off,
                "density": den,
            }
        )
    return samples


def append_patch_size_suffix(path: Path, patch_size: int):
    base_name = re.sub(r"_size\d+$", "", path.name)
    return path.with_name(f"{base_name}_size{patch_size}")


def resolve_patch_root(base_patch_root: Path, patch_size: int):
    preferred = append_patch_size_suffix(base_patch_root, patch_size)
    if preferred.exists():
        return preferred
    if base_patch_root.exists():
        print(f"[WARN] Preferred patch root not found, fallback to legacy path: {base_patch_root}")
        return base_patch_root
    return preferred


def main():
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be > 0")

    patch_root = resolve_patch_root(Path(args.patch_root), args.patch_size)
    out_dir = append_patch_size_suffix(Path(args.output_dir), args.patch_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = gather_patch_samples(patch_root)
    if len(samples) == 0:
        print("[ERROR] No complete patch samples (with all 6 targets) found.")
        return

    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, len(samples))
    select_idx = rng.choice(len(samples), size=n, replace=False)
    picked = [samples[int(i)] for i in select_idx]

    report_rows = []
    saved = 0
    skipped = 0

    for item in tqdm(picked, desc="Visualizing patches", ncols=100):
        stem = item["stem"]
        try:
            image = load_array(item["image"])
            semantic = load_array(item["semantic"])
            instance = load_array(item["instance"])
            center = load_array(item["center"])
            offset = load_array(item["offset"])
            density = load_array(item["density"])
        except Exception as e:
            print(f"[WARN] {stem}: load failed: {e}")
            skipped += 1
            continue

        image_bgr = to_bgr_image(image)
        h, w = image_bgr.shape[:2]

        try:
            if to_2d(instance).shape != (h, w):
                raise ValueError(f"instance shape mismatch {to_2d(instance).shape} vs {(h, w)}")
            if to_2d(center).shape != (h, w):
                raise ValueError(f"center shape mismatch {to_2d(center).shape} vs {(h, w)}")
            if to_2d(density).shape != (h, w):
                raise ValueError(f"density shape mismatch {to_2d(density).shape} vs {(h, w)}")
            off_hw2 = to_offset_hw2(offset)
            if off_hw2.shape[:2] != (h, w):
                raise ValueError(f"offset shape mismatch {off_hw2.shape[:2]} vs {(h, w)}")
        except Exception as e:
            print(f"[WARN] {stem}: shape check failed: {e}")
            skipped += 1
            continue

        p1 = panel_image_semantic(image_bgr, semantic, font_scale=args.font_scale)
        p2 = panel_instance_check(image_bgr, instance, font_scale=args.font_scale)
        p3 = panel_center_check(
            image_bgr,
            center,
            instance,
            rel_thresh=args.center_rel_thresh,
            abs_thresh=args.center_abs_thresh,
            font_scale=args.font_scale,
        )
        p4, m4 = panel_offset_check(
            image_bgr,
            off_hw2,
            instance,
            offset_order=args.offset_order,
            arrow_step=args.arrow_step,
            arrow_scale=args.arrow_scale,
            font_scale=args.font_scale,
        )
        p5, m5 = panel_density_check(density, instance, font_scale=args.font_scale)
        p6, m6 = panel_boundary_truncation(image_bgr, instance, font_scale=args.font_scale)

        grid = build_grid_2x3([p1, p2, p3, p4, p5, p6])
        head_h = 42
        head = np.full((head_h, grid.shape[1], 3), 22, dtype=np.uint8)
        cv2.putText(
            head,
            f"{stem} | size={w}x{h} | offset_order={args.offset_order}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out_vis = np.vstack([head, grid])

        out_path = out_dir / f"{stem}_check.jpg"
        ok = cv2.imwrite(str(out_path), out_vis)
        if not ok:
            print(f"[WARN] {stem}: failed to save {out_path}")
            skipped += 1
            continue
        saved += 1

        report_rows.append(
            {
                "stem": stem,
                "output": str(out_path),
                **m4,
                **m5,
                **m6,
            }
        )

    report_csv = out_dir / "summary.csv"
    if report_rows:
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)

    print("\n=== Visualization Done ===")
    print(f"patch_root: {patch_root}")
    print(f"patch_size: {args.patch_size}")
    print(f"output_dir: {out_dir}")
    print(f"requested_samples: {n}")
    print(f"saved: {saved}")
    print(f"skipped: {skipped}")
    print(f"summary_csv: {report_csv if report_rows else 'N/A'}")


if __name__ == "__main__":
    main()
