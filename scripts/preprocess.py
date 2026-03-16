#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    def tqdm(x, **kwargs):
        return x


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true", help=f"Enable {help_text}")
    group.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess rice canopy images for better SAM-style point/box prompted segmentation."
    )

    parser.add_argument("--input_dir", type=str, default="data/raw/original_images/", help="Input image directory")
    parser.add_argument("--output_dir", type=str, default="data/processed/images/", help="Output directory")
    parser.add_argument("--suffix", type=str, default="_pp", help="Output filename suffix, e.g. '_pp' (can be empty)")
    parser.add_argument("--save_compare", action="store_true", help="Save side-by-side original vs preprocessed image")
    parser.add_argument("--compare_suffix", type=str, default="_compare", help="Suffix for compare image filename")

    # A) Resize
    parser.add_argument("--scale", type=float, default=1.0, help="Resize scale, e.g. 1.0 / 0.75 / 0.5")

    # B) CLAHE
    add_bool_arg(parser, "clahe", True, "CLAHE on LAB-L channel")
    parser.add_argument("--clahe_clip", type=float, default=2.8, help="CLAHE clip limit")
    parser.add_argument("--clahe_grid", type=int, default=6, help="CLAHE grid size for tileGridSize=(g,g)")

    # C) Bilateral
    add_bool_arg(parser, "bilateral", True, "bilateral filtering")
    parser.add_argument("--bilateral_d", type=int, default=9, help="Bilateral filter diameter")
    parser.add_argument("--bilateral_sc", type=float, default=80.0, help="Bilateral sigmaColor")
    parser.add_argument("--bilateral_ss", type=float, default=80.0, help="Bilateral sigmaSpace")

    # D) Highlight suppression
    add_bool_arg(parser, "highlight", True, "highlight suppression")
    parser.add_argument("--hl_v_thresh", type=float, default=210.0, help="HSV V threshold for highlights")
    parser.add_argument("--hl_s_thresh", type=float, default=55.0, help="HSV S threshold for low saturation")
    parser.add_argument("--hl_strength", type=float, default=0.5, help="Highlight suppression strength [0,1]")

    # E) Unsharp (optional)
    add_bool_arg(parser, "unsharp", False, "unsharp mask")
    parser.add_argument("--unsharp_amount", type=float, default=0.6, help="Unsharp amount")
    parser.add_argument("--unsharp_sigma", type=float, default=1.0, help="Unsharp Gaussian sigma")

    return parser.parse_args()


def list_images(input_dir: Path):
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return files


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if abs(scale - 1.0) < 1e-9:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def apply_clahe_lab(img_bgr: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    grid_size = max(1, int(grid_size))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid_size, grid_size))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def apply_bilateral(img_bgr: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    d = max(1, int(d))
    return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))


def suppress_highlight_hsv(
    img_bgr: np.ndarray,
    v_thresh: float = 220.0,
    s_thresh: float = 40.0,
    strength: float = 0.35,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    v_thresh = float(np.clip(v_thresh, 0.0, 255.0))
    s_thresh = float(np.clip(s_thresh, 0.0, 255.0))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    v_den = max(1.0, 255.0 - v_thresh)
    s_den = max(1.0, s_thresh)

    # Soft highlight mask: bright + low saturation
    bright = np.clip((v - v_thresh) / v_den, 0.0, 1.0)
    low_sat = np.clip((s_thresh - s) / s_den, 0.0, 1.0)
    mask = bright * low_sat

    # Smooth mask edges to avoid harsh local artifacts
    mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2)

    # Softly compress V toward threshold (not hard clipping)
    reduction = strength * mask
    v_new = v - reduction * (v - v_thresh)
    v_new = np.clip(v_new, 0.0, 255.0)

    out_hsv = cv2.merge([h, s, v_new]).astype(np.uint8)
    return cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)


def apply_unsharp(img_bgr: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    amount = max(0.0, float(amount))
    sigma = max(1e-6, float(sigma))
    blur = cv2.GaussianBlur(img_bgr, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0)
    return sharp


def make_compare_image(original_bgr: np.ndarray, processed_bgr: np.ndarray) -> np.ndarray:
    if original_bgr.shape[:2] != processed_bgr.shape[:2]:
        oh, ow = original_bgr.shape[:2]
        ph, pw = processed_bgr.shape[:2]
        interp = cv2.INTER_AREA if (ph * pw) < (oh * ow) else cv2.INTER_LINEAR
        original_bgr = cv2.resize(original_bgr, (pw, ph), interpolation=interp)
    return np.concatenate([original_bgr, processed_bgr], axis=1)


def preprocess_image(img_bgr: np.ndarray, args) -> np.ndarray:
    out = img_bgr.copy()

    # A) Resize
    out = resize_image(out, args.scale)

    # B) CLAHE
    if args.clahe:
        out = apply_clahe_lab(out, clip_limit=args.clahe_clip, grid_size=args.clahe_grid)

    # C) Bilateral
    if args.bilateral:
        out = apply_bilateral(
            out,
            d=args.bilateral_d,
            sigma_color=args.bilateral_sc,
            sigma_space=args.bilateral_ss,
        )

    # D) Highlight suppression
    if args.highlight:
        out = suppress_highlight_hsv(
            out,
            v_thresh=args.hl_v_thresh,
            s_thresh=args.hl_s_thresh,
            strength=args.hl_strength,
        )

    # E) Unsharp (optional)
    if args.unsharp:
        out = apply_unsharp(out, amount=args.unsharp_amount, sigma=args.unsharp_sigma)

    return out


def main():
    args = parse_args()

    if args.scale <= 0:
        raise ValueError("--scale must be > 0")
    if not (0.0 <= args.hl_strength <= 1.0):
        raise ValueError("--hl_strength must be in [0, 1]")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_dir = output_dir / "compare"
    if args.save_compare:
        compare_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return

    image_paths = list_images(input_dir)
    total = len(image_paths)

    if total == 0:
        print(f"[INFO] No images found in: {input_dir}")
        return

    processed = 0
    skipped = 0
    total_time = 0.0

    for img_path in tqdm(image_paths, desc="Preprocessing", ncols=100):
        t0 = time.perf_counter()
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                print(f"[WARN] Failed to read image, skipped: {img_path.name}")
                continue

            out = preprocess_image(img, args)

            out_name = f"{img_path.stem}{args.suffix}{img_path.suffix}"
            out_path = output_dir / out_name
            ok = cv2.imwrite(str(out_path), out)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for {out_path}")

            if args.save_compare:
                comp = make_compare_image(img, out)
                comp_name = f"{img_path.stem}{args.suffix}{args.compare_suffix}.jpg"
                comp_path = compare_dir / comp_name
                ok2 = cv2.imwrite(str(comp_path), comp)
                if not ok2:
                    raise RuntimeError(f"cv2.imwrite failed for {comp_path}")

            processed += 1
            total_time += (time.perf_counter() - t0)

        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipped {img_path.name}: {e}")

    avg_time = total_time / processed if processed > 0 else 0.0
    print("\n[Done]")
    print(f"Total images   : {total}")
    print(f"Processed      : {processed}")
    print(f"Skipped        : {skipped}")
    print(f"Avg time/image : {avg_time:.4f} sec")


if __name__ == "__main__":
    main()
