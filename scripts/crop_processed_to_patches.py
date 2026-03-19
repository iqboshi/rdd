#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


ALL_TARGETS = ["image", "semantic", "instance", "center", "offset", "density"]
IMAGE_LIKE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
NUMPY_EXTS = [".npy"]
INSTANCE_CENTER_DENSITY_EXTS = [".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
SEMANTIC_EXTS = [".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

# Offset split naming patterns, easy to extend:
# pattern_x.format(stem=xxx), pattern_y.format(stem=xxx)
OFFSET_SPLIT_STEM_PATTERNS = [
    ("{stem}_x", "{stem}_y"),
    ("{stem}_offset_x", "{stem}_offset_y"),
    ("{stem}_dx", "{stem}_dy"),
]


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true", help=f"Enable {help_text}")
    group.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name: default})


def parse_args():
    parser = argparse.ArgumentParser(description="Crop full-image aligned targets into training patches.")
    parser.add_argument("--input_root", type=str, default="data/processed", help="Input root with images/semantic/...")
    parser.add_argument("--output_root", type=str, default="data/patches", help="Output root for patch folders")
    parser.add_argument(
        "--patch_size",
        nargs="+",
        default=["512", "768", "1024"],
        help="Patch size list, e.g. --patch_size 512 640 or --patch_size [512,640]",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding stride; default is patch_size//2 for each patch_size",
    )
    add_bool_arg(parser, "extra_crop_enable", False, "extra shifted crop rounds for one selected patch size")
    parser.add_argument(
        "--extra_crop_size",
        type=int,
        default=1024,
        help="Patch size to apply extra shifted crop rounds on; default uses current patch_size when enabled",
    )
    parser.add_argument(
        "--extra_rounds",
        type=int,
        default=2,
        help="Number of extra shifted crop rounds for the selected patch size",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        help="Targets to crop, e.g. --targets image semantic instance; use 'all' for all",
    )
    add_bool_arg(parser, "remap_instance", True, "instance id remap in each patch")
    add_bool_arg(parser, "skip_small_images", True, "skip images smaller than patch_size")
    return parser.parse_args()


def normalize_targets(raw_targets):
    items = []
    for token in raw_targets:
        parts = [x.strip().lower() for x in token.split(",") if x.strip()]
        items.extend(parts)
    if not items or "all" in items:
        return ALL_TARGETS.copy()
    uniq = []
    for t in items:
        if t not in uniq:
            uniq.append(t)
    invalid = [t for t in uniq if t not in ALL_TARGETS]
    if invalid:
        raise ValueError(f"Invalid targets: {invalid}, valid: {ALL_TARGETS} or all")
    return uniq


def normalize_patch_sizes(raw_patch_sizes):
    tokens = []
    for raw in raw_patch_sizes:
        text = str(raw).strip()
        if not text:
            continue
        text = text.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ")
        parts = [x.strip() for x in text.split(",") if x.strip()]
        if not parts:
            parts = [x.strip() for x in text.split() if x.strip()]
        tokens.extend(parts)

    if not tokens:
        raise ValueError("--patch_size must provide at least one positive integer")

    patch_sizes = []
    for token in tokens:
        try:
            size = int(token)
        except Exception:
            raise ValueError(f"Invalid patch size: {token}") from None
        if size <= 0:
            raise ValueError(f"--patch_size must be > 0, got {size}")
        if size not in patch_sizes:
            patch_sizes.append(size)
    return patch_sizes


def warn(msg, stats):
    stats["warnings"] += 1
    print(f"[WARN] {msg}")


def resolve_first_existing_subdir(input_root: Path, candidates):
    for name in candidates:
        p = input_root / name
        if p.is_dir():
            return p
    return input_root / candidates[0]


def find_file_by_stem(directory: Path, stem: str, allowed_exts):
    for ext in allowed_exts:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # fallback: handle uppercase or uncommon suffix cases
    candidates = sorted(directory.glob(f"{stem}.*"))
    for c in candidates:
        if c.suffix.lower() in allowed_exts:
            return c
    return None


def load_array(path: Path):
    if path.suffix.lower() == ".npy":
        return np.load(str(path))
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to read file: {path}")
    return arr


def save_array(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(str(path), arr)
        return
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write file: {path}")


def to_hw2_offset(offset: np.ndarray):
    if offset.ndim == 3 and offset.shape[-1] == 2:
        return offset
    if offset.ndim == 3 and offset.shape[0] == 2:
        return np.transpose(offset, (1, 2, 0))
    raise ValueError(f"Offset combined file must be HxWx2 or 2xHxW, got shape={offset.shape}")


def resolve_offset_input(offset_dir: Path, stem: str, offset_x_dir: Path = None, offset_y_dir: Path = None):
    # Case 1: one combined file, expected shape HxWx2 (or 2xHxW)
    if offset_dir is not None and offset_dir.exists():
        combined = find_file_by_stem(offset_dir, stem, NUMPY_EXTS)
        if combined is not None:
            return {"mode": "combined", "path": combined}

    # Case 2: split x/y files. Update patterns above if your naming differs.
    split_exts = NUMPY_EXTS + IMAGE_LIKE_EXTS
    if offset_dir is not None and offset_dir.exists():
        for pattern_x, pattern_y in OFFSET_SPLIT_STEM_PATTERNS:
            stem_x = pattern_x.format(stem=stem)
            stem_y = pattern_y.format(stem=stem)
            x_path = find_file_by_stem(offset_dir, stem_x, split_exts)
            y_path = find_file_by_stem(offset_dir, stem_y, split_exts)
            if x_path is not None and y_path is not None:
                return {"mode": "split", "x_path": x_path, "y_path": y_path}

    # Also support separate directories: input_root/offset_x and input_root/offset_y
    if offset_x_dir is not None and offset_y_dir is not None and offset_x_dir.exists() and offset_y_dir.exists():
        x_path = find_file_by_stem(offset_x_dir, stem, split_exts)
        y_path = find_file_by_stem(offset_y_dir, stem, split_exts)
        if x_path is not None and y_path is not None:
            return {"mode": "split", "x_path": x_path, "y_path": y_path}
    return None


def load_offset(offset_info):
    if offset_info["mode"] == "combined":
        arr = load_array(offset_info["path"])
        return to_hw2_offset(arr)

    ox = load_array(offset_info["x_path"])
    oy = load_array(offset_info["y_path"])
    if ox.ndim == 3:
        ox = ox[:, :, 0]
    if oy.ndim == 3:
        oy = oy[:, :, 0]
    if ox.shape != oy.shape:
        raise ValueError(f"Offset split shape mismatch: x={ox.shape}, y={oy.shape}")
    return np.stack([ox, oy], axis=-1)


def get_positions(length, patch_size, stride):
    # Keep all windows fully inside image, then force-cover right/bottom boundary.
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_crop_coords(h, w, patch_size, stride):
    xs = get_positions(w, patch_size, stride)
    ys = get_positions(h, patch_size, stride)
    coords = []
    for y in ys:
        for x in xs:
            coords.append((x, y))
    return coords


def get_positions_with_offset(length, patch_size, stride, offset):
    last = length - patch_size
    if last < 0:
        return []
    start = min(max(int(offset), 0), last)
    return list(range(start, last + 1, stride))


def generate_shifted_crop_coords(h, w, patch_size, stride, offset_x, offset_y):
    xs = get_positions_with_offset(w, patch_size, stride, offset_x)
    ys = get_positions_with_offset(h, patch_size, stride, offset_y)
    coords = []
    for y in ys:
        for x in xs:
            coords.append((x, y))
    return coords


def generate_crop_plan(h, w, patch_size, stride, extra_crop_enable=False, extra_rounds=1):
    original_coords = generate_crop_coords(h, w, patch_size, stride)
    plan = [{"left": left, "top": top, "tag": "base"} for left, top in original_coords]
    seen = {(left, top) for left, top in original_coords}

    if not extra_crop_enable or extra_rounds <= 0:
        return plan

    for round_idx in range(extra_rounds):
        # Spread extra rounds between original stride anchors with a small diagonal shift.
        shift = max(1, (round_idx + 1) * stride // (extra_rounds + 1))
        shifted_coords = generate_shifted_crop_coords(
            h,
            w,
            patch_size,
            stride,
            offset_x=shift,
            offset_y=shift,
        )
        for left, top in shifted_coords:
            key = (left, top)
            if key in seen:
                continue
            seen.add(key)
            plan.append({"left": left, "top": top, "tag": f"extra_r{round_idx + 1}"})
    return plan


def crop_array(arr, left, top, patch_size):
    return arr[top:top + patch_size, left:left + patch_size]


def remap_instance_ids(inst_patch: np.ndarray):
    # Background keeps 0; non-zero ids in current patch are remapped to 1..N
    if inst_patch.ndim == 3:
        inst_patch = inst_patch[:, :, 0]
    inst_patch = inst_patch.astype(np.int64, copy=False)
    unique_ids = np.unique(inst_patch)
    unique_ids = unique_ids[unique_ids != 0]
    if unique_ids.size == 0:
        return np.zeros_like(inst_patch, dtype=np.int32)

    out = np.zeros_like(inst_patch, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        out[inst_patch == old_id] = new_id
    return out


def prepare_output_dirs(output_root: Path, targets):
    out_dirs = {}
    for t in targets:
        folder = "images" if t == "image" else t
        out_dirs[t] = output_root / folder
        out_dirs[t].mkdir(parents=True, exist_ok=True)
    return out_dirs


def append_patch_size_suffix(output_root: Path, patch_size: int):
    base_name = re.sub(r"_size\d+$", "", output_root.name)
    return output_root.with_name(f"{base_name}_size{patch_size}")


def main():
    args = parse_args()
    targets = normalize_targets(args.targets)
    patch_sizes = normalize_patch_sizes(args.patch_size)
    if args.extra_rounds < 0:
        raise ValueError("--extra_rounds must be >= 0")

    input_root = Path(args.input_root)
    output_root_base = Path(args.output_root)

    images_dir = input_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    target_dirs = {
        "semantic": resolve_first_existing_subdir(input_root, ["semantic_maps", "semantic"]),
        "instance": resolve_first_existing_subdir(input_root, ["instance_maps", "instance"]),
        "center": resolve_first_existing_subdir(input_root, ["center_maps", "center"]),
        "offset": resolve_first_existing_subdir(input_root, ["offset_maps", "offset"]),
        "density": resolve_first_existing_subdir(input_root, ["density_maps", "density"]),
    }
    offset_x_dir = input_root / "offset_x"
    offset_y_dir = input_root / "offset_y"

    for t in targets:
        if t == "image":
            continue
        if t == "offset":
            offset_ok = target_dirs["offset"].exists() or (offset_x_dir.exists() and offset_y_dir.exists())
            if not offset_ok:
                print(f"[WARN] Selected offset dir not found: {target_dirs['offset']} (or offset_x/offset_y pair)")
                print("[ERROR] Missing selected target directory. Stop.")
                return
            continue
        if not target_dirs[t].exists():
            print(f"[WARN] Selected target dir not found: {target_dirs[t]}")
            print("[ERROR] Missing selected target directory. Stop.")
            return

    image_files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_LIKE_EXTS])
    if not image_files:
        print("[INFO] No images found in input images directory.")
        return

    for patch_size in patch_sizes:
        stride = args.stride if args.stride is not None else max(1, patch_size // 2)
        if stride <= 0:
            raise ValueError("--stride must be > 0")
        extra_crop_active = (
            args.extra_crop_enable
            and args.extra_rounds > 0
            and (args.extra_crop_size is None or args.extra_crop_size == patch_size)
        )

        output_root = append_patch_size_suffix(output_root_base, patch_size)
        out_dirs = prepare_output_dirs(output_root, targets)
        stats = {
            "warnings": 0,
            "input_images_total": len(image_files),
            "input_images_processed": 0,
            "total_patches": 0,
            "base_patches": 0,
            "extra_patches": 0,
            "saved_counts": {t: 0 for t in targets},
        }

        print(
            f"\n[INFO] Start cropping patch_size={patch_size}, stride={stride}, "
            f"extra_crop_active={extra_crop_active}, output_root={output_root}"
        )

        for img_path in tqdm(image_files, desc=f"Cropping(ps={patch_size})", ncols=100):
            stem = img_path.stem

            try:
                image = load_array(img_path)
            except Exception as e:
                warn(f"Failed to read image {img_path.name}: {e}", stats)
                continue

            if image.ndim < 2:
                warn(f"Invalid image shape for {img_path.name}: {image.shape}", stats)
                continue

            h, w = image.shape[:2]
            if h < patch_size or w < patch_size:
                msg = f"Image smaller than patch_size, skip: {img_path.name}, shape=({h},{w}), patch={patch_size}"
                if args.skip_small_images:
                    warn(msg, stats)
                    continue
                raise ValueError(msg)

            sample_data = {
                "image": {"array": image, "save_ext": img_path.suffix.lower()},
            }

            skip_sample = False
            for target in targets:
                if target == "image":
                    continue

                target_dir = target_dirs[target]
                try:
                    if target == "semantic":
                        t_path = find_file_by_stem(target_dir, stem, SEMANTIC_EXTS)
                        if t_path is None:
                            warn(f"Missing semantic for {stem}", stats)
                            skip_sample = True
                            break
                        arr = load_array(t_path)
                        save_ext = t_path.suffix.lower()

                    elif target in ("instance", "center", "density"):
                        t_path = find_file_by_stem(target_dir, stem, INSTANCE_CENTER_DENSITY_EXTS)
                        if t_path is None:
                            warn(f"Missing {target} for {stem}", stats)
                            skip_sample = True
                            break
                        arr = load_array(t_path)
                        if target in ("instance", "density"):
                            save_ext = ".npy"
                        else:
                            save_ext = ".npy" if t_path.suffix.lower() == ".npy" else t_path.suffix.lower()

                    elif target == "offset":
                        offset_info = resolve_offset_input(
                            target_dir,
                            stem,
                            offset_x_dir=offset_x_dir,
                            offset_y_dir=offset_y_dir,
                        )
                        if offset_info is None:
                            warn(f"Missing offset for {stem} (combined npy or split x/y)", stats)
                            skip_sample = True
                            break
                        arr = load_offset(offset_info)
                        save_ext = ".npy"
                    else:
                        raise ValueError(f"Unsupported target: {target}")
                except Exception as e:
                    warn(f"Failed loading {target} for {stem}: {e}", stats)
                    skip_sample = True
                    break

                if arr.shape[0] != h or arr.shape[1] != w:
                    warn(
                        f"Shape mismatch for {stem} target={target}: image=({h},{w}), target={arr.shape[:2]}",
                        stats,
                    )
                    skip_sample = True
                    break
                sample_data[target] = {"array": arr, "save_ext": save_ext}

            if skip_sample:
                continue

            crop_plan = generate_crop_plan(
                h,
                w,
                patch_size,
                stride,
                extra_crop_enable=extra_crop_active,
                extra_rounds=args.extra_rounds,
            )
            stats["input_images_processed"] += 1
            stats["total_patches"] += len(crop_plan)
            stats["base_patches"] += sum(1 for item in crop_plan if item["tag"] == "base")
            stats["extra_patches"] += sum(1 for item in crop_plan if item["tag"] != "base")

            for crop_item in crop_plan:
                left = crop_item["left"]
                top = crop_item["top"]
                patch_stem = f"{stem}_x{left}_y{top}"

                for target in targets:
                    arr = sample_data[target]["array"]
                    patch = crop_array(arr, left, top, patch_size)

                    if target == "instance" and args.remap_instance:
                        patch = remap_instance_ids(patch)

                    save_ext = sample_data[target]["save_ext"]
                    save_path = out_dirs[target] / f"{patch_stem}{save_ext}"
                    try:
                        save_array(save_path, patch)
                        stats["saved_counts"][target] += 1
                    except Exception as e:
                        warn(f"Failed saving {target} patch: {save_path.name}, err={e}", stats)

        print("\n=== Crop Summary ===")
        print(f"Patch size: {patch_size}")
        print(f"Stride: {stride}")
        print(f"Output root: {output_root}")
        print(f"Input images total: {stats['input_images_total']}")
        print(f"Input images processed: {stats['input_images_processed']}")
        print(f"Total patches generated: {stats['total_patches']}")
        print(f"Base patches generated: {stats['base_patches']}")
        print(f"Extra patches generated: {stats['extra_patches']}")
        print("Saved patch counts by target:")
        for t in targets:
            print(f"  {t}: {stats['saved_counts'][t]}")
        print(f"Warnings: {stats['warnings']}")


if __name__ == "__main__":
    main()
