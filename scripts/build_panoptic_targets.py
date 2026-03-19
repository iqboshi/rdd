#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build density/semantic/instance/center/offset maps from ISAT JSON."
    )
    parser.add_argument("--image_dir", default="data/processed/images", type=str)
    parser.add_argument("--label_dir", default="data/processed/labels", type=str)

    parser.add_argument("--density_dir", default="data/processed/density_maps", type=str)
    parser.add_argument("--semantic_dir", default="data/processed/semantic_maps", type=str)
    parser.add_argument("--instance_dir", default="data/processed/instance_maps", type=str)
    parser.add_argument("--center_dir", default="data/processed/center_maps", type=str)
    parser.add_argument("--offset_dir", default="data/processed/offset_maps", type=str)

    parser.add_argument("--save_vis", default=True, help="Save visualization images.")
    parser.add_argument("--vis_dir", default="data/processed/target_vis", type=str)
    parser.add_argument(
        "--match_suffix",
        default="_pp",
        type=str,
        help="If exact label name is not found, try removing this suffix from image stem.",
    )
    parser.add_argument(
        "--center_sigma",
        default=4.0,
        type=float,
        help="Gaussian sigma for center heatmap.",
    )
    return parser.parse_args()


def list_images(image_dir):
    paths = []
    for p in glob.glob(os.path.join(image_dir, "*")):
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTS:
            paths.append(p)
    return sorted(paths)


def find_label_path(image_path, label_dir, match_suffix="_pp"):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidate = os.path.join(label_dir, stem + ".json")
    if os.path.exists(candidate):
        return candidate

    if match_suffix and stem.endswith(match_suffix):
        stem2 = stem[: -len(match_suffix)]
        candidate2 = os.path.join(label_dir, stem2 + ".json")
        if os.path.exists(candidate2):
            return candidate2

    return None


def load_json(label_path):
    for encoding in ("utf-8", "gbk"):
        try:
            with open(label_path, "r", encoding=encoding) as f:
                return json.load(f)
        except Exception:
            continue
    raise ValueError(f"Failed to parse json: {label_path}")


def get_canvas_size(label_data, image_shape=None):
    info = label_data.get("info", {})
    w = int(info.get("width", 0)) if info.get("width", 0) else 0
    h = int(info.get("height", 0)) if info.get("height", 0) else 0

    if w > 0 and h > 0:
        return h, w
    if image_shape is not None:
        return image_shape[:2]
    raise ValueError("Cannot infer canvas size from label info or image.")


def segmentation_to_polygons(segmentation):
    polygons = []
    if segmentation is None:
        return polygons

    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return polygons

        # Case A: [[x,y], [x,y], ...]
        try:
            arr = np.asarray(segmentation, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 3:
                polygons.append(arr[:, :2])
                return polygons
        except Exception:
            pass

        # Case B: [ [x,y,...], [x,y,...], ... ] or [ [[x,y],...], [[x,y],...] ]
        for part in segmentation:
            try:
                arr = np.asarray(part, dtype=np.float32)
            except Exception:
                continue
            if arr.ndim == 1 and arr.size >= 6 and arr.size % 2 == 0:
                polygons.append(arr.reshape(-1, 2))
            elif arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 3:
                polygons.append(arr[:, :2])

    return polygons


def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask.astype(bool)

    for poly in polygons:
        pts = np.round(poly).astype(np.int32)
        if pts.shape[0] < 3:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def normalize_group_id(group_value):
    if group_value is None:
        return None
    if isinstance(group_value, (int, np.integer)):
        return int(group_value)
    if isinstance(group_value, (float, np.floating)):
        if np.isfinite(group_value):
            rounded = int(round(float(group_value)))
            if abs(float(group_value) - rounded) < 1e-6:
                return rounded
            return float(group_value)
        return None
    return str(group_value)


def parse_layer_value(layer_value, default_value):
    if layer_value is None:
        return float(default_value)
    try:
        return float(layer_value)
    except Exception:
        return float(default_value)


def majority_vote(values, default_value="unknown"):
    filtered = [v for v in values if isinstance(v, str) and v.strip() != ""]
    if not filtered:
        return default_value
    uniq, counts = np.unique(np.asarray(filtered), return_counts=True)
    return str(uniq[int(np.argmax(counts))])


def extract_instances_from_isat(label_data, height, width):
    """
    Merge all object parts sharing the same group into one instance.
    """
    objects = label_data.get("objects", [])
    merged = {}

    for idx, obj in enumerate(objects):
        segmentation = obj.get("segmentation", None)
        polygons = segmentation_to_polygons(segmentation)
        if not polygons:
            continue

        mask = polygons_to_mask(polygons, height, width)
        if int(mask.sum()) <= 0:
            continue

        group_id = normalize_group_id(obj.get("group", None))
        key = f"__obj_{idx}" if group_id is None else f"group_{group_id}"

        category = str(obj.get("category", "")).strip()
        layer = parse_layer_value(obj.get("layer", None), default_value=idx)

        if key not in merged:
            merged[key] = {
                "mask": mask,
                "categories": [category],
                "layer": layer,
                "key": key,
            }
        else:
            merged[key]["mask"] = np.logical_or(merged[key]["mask"], mask)
            merged[key]["categories"].append(category)
            merged[key]["layer"] = max(merged[key]["layer"], layer)

    instances = []
    for key, item in merged.items():
        category = majority_vote(item["categories"], default_value="unknown")
        if int(item["mask"].sum()) <= 0:
            continue
        instances.append(
            {
                "key": key,
                "mask": item["mask"],
                "category": category,
                "layer": float(item["layer"]),
            }
        )

    # Lower layer first; later instances overwrite on overlap.
    instances = sorted(instances, key=lambda x: (x["layer"], x["key"]))
    return instances


def get_or_add_class_id(class_to_id, category_name):
    name = str(category_name).strip()
    if name == "" or name == "unknown":
        return 0
    if name not in class_to_id:
        class_to_id[name] = len(class_to_id) + 1
    return class_to_id[name]


def build_instance_density_map(label):
    """
    label: 2D instance-id map, background is 0
    returns: density_map(float32), instance_count
    """
    if not isinstance(label, np.ndarray) or label.ndim != 2:
        raise ValueError("label must be a 2D numpy array.")

    density = np.zeros(label.shape, dtype=np.float32)
    instance_ids = np.unique(label)
    instance_ids = instance_ids[instance_ids > 0]
    for inst_id in instance_ids:
        region = (label == inst_id)
        area = int(region.sum())
        if area > 0:
            density[region] = 1.0 / float(area)
    return density, int(len(instance_ids))


def draw_gaussian(heatmap, cx, cy, sigma):
    h, w = heatmap.shape

    if sigma <= 0:
        x = int(round(cx))
        y = int(round(cy))
        if 0 <= x < w and 0 <= y < h:
            heatmap[y, x] = max(heatmap[y, x], 1.0)
        return

    radius = max(1, int(round(3.0 * sigma)))
    x0 = max(0, int(np.floor(cx)) - radius)
    y0 = max(0, int(np.floor(cy)) - radius)
    x1 = min(w, int(np.floor(cx)) + radius + 1)
    y1 = min(h, int(np.floor(cy)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    xs = np.arange(x0, x1, dtype=np.float32)[None, :]
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], g)


def build_center_offset_maps(instance_map, sigma=4.0):
    """
    center_map: HxW float32, one gaussian peak per instance center
    offset_map: HxWx2 float32
      offset_map[...,0] = center_y - y   (dy)
      offset_map[...,1] = center_x - x   (dx)
    """
    if instance_map.ndim != 2:
        raise ValueError("instance_map must be 2D.")

    h, w = instance_map.shape
    center_map = np.zeros((h, w), dtype=np.float32)
    offset_map = np.zeros((h, w, 2), dtype=np.float32)
    centers = []

    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]

    for inst_id in instance_ids:
        ys, xs = np.where(instance_map == inst_id)
        if ys.size == 0:
            continue

        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        centers.append((cx, cy, int(inst_id)))
        draw_gaussian(center_map, cx=cx, cy=cy, sigma=float(sigma))

        offset_map[ys, xs, 0] = cy - ys.astype(np.float32)  # dy
        offset_map[ys, xs, 1] = cx - xs.astype(np.float32)  # dx

    return center_map, offset_map, centers


def build_all_maps_from_instances(instances, height, width, class_to_id, center_sigma):
    instance_map = np.zeros((height, width), dtype=np.int32)
    semantic_map = np.zeros((height, width), dtype=np.int32)

    next_instance_id = 1
    for inst in instances:
        mask = inst["mask"]
        if mask is None or int(mask.sum()) <= 0:
            continue

        class_id = get_or_add_class_id(class_to_id, inst["category"])
        instance_map[mask] = next_instance_id
        semantic_map[mask] = class_id
        next_instance_id += 1

    density_map, instance_count = build_instance_density_map(instance_map)
    center_map, offset_map, centers = build_center_offset_maps(instance_map, sigma=center_sigma)

    return semantic_map, instance_map, density_map, center_map, offset_map, centers, instance_count


def save_class_mapping(class_to_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mapping_path = os.path.join(out_dir, "class_id_mapping.txt")
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("0\tbackground\n")
        for name, idx in sorted(class_to_id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")


def save_single_channel_png(path, arr):
    if arr.ndim != 2:
        raise ValueError("save_single_channel_png expects 2D array.")

    max_val = int(arr.max()) if arr.size > 0 else 0
    if max_val <= 255:
        cv2.imwrite(path, arr.astype(np.uint8))
    elif max_val <= 65535:
        cv2.imwrite(path, arr.astype(np.uint16))
    else:
        # Fallback visualization only for very large ids.
        scaled = (arr.astype(np.float32) / float(max_val) * 65535.0).astype(np.uint16)
        cv2.imwrite(path, scaled)


def heatmap_to_vis(heat):
    if heat.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    nonzero = heat[heat > 0]
    if nonzero.size == 0:
        return np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.uint8)

    log_map = np.log1p(heat)
    vmax = float(np.percentile(np.log1p(nonzero), 99.5))
    if vmax <= 0:
        vmax = float(np.max(np.log1p(nonzero)))
    if vmax <= 0:
        return np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.uint8)

    gray = np.clip((log_map / vmax) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[heat <= 0] = 0
    return color


def id_to_bgr(idx):
    if idx <= 0:
        return (0, 0, 0)
    # Deterministic vivid color from id
    b = (37 * idx + 23) % 256
    g = (17 * idx + 91) % 256
    r = (97 * idx + 53) % 256
    return (int(b), int(g), int(r))


def label_to_color_vis(label_map):
    if label_map.ndim != 2:
        raise ValueError("label_to_color_vis expects 2D array.")
    h, w = label_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    ids = np.unique(label_map)
    ids = ids[ids > 0]
    for idx in ids:
        color[label_map == idx] = id_to_bgr(int(idx))
    return color


def build_center_overlay(image_bgr, centers, radius=3):
    overlay = image_bgr.copy()
    for cx, cy, _ in centers:
        x = int(round(cx))
        y = int(round(cy))
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            cv2.circle(overlay, (x, y), radius=radius, color=(0, 0, 255), thickness=-1)
            cv2.circle(overlay, (x, y), radius=radius + 1, color=(255, 255, 255), thickness=1)
    return overlay


def offset_to_vis(offset_channel):
    vis = np.zeros(offset_channel.shape, dtype=np.uint8)
    nonzero = np.abs(offset_channel[offset_channel != 0])
    if nonzero.size == 0:
        return vis

    vmax = float(np.percentile(nonzero, 99.0))
    if vmax <= 0:
        vmax = float(np.max(nonzero))
    if vmax <= 0:
        return vis

    scaled = np.clip(offset_channel / vmax, -1.0, 1.0)
    vis = ((scaled + 1.0) * 127.5).astype(np.uint8)
    return vis


def main():
    args = parse_args()

    os.makedirs(args.density_dir, exist_ok=True)
    os.makedirs(args.semantic_dir, exist_ok=True)
    os.makedirs(args.instance_dir, exist_ok=True)
    os.makedirs(args.center_dir, exist_ok=True)
    os.makedirs(args.offset_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)

    image_paths = list_images(args.image_dir)
    if len(image_paths) == 0:
        print(f"[INFO] No images found in {args.image_dir}")
        return

    class_to_id = {}
    processed = 0
    skipped = 0

    for image_path in tqdm(image_paths, desc="Building target maps"):
        image_name = os.path.basename(image_path)
        stem = os.path.splitext(image_name)[0]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            print(f"[WARN] Failed to read image: {image_name}")
            continue

        label_path = find_label_path(image_path, args.label_dir, match_suffix=args.match_suffix)
        if label_path is None:
            skipped += 1
            print(f"[WARN] Missing label for image: {image_name}")
            continue

        try:
            label_data = load_json(label_path)
        except Exception as e:
            skipped += 1
            print(f"[WARN] Failed to load label {os.path.basename(label_path)}: {e}")
            continue

        try:
            h, w = get_canvas_size(label_data, image_shape=image.shape)
        except Exception as e:
            skipped += 1
            print(f"[WARN] Invalid size for {image_name}: {e}")
            continue

        if (h, w) != image.shape[:2]:
            print(
                f"[WARN] Size mismatch for {image_name}: "
                f"image={image.shape[1]}x{image.shape[0]}, label={w}x{h}. Use label size."
            )

        instances = extract_instances_from_isat(label_data, h, w)
        semantic_map, instance_map, density_map, center_map, offset_map, centers, instance_count = (
            build_all_maps_from_instances(
                instances=instances,
                height=h,
                width=w,
                class_to_id=class_to_id,
                center_sigma=args.center_sigma,
            )
        )

        density_sum = float(density_map.sum())
        is_close = bool(np.isclose(density_sum, float(instance_count), rtol=1e-3, atol=1e-3))

        np.save(os.path.join(args.density_dir, stem + ".npy"), density_map.astype(np.float32))
        np.save(os.path.join(args.semantic_dir, stem + ".npy"), semantic_map.astype(np.int32))
        np.save(os.path.join(args.instance_dir, stem + ".npy"), instance_map.astype(np.int32))
        np.save(os.path.join(args.center_dir, stem + ".npy"), center_map.astype(np.float32))
        np.save(os.path.join(args.offset_dir, stem + ".npy"), offset_map.astype(np.float32))

        if args.save_vis:
            density_vis = heatmap_to_vis(density_map)
            center_vis = heatmap_to_vis(center_map)
            semantic_vis = label_to_color_vis(semantic_map)
            instance_vis = label_to_color_vis(instance_map)

            vis_base = image
            if vis_base.shape[:2] != (h, w):
                interp = cv2.INTER_AREA if (h * w) < (vis_base.shape[0] * vis_base.shape[1]) else cv2.INTER_LINEAR
                vis_base = cv2.resize(vis_base, (w, h), interpolation=interp)
            center_overlay = build_center_overlay(vis_base, centers, radius=3)

            cv2.imwrite(os.path.join(args.vis_dir, stem + "_density.png"), density_vis)
            cv2.imwrite(os.path.join(args.vis_dir, stem + "_center_heatmap.png"), center_vis)
            cv2.imwrite(os.path.join(args.vis_dir, stem + "_center_overlay.png"), center_overlay)
            cv2.imwrite(os.path.join(args.vis_dir, stem + "_offset_dy.png"), offset_to_vis(offset_map[..., 0]))
            cv2.imwrite(os.path.join(args.vis_dir, stem + "_offset_dx.png"), offset_to_vis(offset_map[..., 1]))
            cv2.imwrite(os.path.join(args.vis_dir, stem + "_semantic.png"), semantic_vis)
            cv2.imwrite(os.path.join(args.vis_dir, stem + "_instance.png"), instance_vis)

        print(
            f"[CHECK] {image_name} | instances={instance_count} | "
            f"density_sum={density_sum:.6f} | consistent={is_close}"
        )
        processed += 1

    save_class_mapping(class_to_id, args.semantic_dir)

    print("-" * 60)
    print(f"[DONE] total_images={len(image_paths)} processed={processed} skipped={skipped}")
    print(f"[DONE] density_dir={args.density_dir}")
    print(f"[DONE] semantic_dir={args.semantic_dir}")
    print(f"[DONE] instance_dir={args.instance_dir}")
    print(f"[DONE] center_dir={args.center_dir}")
    print(f"[DONE] offset_dir={args.offset_dir}")
    if args.save_vis:
        print(f"[DONE] vis_dir={args.vis_dir}")
    print(f"[DONE] class mapping: {os.path.join(args.semantic_dir, 'class_id_mapping.txt')}")


if __name__ == "__main__":
    main()
