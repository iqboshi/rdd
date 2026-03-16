#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
LABEL_EXTS = [".png"]

# RiceSEG canonical class ids (commonly used in labels)
RICESEG_CLASS_NAMES = {
    0: "background",
    1: "green_vegetation",
    2: "senescent_vegetation",
    3: "panicle",
    4: "weeds",
    5: "duckweed",
}
DEFAULT_LEAF_CLASS_IDS = (1, 2)


def resolve_riceseg_root(data_root: Union[str, Path]) -> Path:
    root = Path(data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"RiceSEG root does not exist: {root}")

    if root.name.lower() == "global rice segmentation":
        return root

    candidate = root / "RiceSEG" / "global rice segmentation"
    if candidate.is_dir():
        return candidate

    candidate = root / "global rice segmentation"
    if candidate.is_dir():
        return candidate

    return root


def read_image_rgb(path: Union[str, Path]) -> np.ndarray:
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_label_mask(path: Union[str, Path]) -> np.ndarray:
    p = Path(path)
    mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read label: {p}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"Label must be 2D mask, got shape={mask.shape}, file={p}")
    return mask.astype(np.int64, copy=False)


def map_riceseg_to_leaf_binary(raw_mask: np.ndarray, leaf_class_ids: Sequence[int] = DEFAULT_LEAF_CLASS_IDS) -> np.ndarray:
    raw_mask = np.asarray(raw_mask)
    out = np.zeros(raw_mask.shape, dtype=np.uint8)
    for class_id in leaf_class_ids:
        out[raw_mask == int(class_id)] = 1
    return out


def infer_source_id_from_stem(stem: str) -> str:
    stem = str(stem)
    matched = re.match(r"^(.*?)(?:_subset_overlap_\d+_\d+)$", stem, flags=re.IGNORECASE)
    if matched:
        s = matched.group(1).strip("_")
        return s if s else stem
    return stem


def find_image_for_label(label_path: Path, rgb_dir: Path) -> Optional[Path]:
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        p = rgb_dir / f"{stem}{ext}"
        if p.exists():
            return p
    candidates = sorted(rgb_dir.glob(f"{stem}.*"))
    for p in candidates:
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def collect_riceseg_samples(data_root: Union[str, Path]) -> List[Dict]:
    root = resolve_riceseg_root(data_root)
    samples: List[Dict] = []

    label_files = sorted(root.rglob("label/*.png"))
    for label_path in label_files:
        rgb_dir = label_path.parent.parent / "rgb"
        if not rgb_dir.is_dir():
            continue
        image_path = find_image_for_label(label_path, rgb_dir)
        if image_path is None:
            continue

        rel_label = label_path.relative_to(root)
        parts = rel_label.parts
        country = parts[0] if len(parts) >= 3 else "unknown"
        region = parts[1] if len(parts) >= 4 else ""
        source_stem = infer_source_id_from_stem(label_path.stem)
        group_id = f"{country}/{region}/{source_stem}" if region else f"{country}/{source_stem}"

        samples.append(
            {
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_name": image_path.name,
                "label_name": label_path.name,
                "stem": label_path.stem,
                "country": country,
                "region": region,
                "source_id": source_stem,
                "group_id": group_id,
            }
        )
    return samples


def split_samples_by_source(
    samples: Sequence[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    if not (0.0 <= float(val_ratio) <= 1.0):
        raise ValueError("val_ratio must be in [0, 1]")

    group_to_samples: Dict[str, List[Dict]] = {}
    for s in samples:
        key = str(s.get("group_id", s.get("stem", "")))
        group_to_samples.setdefault(key, []).append(dict(s))

    group_keys = list(group_to_samples.keys())
    if len(group_keys) == 0:
        return [], []
    if len(group_keys) == 1:
        return list(group_to_samples[group_keys[0]]), []

    rng = np.random.default_rng(int(seed))
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_val_groups = int(round(n_groups * float(val_ratio)))
    if val_ratio > 0 and n_val_groups == 0:
        n_val_groups = 1
    if val_ratio < 1 and n_val_groups == n_groups:
        n_val_groups = n_groups - 1

    val_set = set(group_keys[:n_val_groups])
    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    for k in group_keys:
        if k in val_set:
            val_samples.extend(group_to_samples[k])
        else:
            train_samples.extend(group_to_samples[k])

    train_samples.sort(key=lambda x: x["image_name"])
    val_samples.sort(key=lambda x: x["image_name"])
    return train_samples, val_samples


class RiceSegLeafDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict],
        transform=None,
        leaf_class_ids: Sequence[int] = DEFAULT_LEAF_CLASS_IDS,
        return_raw_label: bool = False,
    ):
        self.samples = [dict(s) for s in samples]
        self.transform = transform
        self.leaf_class_ids = tuple(int(x) for x in leaf_class_ids)
        self.return_raw_label = bool(return_raw_label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        image = read_image_rgb(s["image_path"])
        raw_label = read_label_mask(s["label_path"])
        leaf_mask = map_riceseg_to_leaf_binary(raw_label, self.leaf_class_ids)

        if image.shape[:2] != raw_label.shape[:2]:
            raise ValueError(
                f"Image/label shape mismatch: image={image.shape[:2]}, label={raw_label.shape[:2]}, "
                f"image={s['image_path']}, label={s['label_path']}"
            )

        if self.transform is not None:
            transformed = self.transform(image, leaf_mask)
            if isinstance(transformed, dict):
                image = transformed.get("image", image)
                leaf_mask = transformed.get("mask", leaf_mask)
            elif isinstance(transformed, (tuple, list)) and len(transformed) >= 2:
                image, leaf_mask = transformed[0], transformed[1]
            else:
                image = transformed

        if torch.is_tensor(image):
            image_t = image.float()
            if image_t.ndim == 2:
                image_t = image_t.unsqueeze(0)
            elif image_t.ndim == 3 and image_t.shape[0] not in (1, 3, 4) and image_t.shape[-1] in (1, 3, 4):
                image_t = image_t.permute(2, 0, 1)
            if image_t.numel() > 0 and float(image_t.max()) > 1.0:
                image_t = image_t / 255.0
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
            image_t = torch.from_numpy(arr).float() / 255.0

        if torch.is_tensor(leaf_mask):
            mask_t = leaf_mask.long()
            if mask_t.ndim == 3:
                mask_t = mask_t.squeeze(0) if mask_t.shape[0] == 1 else mask_t[:, :, 0]
        else:
            mask_arr = np.asarray(leaf_mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            mask_t = torch.from_numpy(mask_arr.astype(np.int64))

        meta = {
            "index": int(index),
            "image_name": s["image_name"],
            "label_name": s["label_name"],
            "country": s["country"],
            "region": s["region"],
            "source_id": s["source_id"],
            "group_id": s["group_id"],
            "image_path": s["image_path"],
            "label_path": s["label_path"],
            "height": int(mask_t.shape[0]),
            "width": int(mask_t.shape[1]),
        }

        out = {
            "image": image_t.contiguous(),
            "mask": mask_t.contiguous(),
            "meta": meta,
        }
        if self.return_raw_label:
            raw = torch.from_numpy(raw_label.astype(np.int64))
            out["raw_label"] = raw
        return out


def build_train_val_datasets(
    data_root: Union[str, Path],
    train_transform=None,
    val_transform=None,
    val_ratio: float = 0.2,
    seed: int = 42,
    leaf_class_ids: Sequence[int] = DEFAULT_LEAF_CLASS_IDS,
) -> Tuple[RiceSegLeafDataset, RiceSegLeafDataset]:
    samples = collect_riceseg_samples(data_root)
    train_samples, val_samples = split_samples_by_source(samples, val_ratio=val_ratio, seed=seed)
    train_ds = RiceSegLeafDataset(
        samples=train_samples,
        transform=train_transform,
        leaf_class_ids=leaf_class_ids,
        return_raw_label=False,
    )
    val_ds = RiceSegLeafDataset(
        samples=val_samples,
        transform=val_transform,
        leaf_class_ids=leaf_class_ids,
        return_raw_label=False,
    )
    return train_ds, val_ds


def save_mapping_preview(
    data_root: Union[str, Path],
    save_dir: Union[str, Path],
    num_samples: int = 12,
    leaf_class_ids: Sequence[int] = DEFAULT_LEAF_CLASS_IDS,
    seed: int = 42,
):
    rng = random.Random(seed)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_riceseg_samples(data_root)
    if len(samples) == 0:
        print("[WARN] No RiceSEG samples found for preview.")
        return

    picked = samples if len(samples) <= num_samples else rng.sample(samples, num_samples)
    for i, s in enumerate(picked):
        image = read_image_rgb(s["image_path"])
        raw = read_label_mask(s["label_path"])
        leaf = map_riceseg_to_leaf_binary(raw, leaf_class_ids=leaf_class_ids)

        raw_norm = (raw.astype(np.float32) / max(1.0, float(raw.max()))) * 255.0
        raw_color = cv2.applyColorMap(raw_norm.astype(np.uint8), cv2.COLORMAP_TURBO)
        leaf_u8 = (leaf * 255).astype(np.uint8)
        leaf_color = cv2.applyColorMap(leaf_u8, cv2.COLORMAP_SUMMER)
        overlay = cv2.addWeighted(image, 0.65, leaf_color[:, :, ::-1], 0.35, 0.0)

        panel = np.concatenate(
            [
                image,
                raw_color[:, :, ::-1],
                leaf_color[:, :, ::-1],
                overlay,
            ],
            axis=1,
        )
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.putText(panel_bgr, "Image", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel_bgr, "Raw Label", (image.shape[1] + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel_bgr, "Leaf Binary", (image.shape[1] * 2 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel_bgr, "Overlay", (image.shape[1] * 3 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = save_dir / f"preview_{i:03d}_{s['stem']}.png"
        cv2.imwrite(str(out_path), panel_bgr)

    print(f"[INFO] Saved mapping preview to: {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="RiceSEG leaf/non-leaf dataset sanity check.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/external_data/RiceSEG/global rice segmentation",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview_dir", type=str, default="pretrain_riceseg/outputs/preview")
    parser.add_argument("--preview_num", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    samples = collect_riceseg_samples(args.data_root)
    print(f"[INFO] Collected samples: {len(samples)}")
    if len(samples) > 0:
        label_vals = set()
        for s in samples[:200]:
            raw = read_label_mask(s["label_path"])
            label_vals.update(np.unique(raw).tolist())
        print(f"[INFO] Label values sample: {sorted(label_vals)}")

    train_samples, val_samples = split_samples_by_source(samples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[INFO] Train samples: {len(train_samples)}")
    print(f"[INFO] Val samples: {len(val_samples)}")

    save_mapping_preview(
        data_root=args.data_root,
        save_dir=args.preview_dir,
        num_samples=args.preview_num,
        leaf_class_ids=DEFAULT_LEAF_CLASS_IDS,
        seed=args.seed,
    )
