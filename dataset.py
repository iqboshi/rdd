#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
INSTANCE_EXTS_PRIORITY = [".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def _normalize_root_dirs(root_dirs: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
    if isinstance(root_dirs, (str, Path)):
        roots = [root_dirs]
    else:
        roots = list(root_dirs)
    return [Path(os.fspath(p)).expanduser() for p in roots]


def load_image(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_instance_map(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        instance_map = np.load(str(path))
    elif suffix in INSTANCE_EXTS_PRIORITY:
        instance_map = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if instance_map is None:
            raise FileNotFoundError(f"Failed to read instance map: {path}")
    else:
        raise ValueError(f"Unsupported instance map file: {path}")

    if instance_map.ndim == 3:
        instance_map = instance_map[:, :, 0]
    if instance_map.ndim != 2:
        raise ValueError(f"instance_map must be 2D, got shape={instance_map.shape} from {path}")

    instance_map = instance_map.astype(np.int64, copy=False)
    instance_map[instance_map < 0] = 0
    return instance_map


def remap_instance_ids(instance_map: np.ndarray) -> np.ndarray:
    instance_map = np.asarray(instance_map, dtype=np.int64)
    remapped = np.zeros_like(instance_map, dtype=np.int64)
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]
    for new_id, old_id in enumerate(instance_ids.tolist(), start=1):
        remapped[instance_map == old_id] = new_id
    return remapped


def build_leaf_semantic(instance_map: np.ndarray) -> np.ndarray:
    instance_map = np.asarray(instance_map)
    semantic = (instance_map > 0).astype(np.int64)
    return semantic


def instance_map_to_gt_masks_labels(
    instance_map: Union[np.ndarray, torch.Tensor],
    leaf_class_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(instance_map):
        inst = instance_map.long()
    else:
        inst = torch.as_tensor(np.asarray(instance_map), dtype=torch.long)

    if inst.ndim != 2:
        raise ValueError(f"instance_map must be 2D, got shape={tuple(inst.shape)}")

    instance_ids = torch.unique(inst)
    instance_ids = instance_ids[instance_ids > 0]
    if instance_ids.numel() == 0:
        h, w = int(inst.shape[0]), int(inst.shape[1])
        empty_masks = torch.zeros((0, h, w), dtype=torch.float32)
        empty_labels = torch.zeros((0,), dtype=torch.long)
        return empty_masks, empty_labels

    masks = []
    for ins_id in instance_ids.tolist():
        masks.append((inst == int(ins_id)).float())
    gt_masks = torch.stack(masks, dim=0).contiguous()
    gt_labels = torch.full(
        (gt_masks.shape[0],),
        fill_value=int(leaf_class_id),
        dtype=torch.long,
    )
    return gt_masks, gt_labels


def infer_patch_scale_from_root(root_dir: Union[str, Path]) -> Optional[int]:
    root_name = Path(root_dir).name
    matched = re.search(r"size[_-]?(\d+)", root_name.lower())
    if matched:
        return int(matched.group(1))
    matched = re.search(r"(\d+)", root_name)
    if matched:
        return int(matched.group(1))
    return None


def infer_big_image_id_from_stem(stem: str) -> str:
    """
    Infer source big-image ID from patch stem.
    Example:
      xxx_x512_y768 -> xxx
      xxx_pp_x0_y0  -> xxx_pp
    """
    stem = str(stem)
    matched = re.match(r"^(.*?)(?:_x-?\d+_y-?\d+)$", stem, flags=re.IGNORECASE)
    if matched:
        prefix = matched.group(1).strip("_")
        return prefix if prefix else stem
    return stem


def normalize_patch_scale_weights(
    patch_scale_weights: Optional[Union[Dict[Union[int, str], float], Sequence[Tuple[Union[int, str], float]]]]
) -> Dict[int, float]:
    if patch_scale_weights is None:
        return {}

    if isinstance(patch_scale_weights, dict):
        items = patch_scale_weights.items()
    else:
        items = list(patch_scale_weights)

    normalized: Dict[int, float] = {}
    for key, value in items:
        scale = int(key)
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"patch scale weight must be > 0, got scale={scale}, weight={weight}")
        normalized[scale] = weight
    return normalized


def collect_samples(
    root_dirs: Union[str, Path, Sequence[Union[str, Path]]],
    skip_empty: bool = False,
) -> List[Dict[str, Union[str, int, None]]]:
    roots = _normalize_root_dirs(root_dirs)
    samples: List[Dict[str, Union[str, int, None]]] = []

    for root in roots:
        images_dir = root / "images"
        instance_dir = root / "instance"
        if not images_dir.is_dir() or not instance_dir.is_dir():
            continue

        image_paths: List[Path] = []
        for path_str in glob.glob(str(images_dir / "*")):
            p = Path(path_str)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                image_paths.append(p)
        image_paths = sorted(image_paths)

        instance_by_stem: Dict[str, Path] = {}
        instance_priority: Dict[str, int] = {}
        for path_str in glob.glob(str(instance_dir / "*")):
            p = Path(path_str)
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in INSTANCE_EXTS_PRIORITY:
                continue
            priority = INSTANCE_EXTS_PRIORITY.index(ext)
            stem = p.stem
            if stem not in instance_by_stem or priority < instance_priority[stem]:
                instance_by_stem[stem] = p
                instance_priority[stem] = priority

        patch_scale = infer_patch_scale_from_root(root)
        for image_path in image_paths:
            stem = image_path.stem
            instance_path = instance_by_stem.get(stem, None)
            if instance_path is None:
                continue

            if skip_empty:
                try:
                    inst = load_instance_map(instance_path)
                except Exception:
                    continue
                if not np.any(inst > 0):
                    continue

            samples.append(
                {
                    "image_name": image_path.name,
                    "image_stem": image_path.stem,
                    "big_image_id": infer_big_image_id_from_stem(image_path.stem),
                    "root_dir": str(root),
                    "patch_scale": patch_scale,
                    "image_path": str(image_path),
                    "instance_path": str(instance_path),
                }
            )

    samples.sort(key=lambda x: (str(x["root_dir"]), str(x["image_name"])))
    return samples


def split_samples_by_big_image(
    samples: Sequence[Dict[str, Union[str, int, None]]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Union[str, int, None]]], List[Dict[str, Union[str, int, None]]]]:
    if not (0.0 <= float(val_ratio) <= 1.0):
        raise ValueError("val_ratio must be in [0, 1]")

    group_to_samples: Dict[str, List[Dict[str, Union[str, int, None]]]] = {}
    for s in samples:
        group_key = str(s.get("big_image_id", s.get("image_stem", s.get("image_name", ""))))
        group_to_samples.setdefault(group_key, []).append(dict(s))

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

    val_group_set = set(group_keys[:n_val_groups])
    train_samples: List[Dict[str, Union[str, int, None]]] = []
    val_samples: List[Dict[str, Union[str, int, None]]] = []

    for key in group_keys:
        if key in val_group_set:
            val_samples.extend(group_to_samples[key])
        else:
            train_samples.extend(group_to_samples[key])

    train_samples.sort(key=lambda x: (str(x["root_dir"]), str(x["image_name"])))
    val_samples.sort(key=lambda x: (str(x["root_dir"]), str(x["image_name"])))
    return train_samples, val_samples


class LeafOnlyInstanceDataset(Dataset):
    def __init__(
        self,
        root_dirs: Union[str, Path, Sequence[Union[str, Path]]],
        transform=None,
        remap_instance_ids: bool = False,
        skip_empty: bool = False,
        patch_scales: Optional[Sequence[int]] = None,
        enable_patch_scale_weighting: bool = False,
        patch_scale_weights: Optional[Union[Dict[Union[int, str], float], Sequence[Tuple[Union[int, str], float]]]] = None,
        samples: Optional[Sequence[Dict[str, Union[str, int, None]]]] = None,
    ):
        self.root_dirs = _normalize_root_dirs(root_dirs)
        self.transform = transform
        self.remap_instance_ids_enabled = bool(remap_instance_ids)
        self.skip_empty = bool(skip_empty)
        self.patch_scales = None if patch_scales is None else {int(s) for s in patch_scales}
        self.enable_patch_scale_weighting = bool(enable_patch_scale_weighting)
        self.patch_scale_weights = normalize_patch_scale_weights(patch_scale_weights)

        if samples is None:
            self.samples = collect_samples(self.root_dirs, skip_empty=self.skip_empty)
        else:
            self.samples = [dict(s) for s in samples]
        if self.patch_scales is not None:
            self.samples = [s for s in self.samples if s["patch_scale"] in self.patch_scales]
        self.sample_weights = self._build_sample_weights() if self.enable_patch_scale_weighting else None

    def __len__(self) -> int:
        return len(self.samples)

    def _build_sample_weights(self) -> torch.Tensor:
        if len(self.samples) == 0:
            return torch.zeros((0,), dtype=torch.double)

        weights = []
        for sample in self.samples:
            scale = sample.get("patch_scale", None)
            if scale is None:
                weights.append(1.0)
            else:
                weights.append(float(self.patch_scale_weights.get(int(scale), 1.0)))
        return torch.as_tensor(weights, dtype=torch.double)

    def get_sample_weights(self) -> Optional[torch.Tensor]:
        if self.sample_weights is None:
            return None
        return self.sample_weights.clone()

    def build_weighted_sampler(
        self,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ) -> Optional[WeightedRandomSampler]:
        if self.sample_weights is None:
            return None
        if len(self.sample_weights) == 0:
            return None
        if num_samples is None:
            num_samples = len(self.samples)
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=int(num_samples),
            replacement=bool(replacement),
        )

    def get_patch_scale_sampling_summary(self) -> Dict[int, Dict[str, float]]:
        summary: Dict[int, Dict[str, float]] = {}
        if len(self.samples) == 0:
            return summary

        for idx, sample in enumerate(self.samples):
            scale = sample.get("patch_scale", None)
            if scale is None:
                continue
            scale = int(scale)
            item = summary.setdefault(scale, {"count": 0, "weight": 1.0})
            item["count"] += 1
            if self.sample_weights is not None:
                item["weight"] = float(self.sample_weights[idx].item())
            else:
                item["weight"] = float(self.patch_scale_weights.get(scale, 1.0))
        return summary

    def _apply_transform(self, image, semantic_mask, instance_map):
        if self.transform is None:
            return image, semantic_mask, instance_map

        try:
            transformed = self.transform(
                image=image,
                semantic_mask=semantic_mask,
                instance_map=instance_map,
            )
        except TypeError:
            transformed = self.transform(image, semantic_mask, instance_map)

        if isinstance(transformed, dict):
            image = transformed.get("image", image)
            semantic_mask = transformed.get("semantic_mask", transformed.get("mask", semantic_mask))
            instance_map = transformed.get("instance_map", transformed.get("instance", instance_map))
        elif isinstance(transformed, (tuple, list)):
            if len(transformed) >= 3:
                image, semantic_mask, instance_map = transformed[0], transformed[1], transformed[2]
            elif len(transformed) == 2:
                image, semantic_mask = transformed[0], transformed[1]
            elif len(transformed) == 1:
                image = transformed[0]
        else:
            image = transformed

        return image, semantic_mask, instance_map

    @staticmethod
    def _to_image_tensor(image) -> torch.Tensor:
        if torch.is_tensor(image):
            t = image
            if t.ndim == 2:
                t = t.unsqueeze(0)
            elif t.ndim == 3 and t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
                t = t.permute(2, 0, 1)
            t = t.float()
            if t.numel() > 0 and float(t.max()) > 1.0:
                t = t / 255.0
            return t.contiguous()

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"image must be HxW or HxWxC, got shape={arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        t = torch.from_numpy(arr).float()
        if t.numel() > 0 and float(t.max()) > 1.0:
            t = t / 255.0
        return t

    @staticmethod
    def _to_long_2d_tensor(mask, name: str) -> torch.Tensor:
        if torch.is_tensor(mask):
            t = mask
            if t.ndim == 3:
                if t.shape[0] == 1:
                    t = t.squeeze(0)
                elif t.shape[-1] == 1:
                    t = t.squeeze(-1)
                else:
                    t = t[0]
            if t.ndim != 2:
                raise ValueError(f"{name} must be 2D after conversion, got shape={tuple(t.shape)}")
            return t.long().contiguous()

        arr = np.asarray(mask)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            else:
                arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D after conversion, got shape={arr.shape}")
        arr = np.ascontiguousarray(arr.astype(np.int64))
        return torch.from_numpy(arr)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = Path(sample["image_path"])
        instance_path = Path(sample["instance_path"])

        image = load_image(image_path)
        instance_map = load_instance_map(instance_path)

        if image.shape[:2] != instance_map.shape[:2]:
            raise ValueError(
                f"Image/instance shape mismatch: image={image.shape[:2]}, "
                f"instance={instance_map.shape[:2]}, stem={image_path.stem}"
            )

        if self.remap_instance_ids_enabled:
            instance_map = remap_instance_ids(instance_map)

        semantic_mask = build_leaf_semantic(instance_map)

        image, semantic_mask, instance_map = self._apply_transform(image, semantic_mask, instance_map)

        image_tensor = self._to_image_tensor(image)
        instance_tensor = self._to_long_2d_tensor(instance_map, name="instance_map")
        semantic_tensor = (instance_tensor > 0).long()

        unique_ids = torch.unique(instance_tensor)
        unique_ids = unique_ids[unique_ids > 0]
        instance_ids = unique_ids.cpu().numpy().astype(np.int64).tolist()
        gt_masks, gt_labels = instance_map_to_gt_masks_labels(instance_tensor, leaf_class_id=0)

        height, width = int(instance_tensor.shape[0]), int(instance_tensor.shape[1])
        meta = {
            "image_name": str(sample["image_name"]),
            "image_stem": str(sample.get("image_stem", image_path.stem)),
            "big_image_id": str(sample.get("big_image_id", infer_big_image_id_from_stem(image_path.stem))),
            "root_dir": str(sample["root_dir"]),
            "patch_scale": sample["patch_scale"],
            "index": int(index),
            "height": height,
            "width": width,
            "image_path": str(sample["image_path"]),
            "instance_path": str(sample["instance_path"]),
        }

        return {
            "image": image_tensor,
            "semantic_mask": semantic_tensor,
            "instance_map": instance_tensor,
            "instance_ids": instance_ids,
            "gt_masks": gt_masks,
            "gt_labels": gt_labels,
            "meta": meta,
        }


def build_train_val_datasets_by_big_image(
    root_dirs: Union[str, Path, Sequence[Union[str, Path]]],
    train_transform=None,
    val_transform=None,
    val_ratio: float = 0.2,
    seed: int = 42,
    remap_instance_ids: bool = False,
    skip_empty: bool = False,
    patch_scales: Optional[Sequence[int]] = None,
    enable_patch_scale_weighting: bool = False,
    patch_scale_weights: Optional[Union[Dict[Union[int, str], float], Sequence[Tuple[Union[int, str], float]]]] = None,
) -> Tuple[LeafOnlyInstanceDataset, LeafOnlyInstanceDataset]:
    all_samples = collect_samples(root_dirs=root_dirs, skip_empty=skip_empty)
    if patch_scales is not None:
        scale_set = {int(s) for s in patch_scales}
        all_samples = [s for s in all_samples if s["patch_scale"] in scale_set]

    train_samples, val_samples = split_samples_by_big_image(
        samples=all_samples,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_ds = LeafOnlyInstanceDataset(
        root_dirs=root_dirs,
        transform=train_transform,
        remap_instance_ids=remap_instance_ids,
        skip_empty=skip_empty,
        patch_scales=patch_scales,
        enable_patch_scale_weighting=enable_patch_scale_weighting,
        patch_scale_weights=patch_scale_weights,
        samples=train_samples,
    )
    val_ds = LeafOnlyInstanceDataset(
        root_dirs=root_dirs,
        transform=val_transform,
        remap_instance_ids=remap_instance_ids,
        skip_empty=skip_empty,
        patch_scales=patch_scales,
        enable_patch_scale_weighting=False,
        patch_scale_weights=None,
        samples=val_samples,
    )
    return train_ds, val_ds


if __name__ == "__main__":
    default_roots = [
        "data/patches_size512",
        "data/patches_size768",
        "data/patches_size1024",
    ]

    print("=== Single-scale dataset ===")
    ds_single = LeafOnlyInstanceDataset(default_roots[0], remap_instance_ids=False, skip_empty=False)
    print(f"length: {len(ds_single)}")

    if len(ds_single) > 0:
        sample = ds_single[0]
        print(f"image shape: {tuple(sample['image'].shape)}")
        print(f"semantic shape: {tuple(sample['semantic_mask'].shape)}")
        print(f"instance shape: {tuple(sample['instance_map'].shape)}")
        print(f"instance count: {len(sample['instance_ids'])}")
        print(f"patch_scale: {sample['meta']['patch_scale']}")
        print(f"root_dir: {sample['meta']['root_dir']}")
    else:
        print("No sample found in single-scale dataset.")

    print("\n=== Multi-scale dataset ===")
    ds_multi = LeafOnlyInstanceDataset(default_roots, remap_instance_ids=False, skip_empty=False)
    print(f"length: {len(ds_multi)}")

    if len(ds_multi) > 0:
        sample = ds_multi[0]
        print(f"image shape: {tuple(sample['image'].shape)}")
        print(f"semantic shape: {tuple(sample['semantic_mask'].shape)}")
        print(f"instance shape: {tuple(sample['instance_map'].shape)}")
        print(f"instance count: {len(sample['instance_ids'])}")
        print(f"patch_scale: {sample['meta']['patch_scale']}")
        print(f"root_dir: {sample['meta']['root_dir']}")
    else:
        print("No sample found in multi-scale dataset.")

    print("\n=== Weighted sampling summary ===")
    ds_weighted = LeafOnlyInstanceDataset(
        default_roots,
        remap_instance_ids=False,
        skip_empty=False,
        enable_patch_scale_weighting=True,
        patch_scale_weights={512: 1.0, 768: 1.5, 1024: 2.0},
    )
    print(ds_weighted.get_patch_scale_sampling_summary())
    print(f"weighted sampler enabled: {ds_weighted.build_weighted_sampler() is not None}")

    print("\n=== Train/Val split by big image ===")
    train_ds, val_ds = build_train_val_datasets_by_big_image(
        root_dirs=default_roots,
        val_ratio=0.2,
        seed=42,
        remap_instance_ids=False,
        skip_empty=False,
    )
    print(f"train length: {len(train_ds)}")
    print(f"val length: {len(val_ds)}")

    train_big_ids = {str(s.get('big_image_id', '')) for s in train_ds.samples}
    val_big_ids = {str(s.get('big_image_id', '')) for s in val_ds.samples}
    overlap = train_big_ids.intersection(val_big_ids)
    print(f"big-image overlap count: {len(overlap)}")
