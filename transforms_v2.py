#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_hw(target_size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(target_size, int):
        if target_size <= 0:
            raise ValueError("target_size must be > 0")
        return target_size, target_size
    if isinstance(target_size, (tuple, list)) and len(target_size) == 2:
        h, w = int(target_size[0]), int(target_size[1])
        if h <= 0 or w <= 0:
            raise ValueError("target_size values must be > 0")
        return h, w
    raise ValueError("target_size must be int or (H, W)")


def _ensure_2d(mask: np.ndarray, name: str) -> np.ndarray:
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        else:
            mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={mask.shape}")
    return mask


def _pack_output(image, semantic_mask, instance_map) -> Dict[str, ArrayLike]:
    return {
        "image": image,
        "semantic_mask": semantic_mask,
        "instance_map": instance_map,
    }


class Compose:
    def __init__(self, transforms: List):
        self.transforms = list(transforms)

    def __call__(self, image, semantic_mask, instance_map):
        data = _pack_output(image, semantic_mask, instance_map)
        for t in self.transforms:
            data = t(data["image"], data["semantic_mask"], data["instance_map"])
        return data


class ResizeTransform:
    def __init__(self, target_size: Union[int, Sequence[int]]):
        self.target_h, self.target_w = _to_hw(target_size)

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray, instance_map: np.ndarray):
        image = np.asarray(image)
        semantic_mask = _ensure_2d(np.asarray(semantic_mask), "semantic_mask")
        instance_map = _ensure_2d(np.asarray(instance_map), "instance_map")

        image_resized = cv2.resize(
            image,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        semantic_resized = cv2.resize(
            semantic_mask,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        instance_resized = cv2.resize(
            instance_map,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        semantic_resized = _ensure_2d(np.asarray(semantic_resized), "semantic_mask")
        instance_resized = _ensure_2d(np.asarray(instance_resized), "instance_map")
        return _pack_output(image_resized, semantic_resized, instance_resized)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray, instance_map: np.ndarray):
        if random.random() >= self.p:
            return _pack_output(image, semantic_mask, instance_map)

        image = cv2.flip(np.asarray(image), 1)
        semantic_mask = cv2.flip(_ensure_2d(np.asarray(semantic_mask), "semantic_mask"), 1)
        instance_map = cv2.flip(_ensure_2d(np.asarray(instance_map), "instance_map"), 1)
        return _pack_output(image, semantic_mask, instance_map)


class RandomVerticalFlip:
    def __init__(self, p: float = 0.1):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray, instance_map: np.ndarray):
        if random.random() >= self.p:
            return _pack_output(image, semantic_mask, instance_map)

        image = cv2.flip(np.asarray(image), 0)
        semantic_mask = cv2.flip(_ensure_2d(np.asarray(semantic_mask), "semantic_mask"), 0)
        instance_map = cv2.flip(_ensure_2d(np.asarray(instance_map), "instance_map"), 0)
        return _pack_output(image, semantic_mask, instance_map)


class RandomRotate90:
    def __init__(self, p: float = 0.3):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.p = float(p)

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray, instance_map: np.ndarray):
        if random.random() >= self.p:
            return _pack_output(image, semantic_mask, instance_map)

        k = random.choice([1, 2, 3])
        image = np.ascontiguousarray(np.rot90(np.asarray(image), k=k))
        semantic_mask = np.ascontiguousarray(np.rot90(_ensure_2d(np.asarray(semantic_mask), "semantic_mask"), k=k))
        instance_map = np.ascontiguousarray(np.rot90(_ensure_2d(np.asarray(instance_map), "instance_map"), k=k))
        return _pack_output(image, semantic_mask, instance_map)


class RandomBrightnessContrast:
    def __init__(self, brightness: float = 0.25, contrast: float = 0.25, p: float = 0.7):
        if brightness < 0 or contrast < 0:
            raise ValueError("brightness and contrast must be >= 0")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.p = float(p)

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray, instance_map: np.ndarray):
        if random.random() >= self.p:
            return _pack_output(image, semantic_mask, instance_map)

        img = np.asarray(image).astype(np.float32)
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = 255.0 * random.uniform(-self.brightness, self.brightness)
        img = img * alpha + beta
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return _pack_output(img, semantic_mask, instance_map)


class RandomGamma:
    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.4):
        if gamma_range[0] <= 0 or gamma_range[1] <= 0:
            raise ValueError("gamma_range values must be > 0")
        if gamma_range[0] > gamma_range[1]:
            raise ValueError("gamma_range min must be <= max")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.p = float(p)

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray, instance_map: np.ndarray):
        if random.random() >= self.p:
            return _pack_output(image, semantic_mask, instance_map)

        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        img = np.asarray(image).astype(np.float32) / 255.0
        img = np.power(np.clip(img, 0.0, 1.0), gamma)
        img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        return _pack_output(img, semantic_mask, instance_map)


class RandomGaussianBlur:
    def __init__(self, p: float = 0.2, ksize_choices: Tuple[int, ...] = (3, 5), sigma_range: Tuple[float, float] = (0.1, 1.2)):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        if len(ksize_choices) == 0:
            raise ValueError("ksize_choices must not be empty")
        for k in ksize_choices:
            if int(k) <= 0 or int(k) % 2 == 0:
                raise ValueError("Gaussian blur kernel size must be odd and > 0")
        if sigma_range[0] < 0 or sigma_range[1] < 0 or sigma_range[0] > sigma_range[1]:
            raise ValueError("Invalid sigma_range")
        self.p = float(p)
        self.ksize_choices = tuple(int(k) for k in ksize_choices)
        self.sigma_range = (float(sigma_range[0]), float(sigma_range[1]))

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray, instance_map: np.ndarray):
        if random.random() >= self.p:
            return _pack_output(image, semantic_mask, instance_map)

        ksize = random.choice(self.ksize_choices)
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        img = cv2.GaussianBlur(np.asarray(image), (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return _pack_output(img, semantic_mask, instance_map)


class ToTensor:
    def __init__(self, semantic_dtype=torch.long, instance_dtype=torch.long):
        self.semantic_dtype = semantic_dtype
        self.instance_dtype = instance_dtype

    def __call__(self, image: np.ndarray, semantic_mask: np.ndarray, instance_map: np.ndarray):
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim != 3:
            raise ValueError(f"image must be HxWxC, got shape={image.shape}")

        semantic_mask = _ensure_2d(np.asarray(semantic_mask), "semantic_mask")
        instance_map = _ensure_2d(np.asarray(instance_map), "instance_map")

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image_tensor = torch.from_numpy(image).float() / 255.0
        semantic_tensor = torch.from_numpy((semantic_mask > 0).astype(np.int64)).to(self.semantic_dtype)
        instance_tensor = torch.from_numpy(instance_map.astype(np.int64)).to(self.instance_dtype)
        return _pack_output(image_tensor, semantic_tensor, instance_tensor)


def get_train_transform(
    target_size: Union[int, Sequence[int]] = 512,
    hflip_p: float = 0.5,
    vflip_p: float = 0.12,
    rotate90_p: float = 0.35,
    color_jitter_p: float = 0.7,
    brightness: float = 0.25,
    contrast: float = 0.25,
    gamma_p: float = 0.4,
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    blur_p: float = 0.2,
):
    transforms = [
        ResizeTransform(target_size=target_size),
        RandomHorizontalFlip(p=hflip_p),
        RandomVerticalFlip(p=vflip_p),
        RandomRotate90(p=rotate90_p),
        RandomBrightnessContrast(
            brightness=brightness,
            contrast=contrast,
            p=color_jitter_p,
        ),
        RandomGamma(gamma_range=gamma_range, p=gamma_p),
        RandomGaussianBlur(p=blur_p),
        ToTensor(),
    ]
    return Compose(transforms)


def get_val_transform(target_size: Union[int, Sequence[int]] = 512):
    return Compose(
        [
            ResizeTransform(target_size=target_size),
            ToTensor(),
        ]
    )


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    h, w = 768, 1024
    image = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)

    instance_map = np.zeros((h, w), dtype=np.int32)
    instance_map[50:220, 100:320] = 1
    instance_map[260:520, 420:760] = 2
    instance_map[540:730, 800:980] = 3
    semantic_mask = (instance_map > 0).astype(np.uint8)

    train_tf = get_train_transform(target_size=512)
    out = train_tf(image, semantic_mask, instance_map)

    print("image:", out["image"].shape, out["image"].dtype)
    print("semantic_mask:", out["semantic_mask"].shape, out["semantic_mask"].dtype)
    print("instance_map:", out["instance_map"].shape, out["instance_map"].dtype)
