#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch


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


def _ensure_2d(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")
    return mask


def _pack(image, mask) -> Dict:
    return {"image": image, "mask": mask}


class Compose:
    def __init__(self, transforms: List):
        self.transforms = list(transforms)

    def __call__(self, image, mask):
        out = _pack(image, mask)
        for t in self.transforms:
            out = t(out["image"], out["mask"])
        return out


class Resize:
    def __init__(self, target_size: Union[int, Sequence[int]]):
        self.target_h, self.target_w = _to_hw(target_size)

    def __call__(self, image, mask):
        image = np.asarray(image)
        mask = _ensure_2d(np.asarray(mask))
        image = cv2.resize(image, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        mask = _ensure_2d(mask)
        return _pack(image, mask)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, image, mask):
        if random.random() >= self.p:
            return _pack(image, mask)
        image = cv2.flip(np.asarray(image), 1)
        mask = cv2.flip(_ensure_2d(np.asarray(mask)), 1)
        return _pack(image, mask)


class RandomVerticalFlip:
    def __init__(self, p: float = 0.1):
        self.p = float(p)

    def __call__(self, image, mask):
        if random.random() >= self.p:
            return _pack(image, mask)
        image = cv2.flip(np.asarray(image), 0)
        mask = cv2.flip(_ensure_2d(np.asarray(mask)), 0)
        return _pack(image, mask)


class RandomRotate90:
    def __init__(self, p: float = 0.25):
        self.p = float(p)

    def __call__(self, image, mask):
        if random.random() >= self.p:
            return _pack(image, mask)
        k = random.choice([1, 2, 3])
        image = np.ascontiguousarray(np.rot90(np.asarray(image), k=k))
        mask = np.ascontiguousarray(np.rot90(_ensure_2d(np.asarray(mask)), k=k))
        return _pack(image, mask)


class RandomBrightnessContrast:
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.p = float(p)

    def __call__(self, image, mask):
        if random.random() >= self.p:
            return _pack(image, mask)
        img = np.asarray(image).astype(np.float32)
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = 255.0 * random.uniform(-self.brightness, self.brightness)
        img = np.clip(img * alpha + beta, 0.0, 255.0).astype(np.uint8)
        return _pack(img, mask)


class RandomGamma:
    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.3):
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.p = float(p)

    def __call__(self, image, mask):
        if random.random() >= self.p:
            return _pack(image, mask)
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        img = np.asarray(image).astype(np.float32) / 255.0
        img = np.clip(np.power(np.clip(img, 0.0, 1.0), gamma), 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        return _pack(img, mask)


class RandomGaussianBlur:
    def __init__(self, p: float = 0.15):
        self.p = float(p)

    def __call__(self, image, mask):
        if random.random() >= self.p:
            return _pack(image, mask)
        ksize = random.choice([3, 5])
        sigma = random.uniform(0.1, 1.2)
        img = cv2.GaussianBlur(np.asarray(image), (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return _pack(img, mask)


class ToTensor:
    def __call__(self, image, mask):
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image_t = torch.from_numpy(image).float() / 255.0

        mask = _ensure_2d(np.asarray(mask))
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return _pack(image_t, mask_t)


def get_train_transform(
    target_size: Union[int, Sequence[int]] = 512,
    hflip_p: float = 0.5,
    vflip_p: float = 0.1,
    rotate90_p: float = 0.25,
    color_jitter_p: float = 0.5,
    gamma_p: float = 0.3,
    blur_p: float = 0.15,
):
    return Compose(
        [
            Resize(target_size),
            RandomHorizontalFlip(hflip_p),
            RandomVerticalFlip(vflip_p),
            RandomRotate90(rotate90_p),
            RandomBrightnessContrast(p=color_jitter_p),
            RandomGamma(p=gamma_p),
            RandomGaussianBlur(p=blur_p),
            ToTensor(),
        ]
    )


def get_val_transform(target_size: Union[int, Sequence[int]] = 512):
    return Compose(
        [
            Resize(target_size),
            ToTensor(),
        ]
    )


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    img = np.random.randint(0, 256, size=(768, 1024, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=(768, 1024), dtype=np.uint8)
    tf = get_train_transform(target_size=512)
    out = tf(img, mask)
    print("image:", out["image"].shape, out["image"].dtype)
    print("mask:", out["mask"].shape, out["mask"].dtype)
