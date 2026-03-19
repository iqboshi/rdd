#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("pretrain_riceseg/model.py requires `timm`.") from e


class SwinBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        input_size: int = 512,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            img_size=self.input_size,
            strict_img_size=True,
        )
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        out_feats: List[torch.Tensor] = []
        for feat, c in zip(feats, self.out_channels):
            if feat.ndim == 4 and feat.shape[-1] == c and feat.shape[1] != c:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out_feats.append(feat)
        return out_feats


class SimpleFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels])
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )
        self.seg_feature_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        if len(feats) != len(self.lateral_convs):
            raise ValueError(f"FPN got {len(feats)} features, expected {len(self.lateral_convs)}")
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        outs = [None] * len(laterals)
        outs[-1] = self.output_convs[-1](laterals[-1])
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(outs[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            outs[i] = self.output_convs[i](laterals[i] + up)
        seg_feature = self.seg_feature_proj(outs[0])
        return {"fpn_feats": outs, "seg_feature": seg_feature}


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_classes: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RiceSegPretrainModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        pretrained: bool = True,
        input_size: int = 512,
        upsample_to_input: bool = True,
    ):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=pretrained, input_size=input_size)
        self.fpn = SimpleFPN(in_channels=self.backbone.out_channels, out_channels=hidden_dim)
        self.seg_head = SegmentationHead(in_channels=hidden_dim, num_classes=num_classes)
        self.upsample_to_input = bool(upsample_to_input)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(image)
        neck = self.fpn(feats)
        logits = self.seg_head(neck["seg_feature"])
        if self.upsample_to_input:
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return {"seg_logits": logits}


def extract_backbone_state_dict_for_instance(pretrain_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Keys will match LeafInstanceSegModel backbone keys: "backbone.backbone.*"
    return {k: v for k, v in pretrain_model_state.items() if k.startswith("backbone.")}


if __name__ == "__main__":
    model = RiceSegPretrainModel(pretrained=False, input_size=512, num_classes=2)
    model.eval()
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print("seg_logits.shape:", out["seg_logits"].shape)
