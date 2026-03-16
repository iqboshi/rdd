#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("model.py requires `timm`. Please install timm in your environment.") from e


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
            # timm Swin features may be returned as [B, H, W, C].
            if feat.ndim == 4 and feat.shape[-1] == c and feat.shape[1] != c:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out_feats.append(feat)
        return out_feats


class SimpleFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels]
        )
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
        self.mask_feature_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        if len(feats) != len(self.lateral_convs):
            raise ValueError(f"FPN got {len(feats)} features, expected {len(self.lateral_convs)}")

        laterals = [lat_conv(f) for lat_conv, f in zip(self.lateral_convs, feats)]
        outs = [None] * len(laterals)
        outs[-1] = self.output_convs[-1](laterals[-1])

        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(outs[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            outs[i] = self.output_convs[i](laterals[i] + up)

        mask_feature = self.mask_feature_proj(outs[0])
        return {"fpn_feats": outs, "mask_feature": mask_feature}


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryMaskHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 50,
        num_classes: int = 2,
        mask_embed_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.mask_embed_dim = mask_embed_dim

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_embed_dim, num_layers=3)
        self.pixel_embed = nn.Conv2d(hidden_dim, mask_embed_dim, kernel_size=1)

    def forward(self, mask_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = mask_feature.shape
        memory = mask_feature.flatten(2).transpose(1, 2)  # [B, HW, C]

        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, Q, C]
        attn_out, _ = self.cross_attn(query=q, key=memory, value=memory)
        q = self.norm1(q + attn_out)
        q2 = self.ffn(q)
        q = self.norm2(q + q2)

        pred_logits = self.class_embed(q)  # [B, Q, num_classes]
        query_mask_embed = self.mask_embed(q)  # [B, Q, mask_embed_dim]
        pixel_embed = self.pixel_embed(mask_feature)  # [B, mask_embed_dim, H, W]
        pred_masks = torch.einsum("bqc,bchw->bqhw", query_mask_embed, pixel_embed)

        return {"pred_logits": pred_logits, "pred_masks": pred_masks}


class LeafInstanceSegModel(nn.Module):
    def __init__(
        self,
        num_queries: int = 50,
        hidden_dim: int = 256,
        num_classes: int = 2,
        mask_embed_dim: int = 256,
        pretrained: bool = True,
        input_size: int = 512,
        upsample_masks_to_input: bool = True,
    ):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=pretrained, input_size=input_size)
        self.fpn = SimpleFPN(in_channels=self.backbone.out_channels, out_channels=hidden_dim)
        self.mask_head = QueryMaskHead(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            mask_embed_dim=mask_embed_dim,
        )
        self.upsample_masks_to_input = upsample_masks_to_input

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(image)
        neck_out = self.fpn(feats)
        head_out = self.mask_head(neck_out["mask_feature"])

        pred_masks = head_out["pred_masks"]
        if self.upsample_masks_to_input:
            pred_masks = F.interpolate(
                pred_masks,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return {
            "pred_logits": head_out["pred_logits"],
            "pred_masks": pred_masks,
        }


if __name__ == "__main__":
    model = LeafInstanceSegModel(
        num_queries=50,
        hidden_dim=256,
        num_classes=2,
        mask_embed_dim=256,
        pretrained=False,
        input_size=512,
        upsample_masks_to_input=True,
    )
    model.eval()

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    print("pred_logits.shape:", out["pred_logits"].shape)
    print("pred_masks.shape:", out["pred_masks"].shape)
