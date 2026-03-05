import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .backbones import Backbone
from .modules.aspp import ASPP
from .modules.cbam import CBAM

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 分割网络实现
    支持 EfficientNet-B3 或 ConvNeXt-V2-Tiny 作为骨干网络
    支持 CBAM 注意力机制热插拔
    """
    def __init__(self, num_classes=2, backbone_name='efficientnet_b3', pretrained=True, output_stride=16, use_cbam=False):
        super(DeepLabV3Plus, self).__init__()
        
        # DeepLabV3+ 需要低级特征 (Low-level features) 和高级特征 (High-level features)
        # EfficientNet: 索引 1 (stride 4), 索引 4 (stride 32 或 16)
        # ConvNeXt: 索引 0 (stride 4), 索引 3 (stride 32 或 16)
        
        if 'efficientnet' in backbone_name:
            out_indices = (1, 4)
        elif 'convnext' in backbone_name:
            out_indices = (0, 3)
        else:
            out_indices = (1, 4) # 默认回退

        # 初始化骨干网络
        self.backbone = Backbone(backbone_name, pretrained=pretrained, output_stride=output_stride, out_indices=out_indices)
        channels = self.backbone.channels
        low_level_channels = channels[0] # 低级特征通道数
        high_level_channels = channels[1] # 高级特征通道数

        # ASPP 模块配置
        # output_stride=16 -> rates=[6, 12, 18]
        # output_stride=8 -> rates=[12, 24, 36]
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            atrous_rates = [6, 12, 18]

        self.aspp = ASPP(high_level_channels, 256, atrous_rates=atrous_rates)
        
        # CBAM 注意力机制
        self.use_cbam = use_cbam
        if use_cbam:
            # 在骨干网络输出的高级特征和低级特征处插入 CBAM
            self.cbam_high = CBAM(high_level_channels)
            self.cbam_low = CBAM(low_level_channels)
        
        # 解码器部分
        # 1x1 卷积调整低级特征通道数
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # 最终解码卷积块
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 分类头
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # 骨干网络特征提取
        features = self.backbone(x)
        low_level_features = features[0]
        high_level_features = features[1]
        
        # 如果启用 CBAM，应用注意力机制
        if self.use_cbam:
            high_level_features = self.cbam_high(high_level_features)
            low_level_features = self.cbam_low(low_level_features)
            
        # 通过 ASPP 模块
        x = self.aspp(high_level_features)
        
        # 上采样 ASPP 输出以匹配低级特征尺寸
        x = F.interpolate(x, size=low_level_features.shape[-2:], mode='bilinear', align_corners=True)
        
        # 低级特征投影
        low_level_features = self.low_level_conv(low_level_features)
        
        # 特征拼接
        x = torch.cat([x, low_level_features], dim=1)
        
        # 解码器卷积
        x = self.decoder_conv(x)
        
        # 分类预测
        x = self.classifier(x)
        
        # 上采样至原图尺寸
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        return x

def get_maskrcnn_model(num_classes=2, pretrained=True, min_size=512, max_size=800, trainable_backbone_layers=3):
    """
    构建 Mask R-CNN 模型
    Args:
        num_classes (int): 类别数量 (包含背景)
        pretrained (bool): 是否加载预训练权重 (COCO)
        min_size (int): 输入图像最小尺寸
        max_size (int): 输入图像最大尺寸
        trainable_backbone_layers (int): 可训练的骨干网络层数 (0-5)
    Returns:
        model (nn.Module): Mask R-CNN 模型
    """
    # 加载预训练模型 (ResNet50 + FPN)
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    else:
        weights = None
        
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=trainable_backbone_layers
    )

    # 替换分类头 (Box Predictor)
    # 获取输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 替换为新的分类头
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 替换掩码头 (Mask Predictor)
    # 获取掩码头的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 替换为新的掩码头
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
