import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化和最大池化共享权重的多层感知机
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP
        # 降维再升维，ratio控制压缩比
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 两个分支相加后通过Sigmoid激活
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # kernel_size必须是3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # 将平均池化和最大池化的结果在通道维度拼接，输入通道为2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上做平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 卷积 + Sigmoid
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    包含通道注意力(Channel Attention)和空间注意力(Spatial Attention)
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先经过通道注意力
        out = x * self.ca(x)
        # 再经过空间注意力
        out = out * self.sa(out)
        return out
