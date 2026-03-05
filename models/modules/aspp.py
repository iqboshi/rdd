import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        # 1x1 Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 3x3 Conv with different dilation rates
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)
