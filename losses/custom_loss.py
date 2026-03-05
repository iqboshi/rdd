import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss 实现，用于解决样本不平衡问题
    """
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [N, C, H, W] (模型输出的logits)
        # targets: [N, H, W] (标签索引)
        
        # 在通道维度上进行Softmax
        inputs = F.softmax(inputs, dim=1)
        
        # 获取类别数
        num_classes = inputs.size(1)
        
        # 处理忽略索引 (ignore_index)
        valid_mask = (targets != self.ignore_index)
        targets = targets * valid_mask.long() # 防止索引越界
        
        # 将标签转换为 One-hot 编码: [N, H, W] -> [N, C, H, W]
        target_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # 扩展掩码维度: [N, H, W] -> [N, 1, H, W]
        valid_mask = valid_mask.unsqueeze(1).float()
        
        # 应用掩码，过滤掉无效像素
        inputs = inputs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        # 计算交集和并集
        intersection = (inputs * target_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 返回 1 - Dice 作为损失
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss 实现，用于解决难易样本不平衡问题
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失，不进行归约以便后续加权
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        # 计算预测概率 pt
        pt = torch.exp(-ce_loss)
        # 计算 Focal Loss: alpha * (1-pt)^gamma * ce_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class OHEMCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Cross Entropy Loss
    在线难例挖掘交叉熵损失，针对高光和阴影等难分像素
    """
    def __init__(self, thresh=0.7, min_kept=100000, ignore_index=255):
        super(OHEMCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept # 最少保留的像素数
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets):
        # 计算逐像素损失
        pixel_losses = self.criterion(inputs, targets).contiguous().view(-1)
        
        # 根据损失值排序，选取损失最大的 Top-K 个像素
        # 这里的 top_k 取决于 min_kept，确保至少有一定数量的像素参与反向传播
        top_k = int(self.min_kept)
        if len(pixel_losses) < top_k:
            top_k = len(pixel_losses)
            
        # 获取 Top-K 难例的损失值
        loss_val, _ = torch.topk(pixel_losses, top_k)
        
        return loss_val.mean()

class LabelSmoothingLoss(nn.Module):
    """
    带有标签平滑的交叉熵损失
    """
    def __init__(self, classes, smoothing=0.1, dim=-1, ignore_index=255):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1) # [N, H, W, C]
        pred = F.log_softmax(pred, dim=self.dim)
        
        with torch.no_grad():
            # 创建平滑后的目标分布
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            
            # 处理 ignore_index
            mask = target != self.ignore_index
            target = target * mask.long()
            
            true_dist.scatter_(self.dim, target.unsqueeze(self.dim), self.confidence)
            
            # 将 ignore_index 对应的分布置零
            mask_expanded = mask.unsqueeze(-1).expand_as(true_dist)
            true_dist = true_dist * mask_expanded.float()

        # 计算 KL 散度
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CombinedLoss(nn.Module):
    """
    组合损失函数: Dice Loss + (Focal Loss / OHEM / Label Smoothing)
    """
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.config = config
        self.ignore_index = config.get('ignore_index', 255)
        self.num_classes = config.get('num_classes', 2)
        
        # 1. Dice Loss (始终启用)
        self.dice_loss = DiceLoss(ignore_index=self.ignore_index)
        
        # 2. 辅助损失 (互斥选择: OHEM > Focal > Label Smoothing > Standard CE)
        if config.get('use_ohem', False):
            print("Using OHEM Loss")
            self.ce_loss = OHEMCrossEntropyLoss(ignore_index=self.ignore_index, min_kept=config.get('ohem_min_kept', 100000))
        elif config.get('use_focal', True):
            print("Using Focal Loss")
            self.ce_loss = FocalLoss(
                alpha=config.get('focal_alpha', 0.25),
                gamma=config.get('focal_gamma', 2.0),
                ignore_index=self.ignore_index
            )
        elif config.get('label_smoothing', 0.0) > 0:
            print("Using Label Smoothing Loss")
            self.ce_loss = LabelSmoothingLoss(
                classes=self.num_classes, 
                smoothing=config.get('label_smoothing'), 
                ignore_index=self.ignore_index
            )
        else:
            print("Using Standard Cross Entropy Loss")
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, inputs, targets):
        loss_dice = self.dice_loss(inputs, targets)
        loss_ce = self.ce_loss(inputs, targets)
        
        # 返回加权和 (这里默认 1:1)
        return loss_dice + loss_ce
