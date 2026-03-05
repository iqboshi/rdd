import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms(img_size=512):
    """
    训练集数据增强:
    - 基础几何变换 (Flip, ShiftScaleRotate)
    - 复杂光照增强 (ColorJitter, RandomBrightnessContrast)
    - 归一化 + Tensor 转换
    
    注意: Mask 插值严格设置为 cv2.INTER_NEAREST
    """
    return A.Compose([
        # 基础几何变换: 随机水平/垂直翻转
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # 随机旋转和平移
        # ShiftScaleRotate (Albumentations 2.0+ 参数调整)
        # mask_interpolation 显式设置为 cv2.INTER_NEAREST (0)
        # value -> fill, mask_value -> fill_mask
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,      # 填充背景色 (黑色)
            fill_mask=0, # Mask 填充值
            mask_interpolation=cv2.INTER_NEAREST, # 严格最近邻插值
            p=0.5
        ),
        
        # 应对复杂光照: 随机色彩抖动
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        
        # 随机亮度对比度
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # 归一化 (使用 ImageNet 均值和标准差)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # 转换为 PyTorch Tensor
        ToTensorV2()
    ])

def get_val_transforms(img_size=512):
    """
    验证集数据处理:
    - 归一化
    - Tensor 转换
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
