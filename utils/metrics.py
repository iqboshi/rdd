import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, label):
        """
        Args:
            pred (np.ndarray): [N, H, W]
            label (np.ndarray): [N, H, W]
        """
        mask = (label != self.ignore_index)
        pred = pred[mask]
        label = label[mask]
        
        if len(pred) == 0:
            return

        # Flatten
        pred = pred.flatten()
        label = label.flatten()
        
        # Calculate confusion matrix
        count = np.bincount(self.num_classes * label + pred, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion_matrix

    def get_results(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
        mAcc = np.nanmean(acc_cls)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mIoU = np.nanmean(iou)
        precision = np.diag(hist) / (hist.sum(axis=0) + 1e-10)
        recall = acc_cls
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            "Overall Acc": acc,
            "Mean Acc": mAcc,
            "Mean IoU": mIoU,
            "Class IoU": iou,
            "Class Precision": precision,
            "Class Recall": recall,
            "Class F1": f1
        }

class InstanceMetrics:
    """
    封装 torchmetrics 的 MeanAveragePrecision 用于实例分割评估
    """
    def __init__(self):
        # iou_type="segm" 表示计算掩码的 mAP
        self.metric = MeanAveragePrecision(iou_type="segm")

    def reset(self):
        self.metric.reset()

    def update(self, preds, targets):
        """
        Args:
            preds (list[dict]): 模型输出，包含 boxes, labels, scores, masks
            targets (list[dict]): 真实标签，包含 boxes, labels, masks
        """
        # 确保数据在 CPU 上，torchmetrics 要求
        preds_cpu = []
        for p in preds:
            item = {k: v.detach().cpu() for k, v in p.items()}
            # 确保 masks 是 bool 或 uint8 类型 (pycocotools 要求)
            if 'masks' in item:
                # Mask R-CNN 输出 masks 为 [N, 1, H, W] 的 float (sigmoid 后)
                # torchmetrics 需要 masks 为 [N, H, W] 的 bool/uint8
                masks = item['masks']
                if masks.dim() == 4:
                    masks = masks.squeeze(1)
                item['masks'] = masks > 0.5
            preds_cpu.append(item)
            
        targets_cpu = []
        for t in targets:
            # 过滤掉空的 target (有些图可能没有实例)
            if t["boxes"].shape[0] == 0:
                continue
            item = {k: v.detach().cpu() for k, v in t.items()}
            # 确保 masks 是 bool 或 uint8 类型
            if 'masks' in item:
                item['masks'] = item['masks'].to(torch.uint8)
            targets_cpu.append(item)
            
        if len(targets_cpu) > 0:
            self.metric.update(preds_cpu, targets_cpu)

    def compute(self):
        """
        计算并返回关键指标
        """
        results = self.metric.compute()
        # 提取关键指标并转换为标量
        return {
            "mAP": results["map"].item(),
            "mAP_50": results["map_50"].item(),
            "mAP_75": results["map_75"].item(),
            "mAP_s": results["map_small"].item(),
            "mAP_m": results["map_medium"].item(),
            "mAP_l": results["map_large"].item(),
        }
