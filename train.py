import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from models.builder import build_model
from losses.custom_loss import CombinedLoss
from datasets.rice_dataset import RiceDataset
from datasets.transforms import get_train_transforms, get_val_transforms
from utils.metrics import SegmentationMetrics, InstanceMetrics
from utils.logger import Logger
from utils.seed import seed_everything

def get_args():
    parser = argparse.ArgumentParser(description='Train Rice Disease Detection Model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='Path to config file')
    return parser.parse_args()

def collate_fn(batch):
    return tuple(zip(*batch))

def train(config):
    # Set seed
    seed = config.get('random_seed', 42)
    seed_everything(seed)
    
    # Task Type
    task_type = config.get('task_type', 'semantic')
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Logger
    log_dir = os.path.join('runs', config.get('exp_name', 'default'))
    logger = Logger(log_dir)
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config}")
    logger.info(f"Task Type: {task_type}")

    # Dataset & DataLoader
    data_root = config['data']['root_dir']
    train_list = os.path.join(data_root, config['data']['train_list'])
    val_list = os.path.join(data_root, config['data']['val_list'])
    img_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')
    img_size = config['train']['img_size']
    batch_size = config['train']['batch_size']

    train_dataset = RiceDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        list_path=train_list,
        transform=get_train_transforms(img_size),
        task_type=task_type
    )
    
    val_dataset = RiceDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        list_path=val_list,
        transform=get_val_transforms(img_size),
        task_type=task_type
    )

    # Collate function for instance segmentation
    if task_type == 'instance':
        collate_func = collate_fn
    else:
        collate_func = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_func
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_func
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = build_model(config).to(device)

    # Loss (Only for Semantic, Instance models usually compute loss internally)
    if task_type == 'semantic':
        criterion = CombinedLoss(config['loss']).to(device)
    else:
        criterion = None

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['train']['epochs'],
        eta_min=1e-6
    )

    # AMP Scaler
    scaler = GradScaler()

    # Metrics
    if task_type == 'semantic':
        metrics = SegmentationMetrics(num_classes=config['model']['num_classes'], ignore_index=config['loss']['ignore_index'])
    else:
        metrics = InstanceMetrics()

    # Training Loop
    best_score = 0.0
    epochs = config['train']['epochs']

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (images, targets) in enumerate(pbar):
            # images: Tensor
            # targets: Tensor (semantic) or List[Dict] (instance)
            
            optimizer.zero_grad()
            
            with autocast():
                if task_type == 'semantic':
                    images = images.to(device)
                    masks = targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                elif task_type == 'instance':
                    # targets is tuple of dicts
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    # Ensure images is list of tensors for Mask R-CNN
                    # RiceDataset returns tensor image.
                    # Collate makes it tuple of tensors.
                    # Convert to list
                    images_list = list(images)
                    images_list = [img.to(device) for img in images_list]
                    
                    # Model forward (returns loss dict in train mode)
                    loss_dict = model(images_list, targets)
                    loss = sum(loss for loss in loss_dict.values())
                else:
                    raise NotImplementedError

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)
        
        # Validation
        if task_type == 'semantic':
            val_loss, val_metrics = evaluate_semantic(model, val_loader, criterion, metrics, device)
            val_score = val_metrics['Mean IoU']
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val mIoU: {val_score:.4f}")
            logger.log_scalar('Loss/Train', train_loss, epoch)
            logger.log_scalar('Loss/Val', val_loss, epoch)
            logger.log_scalar('Metric/mIoU', val_score, epoch)
            
        elif task_type == 'instance':
            val_score, val_metrics = evaluate_instance(model, val_loader, metrics, device)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val mAP: {val_score:.4f}")
            logger.log_scalar('Loss/Train', train_loss, epoch)
            logger.log_scalar('Metric/mAP', val_score, epoch)
            logger.log_scalar('Metric/mAP_50', val_metrics['mAP_50'], epoch)
            logger.log_scalar('Metric/mAP_75', val_metrics['mAP_75'], epoch)

        logger.log_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Scheduler Step
        scheduler.step()

        # Checkpointing
        if val_score > best_score:
            best_score = val_score
            save_path = os.path.join(log_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
            }, save_path)
            logger.info(f"Saved best model with score: {best_score:.4f}")

    logger.close()

def evaluate_semantic(model, dataloader, criterion, metrics, device):
    model.eval()
    val_loss = 0.0
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Val]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds.cpu().numpy(), masks.cpu().numpy())
            
    val_loss /= len(dataloader)
    results = metrics.get_results()
    
    return val_loss, results

def evaluate_instance(model, dataloader, metrics, device):
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Val]")
        for images, targets in pbar:
            # targets is tuple of dicts
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            images_list = list(images)
            images_list = [img.to(device) for img in images_list]
            
            # Model forward (returns list of dicts in eval mode)
            outputs = model(images_list)
            
            # Update metrics
            metrics.update(outputs, targets)
            
    results = metrics.compute()
    return results['mAP'], results

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        exit(1)
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train(config)
