import argparse
import os
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.builder import build_model
from datasets.rice_dataset import RiceDataset
from datasets.transforms import get_val_transforms
from utils.seed import seed_everything

def get_args():
    parser = argparse.ArgumentParser(description='Predict and Visualize Rice Disease Detection')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='runs/default/best_model.pth', help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for instance segmentation')
    return parser.parse_args()

def collate_fn(batch):
    return tuple(zip(*batch))

def predict(args):
    # Load config
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set seed
    seed = config.get('random_seed', 42)
    seed_everything(seed)
    
    # Task Type
    task_type = config.get('task_type', 'semantic')
    print(f"Task Type: {task_type}")
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Dataset & DataLoader
    # We use validation set for prediction/visualization
    data_root = config['data']['root_dir']
    val_list = os.path.join(data_root, config['data']['val_list'])
    img_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')
    img_size = config['train']['img_size']
    
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

    # Use batch_size=1 for easier visualization
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False, # Deterministic order
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_func
    )
    
    # Model
    # For inference, we need to handle pretrained loading carefully.
    # build_model usually loads pretrained backbone weights.
    # We want to load our checkpoint later.
    # However, build_model(config) is fine, we will overwrite weights.
    model = build_model(config).to(device)
    
    # Load Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
        
    print(f"Loading checkpoint: {args.checkpoint}")
    # Fix for PyTorch 2.6+ weights_only default
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    score_key = 'best_miou' if 'best_miou' in checkpoint else 'best_score'
    print(f"Loaded model from epoch {checkpoint['epoch']} with score {checkpoint.get(score_key, 0.0):.4f}")
    
    model.eval()
    
    # Prediction Loop
    count = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader, desc="Predicting")):
            if count >= args.num_samples:
                break
                
            if task_type == 'semantic':
                image = images.to(device)
                mask = targets # [1, H, W]
                
                output = model(image)
                # output: [1, C, H, W] -> argmax -> [1, H, W]
                pred = torch.argmax(output, dim=1)
                
                mask_np = mask.cpu().squeeze(0).numpy()
                pred_np = pred.cpu().squeeze(0).numpy()
                
                # Get numpy image for visualization
                img_tensor = image.cpu().squeeze(0)
                
            elif task_type == 'instance':
                # images is tuple of tensors (batch_size=1) -> images[0]
                image = images[0].to(device)
                # targets is tuple of dicts -> targets[0]
                target = targets[0]
                
                # Model forward (returns list of dicts in eval mode)
                # Input to model must be list of tensors
                outputs = model([image])
                output = outputs[0] # Single image
                
                # Process Prediction
                # Filter by score threshold
                scores = output['scores'].cpu().numpy()
                keep = scores > args.conf_threshold
                
                # Masks: [N, 1, H, W] -> [N, H, W] (bool)
                pred_masks = output['masks'][keep].cpu().squeeze(1).numpy() > 0.5
                
                # Combine masks into single segmentation map for visualization
                # For instance segmentation, we want to see individual instances
                # Let's assign different colors to different instances
                # Or just use a simple mask for now, but user complained it's black and white semantic
                # Let's return list of masks for visualization
                
                # We will change visualize function to handle instance visualization better
                # Instead of flattening to semantic mask, we pass the list of masks
                
                if len(pred_masks) > 0:
                    pred_res = {'masks': pred_masks, 'boxes': output['boxes'][keep].cpu().numpy(), 'scores': scores[keep]}
                else:
                    pred_res = {'masks': [], 'boxes': [], 'scores': []}
                
                gt_masks = target['masks'].cpu().numpy() # [N, H, W]
                gt_boxes = target['boxes'].cpu().numpy() # [N, 4]
                if len(gt_masks) > 0:
                    mask_res = {'masks': gt_masks, 'boxes': gt_boxes}
                else:
                    mask_res = {'masks': [], 'boxes': []}
                    
                img_tensor = image.cpu()

            # Convert image to numpy for visualization
            # Denormalize image: (image * std + mean) * 255
            # Mean/Std from transforms
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            # Visualization
            if task_type == 'semantic':
                visualize_semantic(img_np, mask_np, pred_np, os.path.join(args.output_dir, f"sample_{i}.png"))
            elif task_type == 'instance':
                visualize_instance(img_np, mask_res, pred_res, os.path.join(args.output_dir, f"sample_{i}.png"))
            
            count += 1
            
    print(f"Saved {count} visualizations to {args.output_dir}")

def visualize_semantic(image, mask, pred, save_path):
    """
    Visualize Semantic Segmentation
    """
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    plt.title("Ground Truth (Semantic)")
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
    plt.title("Prediction (Semantic)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_instance(image, target, pred, save_path):
    """
    Visualize Instance Segmentation with Bounding Boxes and Masks
    """
    import matplotlib.patches as patches
    import random
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    ax[0].axis('off')
    
    # Ground Truth
    ax[1].imshow(image)
    ax[1].set_title(f"Ground Truth ({len(target['masks'])} instances)")
    ax[1].axis('off')
    
    colors = []
    for _ in range(100):
        colors.append((random.random(), random.random(), random.random()))
        
    # Draw GT
    if len(target['masks']) > 0:
        for i, (mask, box) in enumerate(zip(target['masks'], target['boxes'])):
            color = colors[i % len(colors)]
            
            # Mask
            masked_image = np.ma.masked_where(mask == 0, mask)
            ax[1].imshow(masked_image, alpha=0.5, cmap=plt.cm.colors.ListedColormap([color]), interpolation='none')
            
            # Box (Removed for GT as well to be cleaner, or keep it?)
            # User asked to remove bbox from prediction results. Let's keep GT box for reference or remove both?
            # User said "prediction results". I will remove from prediction only as requested, 
            # but usually it's better to be consistent. 
            # Let's remove both for a cleaner "Segmentation" view.
            # x, y, x2, y2 = box
            # w, h = x2 - x, y2 - y
            # rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            # ax[1].add_patch(rect)
            
    # Prediction
    ax[2].imshow(image)
    ax[2].set_title(f"Prediction ({len(pred['masks'])} instances)")
    ax[2].axis('off')
    
    if len(pred['masks']) > 0:
        for i, (mask, box, score) in enumerate(zip(pred['masks'], pred['boxes'], pred['scores'])):
            color = colors[(i + 5) % len(colors)] # Offset color to distinguish from GT if needed, or random
            
            # Mask
            masked_image = np.ma.masked_where(mask == 0, mask)
            ax[2].imshow(masked_image, alpha=0.5, cmap=plt.cm.colors.ListedColormap([color]), interpolation='none')
            
            # Box (Removed)
            # x, y, x2, y2 = box
            # w, h = x2 - x, y2 - y
            # rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            # ax[2].add_patch(rect)
            
            # Score (Keep score at centroid or top-left of mask?)
            # If box is removed, score position is tricky.
            # Let's put score at the center of mass of the mask
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                y_center = np.mean(y_indices)
                x_center = np.mean(x_indices)
                ax[2].text(x_center, y_center, f"{score:.2f}", color='white', fontsize=10, fontweight='bold', ha='center', va='center', bbox=dict(facecolor=color, alpha=0.5, edgecolor='none', pad=0.5))
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    args = get_args()
    predict(args)
