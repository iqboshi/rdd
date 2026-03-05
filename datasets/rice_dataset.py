import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class RiceDataset(Dataset):
    """
    Rice Disease Detection Dataset
    Supports both Semantic and Instance Segmentation.
    """
    def __init__(self, img_dir, mask_dir, list_path, transform=None, task_type='semantic'):
        """
        Args:
            img_dir (str): Path to image directory
            mask_dir (str): Path to mask directory
            list_path (str): Path to text file containing list of filenames/IDs
            transform (callable, optional): Albumentations transform pipeline
            task_type (str): 'semantic' or 'instance'
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.task_type = task_type
        
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")
            
        with open(list_path, 'r') as f:
            # Filter out empty lines
            self.ids = [line.strip() for line in f if line.strip()]
            
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # Construct paths
        img_path = os.path.join(self.img_dir, img_id)
        mask_path = os.path.join(self.mask_dir, img_id)
        
        # If file doesn't exist directly, try appending common extensions
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                if os.path.exists(img_path + ext):
                    img_path += ext
                    # Check mask extension logic
                    if not os.path.exists(mask_path) and not mask_path.endswith(ext): 
                         if os.path.exists(mask_path + '.png'):
                             mask_path += '.png'
                    break
        
        # Check if mask path exists as is or with different extension
        if not os.path.exists(mask_path):
            base, _ = os.path.splitext(mask_path)
            if os.path.exists(base + '.png'):
                mask_path = base + '.png'
            elif os.path.exists(base + '.jpg'):
                mask_path = base + '.jpg'

        # Read Image (BGR -> RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read Mask
        # Safe reading: UNCHANGED to support 16-bit or 8-bit masks correctly
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")
            
        # Logic Branching based on task_type
        if self.task_type == 'semantic':
            # Semantic: Binary Mask (0 background, 1 foreground)
            # Map all > 0 to 1
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)
            
            # Apply transforms
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Ensure types
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).long()
            else:
                mask = mask.long()
            
            if mask.dim() == 3:
                mask = mask.squeeze()
                
            return image, mask

        elif self.task_type == 'instance':
            # Instance: Keep IDs
            
            # Apply transforms
            # Note: Albumentations handles masks with arbitrary values if mask_interpolation is NEAREST (default/set in transforms)
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # Ensure Image is Tensor
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                
            # Process Mask for Instance Segmentation
            # Mask should be [H, W] with IDs
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # Extract unique IDs (excluding 0/background)
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[obj_ids > 0]
            
            # Split into binary masks: [N, H, W]
            # broadcasting: (H, W) == (N, 1, 1) -> (N, H, W)
            masks = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)
            
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            # All instances are class 1 (Rice Disease)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            
            # Compute boxes
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                
                # Check for degenerate boxes
                if xmax <= xmin or ymax <= ymin:
                    # Fallback for point-like masks or error
                    xmax = xmin + 1
                    ymax = ymin + 1
                    
                boxes.append([xmin, ymin, xmax, ymax])
                
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            
            target = {}
            target["boxes"] = boxes
            target["masks"] = masks
            target["labels"] = labels
            target["image_id"] = torch.tensor([index])
            
            return image, target
            
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
