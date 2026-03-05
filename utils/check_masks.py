import os
import cv2
import numpy as np
from tqdm import tqdm

def check_mask_values():
    masks_dir = "/root/autodl-fs/rice_disease_detect/data/masks"
    if not os.path.exists(masks_dir):
        print(f"Directory not found: {masks_dir}")
        return

    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    print(f"Checking {len(mask_files)} mask files...")

    all_unique_values = set()
    problematic_files = []

    for fname in tqdm(mask_files):
        path = os.path.join(masks_dir, fname)
        # Read as grayscale
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading {fname}")
            continue
            
        unique = np.unique(mask)
        for val in unique:
            all_unique_values.add(val)
            
        # Check if values are only 0 and 1 (or 255 depending on convention, user said 'should be two values')
        # If user expects binary mask, usually 0 and 1, or 0 and 255.
        if len(unique) > 2:
             problematic_files.append((fname, unique))

    print("\n--- Analysis Result ---")
    print(f"All unique values found across dataset: {sorted(list(all_unique_values))}")
    
    if problematic_files:
        print(f"\nFound {len(problematic_files)} files with more than 2 values:")
        for fname, unique in problematic_files[:10]: # Show first 10
            print(f"  {fname}: {unique}")
        if len(problematic_files) > 10:
            print(f"  ... and {len(problematic_files) - 10} more.")
    else:
        print("\nAll files have <= 2 unique values.")

if __name__ == "__main__":
    check_mask_values()
