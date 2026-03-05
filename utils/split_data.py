import os
import random
import yaml
import glob
from sklearn.model_selection import train_test_split

def split_dataset(config_path="configs/baseline.yaml", val_ratio=0.2):
    """
    Split dataset into train and val sets based on config.
    """
    # Load config
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get random seed
    random_seed = config.get('random_seed', 42)
    print(f"Using random seed: {random_seed}")
    
    # Set random seed
    random.seed(random_seed)

    # Get data directories
    data_root = config['data']['root_dir']
    images_dir = os.path.join(data_root, 'images')
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    # List all image files
    # We look for common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(images_dir, ext)))
        # Also try upper case
        all_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    if not all_files:
        print("No images found.")
        return

    # Get basenames (filenames without path and extension usually, but here we need filename to be written to list)
    # The RiceDataset expects IDs which are usually filenames.
    # Let's extract just the filename (e.g. image_01.jpg)
    
    # However, RiceDataset implementation checks extensions if not provided. 
    # But standard practice is to put the filename or the ID.
    # Previous cleanup script used stems.
    # Let's verify what the previous list.txt had. It had full filenames "IMG_...png".
    # So we should store full filenames.
    
    file_names = [os.path.basename(f) for f in all_files]
    # Sort to ensure deterministic split before shuffling
    file_names.sort()
    
    print(f"Found {len(file_names)} images.")

    # Get val_split from config
    val_ratio = config['data'].get('val_split', val_ratio)
    print(f"Using validation split ratio: {val_ratio}")

    # Split
    train_files, val_files = train_test_split(
        file_names, 
        test_size=val_ratio, 
        random_state=random_seed,
        shuffle=True
    )

    print(f"Train set: {len(train_files)} images")
    print(f"Val set: {len(val_files)} images")

    # Write to files
    train_list_path = os.path.join(data_root, config['data']['train_list'])
    val_list_path = os.path.join(data_root, config['data']['val_list'])

    with open(train_list_path, 'w') as f:
        for item in train_files:
            f.write(f"{item}\n")
            
    with open(val_list_path, 'w') as f:
        for item in val_files:
            f.write(f"{item}\n")

    print(f"Saved train list to {train_list_path}")
    print(f"Saved val list to {val_list_path}")

if __name__ == "__main__":
    # Assuming run from project root
    split_dataset(config_path="/root/autodl-fs/rice_disease_detect/configs/baseline.yaml")
