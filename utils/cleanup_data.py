import os
import glob

# Paths
list_file = "/root/autodl-fs/rice_disease_detect/data/list.txt"
images_dir = "/root/autodl-fs/rice_disease_detect/data/images"
masks_dir = "/root/autodl-fs/rice_disease_detect/data/masks"

def cleanup():
    # 1. Read list.txt and get basenames (stems)
    if not os.path.exists(list_file):
        print(f"Error: {list_file} not found.")
        return

    with open(list_file, 'r') as f:
        valid_files = [line.strip() for line in f if line.strip()]

    # Extract stems (remove extension)
    valid_stems = set()
    for fname in valid_files:
        stem = os.path.splitext(fname)[0]
        valid_stems.add(stem)

    print(f"Found {len(valid_stems)} valid file stems in list.txt")

    # 2. Cleanup images
    if os.path.exists(images_dir):
        image_files = os.listdir(images_dir)
        deleted_images = 0
        for fname in image_files:
            stem = os.path.splitext(fname)[0]
            if stem not in valid_stems:
                file_path = os.path.join(images_dir, fname)
                try:
                    os.remove(file_path)
                    deleted_images += 1
                    # print(f"Deleted image: {fname}")
                except Exception as e:
                    print(f"Error deleting {fname}: {e}")
        print(f"Deleted {deleted_images} files from images directory.")
    else:
        print(f"Images directory {images_dir} does not exist.")

    # 3. Cleanup masks
    if os.path.exists(masks_dir):
        mask_files = os.listdir(masks_dir)
        deleted_masks = 0
        for fname in mask_files:
            stem = os.path.splitext(fname)[0]
            if stem not in valid_stems:
                file_path = os.path.join(masks_dir, fname)
                try:
                    os.remove(file_path)
                    deleted_masks += 1
                    # print(f"Deleted mask: {fname}")
                except Exception as e:
                    print(f"Error deleting {fname}: {e}")
        print(f"Deleted {deleted_masks} files from masks directory.")
    else:
        print(f"Masks directory {masks_dir} does not exist.")

if __name__ == "__main__":
    cleanup()
