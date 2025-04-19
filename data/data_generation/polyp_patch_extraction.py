import cv2
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract polyp and background patches from images")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches to extract (NxN)")
    parser.add_argument("--source_dir", default=None, help="Source dataset directory (optional)")
    parser.add_argument("--target_dir", default=None, help="Target directory for patches (optional)")
    return parser.parse_args()

def extract_patches(source_dataset_dir, target_dataset_dir, patch_size=256):
    # Setup directories
    original_dir = os.path.join(source_dataset_dir, 'PNG', 'Original')
    mask_dir = os.path.join(source_dataset_dir, 'PNG', 'Ground Truth')

    polyp_crop_dir = os.path.join(target_dataset_dir, 'polyp')
    background_crop_dir = os.path.join(target_dataset_dir, 'no_polyp')

    os.makedirs(polyp_crop_dir, exist_ok=True)
    os.makedirs(background_crop_dir, exist_ok=True)

    filenames = os.listdir(original_dir)
    
    print(f"Extracting patches of size {patch_size}x{patch_size} pixels")
    
    for filename in filenames:
        original_path = os.path.join(original_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        # Load images
        image = cv2.imread(original_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        
        # Find contours in the mask (polyp regions)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found, extract bounding boxes for polyp segments.
        if contours:
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Ensure the polyp patch is patch_size x patch_size
                # Two approaches:
                # 1. If the polyp is larger than patch_size, crop it to patch_size
                # 2. If the polyp is smaller, center it in a patch_size x patch_size patch
                
                if w >= patch_size and h >= patch_size:
                    # Crop from center of polyp if it's larger than patch_size
                    center_x = x + w // 2
                    center_y = y + h // 2
                    x1 = max(0, center_x - patch_size // 2)
                    y1 = max(0, center_y - patch_size // 2)
                    x2 = min(image.shape[1], x1 + patch_size)
                    y2 = min(image.shape[0], y1 + patch_size)
                    
                    # Adjust if at boundaries
                    if x2 - x1 < patch_size:
                        x1 = max(0, x2 - patch_size)
                    if y2 - y1 < patch_size:
                        y1 = max(0, y2 - patch_size)
                    
                    crop = image[y1:y1+patch_size, x1:x1+patch_size]
                else:
                    # Center the polyp in a patch_size x patch_size area
                    center_x = x + w // 2
                    center_y = y + h // 2
                    x1 = max(0, center_x - patch_size // 2)
                    y1 = max(0, center_y - patch_size // 2)
                    
                    # Adjust if too close to image boundaries
                    if x1 + patch_size > image.shape[1]:
                        x1 = max(0, image.shape[1] - patch_size)
                    if y1 + patch_size > image.shape[0]:
                        y1 = max(0, image.shape[0] - patch_size)
                    
                    # Create the patch
                    if x1 + patch_size <= image.shape[1] and y1 + patch_size <= image.shape[0]:
                        crop = image[y1:y1+patch_size, x1:x1+patch_size]
                    else:
                        continue  # Skip if we can't extract a full-sized patch
                
                # Make sure we have the right size
                if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
                    continue
                
                crop_filename = f"{os.path.splitext(filename)[0]}_polyp_{i}.jpg"
                print(f"Saving polyp crop: {crop_filename} to {polyp_crop_dir}")
                cv2.imwrite(os.path.join(polyp_crop_dir, crop_filename), crop)
            
            # Extract background patches
            h_img, w_img = image.shape[:2]
            # Ensure the patch fits in the image and does not contain a polyp.
            if h_img > patch_size and w_img > patch_size:
                for _ in range(10):  # Try up to 10 times to find a valid patch
                    top = np.random.randint(0, h_img - patch_size)
                    left = np.random.randint(0, w_img - patch_size)
                    patch = image[top:top+patch_size, left:left+patch_size]
                    mask_patch = mask[top:top+patch_size, left:left+patch_size]

                    # Check if the mask patch contains any polyp (non-zero values)
                    if np.count_nonzero(mask_patch) == 0:
                        patch_filename = f"{os.path.splitext(filename)[0]}_background.jpg"
                        print(f"Saving background patch: {patch_filename} to {background_crop_dir}")
                        cv2.imwrite(os.path.join(background_crop_dir, patch_filename), patch)
                        break

if __name__ == "__main__":
    args = parse_args()
    
    # current file directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use provided directories or defaults
    source_dataset_dir = args.source_dir if args.source_dir else os.path.join(cur_dir, '..', 'datasets', 'cvc_clinic_db_original')
    target_dataset_dir = args.target_dir if args.target_dir else os.path.join(cur_dir, '..', 'datasets', 'cvc_clinic_db_patches')
    
    extract_patches(source_dataset_dir, target_dataset_dir, args.patch_size)
