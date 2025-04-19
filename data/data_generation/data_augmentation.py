import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import random
import albumentations as A


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Data augmentation for image datasets")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save augmented images")
    parser.add_argument("--target_size", type=int, required=True, help="Target dataset size after augmentation")
    parser.add_argument("--img_ext", default=".jpg", help="Image file extension (default: .jpg)")
    return parser.parse_args()


def get_augmentations():
    """Define a list of different augmentation techniques"""
    augmentations = [
        # Flip operations
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        
        # Rotate operations
        A.Compose([A.Rotate(limit=90, p=1.0)]),
        A.Compose([A.Rotate(limit=45, p=1.0)]),
        
        # Color transformations
        A.Compose([A.RandomBrightnessContrast(p=1.0)]),
        A.Compose([A.HueSaturationValue(p=1.0)]),
        
        # Blur and noise
        A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)]),
        A.Compose([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)]),
        
        # Combined transformations
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.8),
            A.Rotate(limit=30, p=0.7),
        ]),
        A.Compose([
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(p=0.8),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
        ]),
    ]
    return augmentations


def augment_dataset(input_dir, output_dir, target_size, img_ext=".jpg"):
    """Augment dataset to reach target size"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all input image paths
    image_files = [f for f in os.listdir(input_dir) if f.endswith(img_ext)]
    current_size = len(image_files)
    
    if current_size >= target_size:
        print(f"Current dataset size ({current_size}) is already >= target size ({target_size}). No augmentation needed.")
        return
    
    print(f"Current dataset size: {current_size}")
    print(f"Target dataset size: {target_size}")
    print(f"Need to generate {target_size - current_size} new images")
    
    augmentations = get_augmentations()
    
    # Copy original images to output directory first
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, img)
    
    # Counter to track how many images we've generated
    generated_count = current_size
    
    # Generate augmented images until we reach the target size
    with tqdm(total=target_size - current_size) as pbar:
        while generated_count < target_size:
            # Randomly select an original image
            img_file = random.choice(image_files)
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Randomly select an augmentation technique
            augmentation = random.choice(augmentations)
            
            # Apply augmentation
            augmented = augmentation(image=img)
            augmented_img = augmented['image']
            
            # Create a new filename and save the augmented image
            basename, ext = os.path.splitext(img_file)
            new_filename = f"{basename}_aug_{generated_count - current_size + 1}{ext}"
            output_path = os.path.join(output_dir, new_filename)
            
            cv2.imwrite(output_path, augmented_img)
            
            generated_count += 1
            pbar.update(1)
            
            # Break if we've reached the target size
            if generated_count >= target_size:
                break

    print(f"Augmentation complete! Final dataset size: {generated_count}")


if __name__ == "__main__":
    args = parse_args()
    augment_dataset(args.input_dir, args.output_dir, args.target_size, args.img_ext)