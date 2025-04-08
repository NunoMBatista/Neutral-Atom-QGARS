# Colon Polyp Detection - Synthetic Data Generation and ML Model
# Google Colab Notebook

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Create directories for saving images
os.makedirs('dataset/polyp', exist_ok=True)
os.makedirs('dataset/no_polyp', exist_ok=True)


# Part 1: Enhanced Synthetic Colon Image Generation
# ---------------------------------------

def generate_colon_background(img_size=256, noise_factor=0.1):
    """Generate a synthetic colon background with a pinkish color and darker center."""
    # Base pink color for colon tissue
    base_color = np.array([210, 140, 140])  # Pinkish color

    # Create base image
    img = np.ones((img_size, img_size, 3))

    # Add radial gradient for tube-like appearance with darker center
    y, x = np.ogrid[:img_size, :img_size]
    center = img_size // 2
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)

    # Normalize to [0, 1]
    dist_from_center = dist_from_center / (np.sqrt(2) * center)

    # Make center darker - add stronger vignetting effect
    for c in range(3):
        # More dramatic darkening in the center
        img[:, :, c] = base_color[c] / 255.0 * (0.6 + 0.4 * dist_from_center)

    # Add noise for texture
    noise = np.random.randn(img_size, img_size, 3) * noise_factor
    img += noise
    img = np.clip(img, 0, 1)

    return img

def add_colon_folds(img, num_folds=6, fold_width_range=(8, 15)):
    """Add concentric rings/folds to the colon image with more contrast."""
    img_size = img.shape[0]
    center = img_size // 2

    # Convert to PIL for drawing
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    # Draw concentric circles for colon folds with alternating brightness
    for i in range(1, num_folds + 1):
        radius = (img_size // (num_folds + 2)) * i

        # Add slight randomness to fold placement
        radius += random.randint(-5, 5)

        # Alternating fold brightness for more contrast
        if i % 2 == 0:
            # Lighter fold
            fold_color = (200, 130, 130)  # Lighter pink
            fold_width = random.randint(fold_width_range[0], fold_width_range[1])
        else:
            # Darker fold
            fold_color = (170, 100, 100)  # Darker pink
            fold_width = random.randint(fold_width_range[0] + 2, fold_width_range[1] + 2)  # Slightly wider

        # Draw with varying opacity based on distance from center for perspective
        opacity_factor = 0.7 + 0.3 * (i / num_folds)  # More pronounced outer folds
        adjusted_color = tuple(int(c * opacity_factor) for c in fold_color)

        # Draw squished ellipses for perspective (more squished near center)
        squish_factor = 0.6 + 0.2 * (i / num_folds)  # More circular as we move outward
        ellipse_bounds = [
            center - radius,
            center - radius * squish_factor,
            center + radius,
            center + radius * squish_factor
        ]

        # Draw the fold
        draw.ellipse(ellipse_bounds, outline=adjusted_color, width=fold_width)

        # Add highlight effect on the upper part of some folds for depth
        if i % 2 == 0 and i > 1:
            highlight_bounds = [
                center - radius + fold_width//2,
                center - radius * squish_factor + fold_width//2,
                center + radius - fold_width//2,
                center - radius * squish_factor//2
            ]
            # Partial highlight arc
            draw.arc(highlight_bounds, 180, 360, fill=(230, 170, 170), width=fold_width//2)

    # Add a dark central area to simulate depth
    central_radius = img_size // (num_folds + 2)
    central_ellipse = [
        center - central_radius,
        center - central_radius * 0.6,
        center + central_radius,
        center + central_radius * 0.6
    ]
    # Draw dark central area with gradient
    for r in range(central_radius, 0, -2):
        darkness = 0.6 - (r / central_radius) * 0.4  # Darker toward center
        dark_color = tuple(int(c * (1-darkness)) for c in (170, 100, 100))
        inner_ellipse = [
            center - r,
            center - r * 0.6,
            center + r,
            center + r * 0.6
        ]
        draw.ellipse(inner_ellipse, fill=dark_color, outline=None)

    # Convert back to numpy
    img = np.array(pil_img) / 255.0
    return img

def generate_polyp(img_size=256, min_radius=10, max_radius=30):
    """Generate a polyp mask and the polyp appearance."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Random position for the polyp
    center_x = random.randint(img_size // 4, 3 * img_size // 4)
    center_y = random.randint(img_size // 4, 3 * img_size // 4)

    # Random radius for the polyp
    radius = random.randint(min_radius, max_radius)

    # Create a PIL image for drawing
    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)

    # Draw an ellipse for the polyp (slightly irregular)
    squish_factor = random.uniform(0.7, 1.0)
    ellipse_bounds = [
        center_x - radius,
        center_y - radius * squish_factor,
        center_x + radius,
        center_y + radius * squish_factor
    ]
    draw.ellipse(ellipse_bounds, fill=1)

    # Add irregularity to some polyps (about 40% of polyps)
    if random.random() < 0.4:
        # Add a random "bump" to the polyp
        bump_angle = random.uniform(0, 2 * np.pi)
        bump_distance = radius * 0.7
        bump_size = radius * random.uniform(0.3, 0.5)

        bump_x = center_x + bump_distance * np.cos(bump_angle)
        bump_y = center_y + bump_distance * np.sin(bump_angle)

        bump_bounds = [
            bump_x - bump_size,
            bump_y - bump_size,
            bump_x + bump_size,
            bump_y + bump_size
        ]
        draw.ellipse(bump_bounds, fill=1)

    # Convert back to numpy
    mask = np.array(mask_img)

    # For polyp appearance (reddish/darker color with texture)
    polyp_color = np.array([180, 100, 100]) / 255.0  # Darker red

    # Make some polyps more pinkish (variety)
    if random.random() < 0.3:
        polyp_color = np.array([220, 120, 120]) / 255.0  # More pinkish

    # Add slight texture variation
    polyp_texture = generate_polyp_texture(mask, polyp_color)

    return mask, polyp_texture

def generate_polyp_texture(mask, base_color):
    """Generate texture for the polyp with more realistic appearance."""
    img_size = mask.shape[0]
    texture = np.zeros((img_size, img_size, 3))

    # Find the center and radius of the polyp
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return texture

    center_y = int(np.mean(y_indices))
    center_x = int(np.mean(x_indices))

    # Estimate radius as distance to furthest point
    max_dist = 0
    for y, x in zip(y_indices, x_indices):
        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_dist = max(max_dist, dist)

    # Create a distance map from center for shading
    y, x = np.ogrid[:img_size, :img_size]
    dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)

    # Normalize distances for shading (0 at center, 1 at edge)
    normalized_dist = np.zeros_like(dist_from_center)
    mask_indices = mask > 0
    if np.any(mask_indices):
        normalized_dist[mask_indices] = dist_from_center[mask_indices] / max_dist

    # Create highlights and shadows
    highlight_dir = random.uniform(0, 2 * np.pi)  # Random direction for highlight
    highlight_x = np.cos(highlight_dir)
    highlight_y = np.sin(highlight_dir)

    # Calculate highlight intensity based on direction
    y_rel = (y - center_y) / (max_dist + 1e-6)
    x_rel = (x - center_x) / (max_dist + 1e-6)
    directional_component = x_rel * highlight_x + y_rel * highlight_y

    # Create texture with highlight and shadow
    for c in range(3):
        # Base color
        color_map = np.ones((img_size, img_size)) * base_color[c]

        # Add radial shading (darker toward edges)
        edge_darkening = 0.3 * normalized_dist  # Darken up to 30% at edges

        # Add directional highlight (brighten in highlight direction, darken in opposite)
        highlight = 0.2 * directional_component  # +/- 20% variation based on direction

        # Apply both effects where the mask is active
        color_map[mask > 0] = color_map[mask > 0] * (1 - edge_darkening[mask > 0] + highlight[mask > 0])

        # Add some random texture variation
        noise = np.random.randn(img_size, img_size) * 0.05
        color_map[mask > 0] += noise[mask > 0]

        # Ensure values are in valid range
        color_map = np.clip(color_map, 0, 1)

        # Apply to texture
        texture[:, :, c] = mask * color_map

    return texture

def add_polyp_to_image(img, mask, polyp_texture):
    """Add a polyp to the colon image with better blending."""
    # Blend the polyp with the background image
    result = img.copy()

    # Create a blurred edge mask for better blending
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return result

    # Find polyp center and max radius
    center_y = int(np.mean(y_indices))
    center_x = int(np.mean(x_indices))

    # Calculate distances for all polyp pixels
    distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
    max_dist = np.max(distances)

    # Create a soft edge mask with blur
    soft_mask = mask.copy().astype(float)

    # For each positive pixel in the mask
    for y, x, dist in zip(y_indices, x_indices, distances):
        # If we're near the edge (within 15% of max radius from edge)
        edge_ratio = dist / max_dist
        if edge_ratio > 0.85:
            # Calculate opacity based on distance to edge (1 at core, 0 at edge)
            fade_factor = 1.0 - ((edge_ratio - 0.85) / 0.15)
            soft_mask[y, x] = fade_factor

    # Apply the soft mask for blending
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - soft_mask) + polyp_texture[:, :, c]

    # Ensure values are valid
    result = np.clip(result, 0, 1)

    return result

def generate_dataset(num_images=500, img_size=28):
    """Generate a synthetic dataset of colon images with and without polyps."""
    polyp_images = []
    no_polyp_images = []
    
    polyp_count = 0
    no_polyp_count = 0

    for i in tqdm(range(num_images), desc="Generating synthetic data"):
        # Generate background
        img = generate_colon_background(img_size)
        img = add_colon_folds(img)

        # With 70% probability, add a polyp
        has_polyp = random.random() < 0.7

        if has_polyp:
            # Generate between 1 and 3 polyps
            num_polyps = random.randint(1, 3)
            combined_mask = np.zeros((img_size, img_size))

            for _ in range(num_polyps):
                polyp_mask, polyp_texture = generate_polyp(img_size)
                img = add_polyp_to_image(img, polyp_mask, polyp_texture)
                combined_mask = np.maximum(combined_mask, polyp_mask)
            
            # Save the images with polyps
            img_array = (img * 255).astype(np.uint8)
            
            Image.fromarray(img_array).save(f'dataset/polyp/{polyp_count:04d}.png')

            
            # Keep in memory for visualization
            polyp_images.append(img)
            polyp_count += 1
        else:
            # Save the images without polyps
            combined_mask = np.zeros((img_size, img_size))
            img_array = (img * 255).astype(np.uint8)

            
            Image.fromarray(img_array).save(f'dataset/no_polyp/{no_polyp_count:04d}.png')
            
            # Keep in memory for visualization
            no_polyp_images.append(img)
            no_polyp_count += 1

    print(f"Generated {polyp_count} images with polyps and {no_polyp_count} images without polyps")
    
    # Return all images and masks for visualization
    all_images = polyp_images + no_polyp_images
    
    return all_images

# Generate the dataset
images = generate_dataset(num_images=70000, img_size=128) 

# Visualize some examples
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
for i in range(3):
    idx = random.randint(0, len(images) - 1)
    axes[i, 0].imshow(images[idx])
    axes[i, 0].set_title(f"Image {idx}")
    axes[i, 0].axis('off')


plt.tight_layout()
plt.show()


