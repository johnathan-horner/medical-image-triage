#!/usr/bin/env python3
"""
Create sample X-ray images for demo mode.
These are simple grayscale images that look like medical X-rays for demonstration purposes.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_chest_xray_pattern(width=512, height=512, condition="Normal"):
    """Create a simple chest X-ray pattern"""
    # Create base grayscale image
    img = np.random.normal(50, 20, (height, width)).astype(np.uint8)

    # Add chest cavity outline
    center_x, center_y = width // 2, height // 2

    # Ribcage pattern
    for i in range(6):
        y = center_y - 120 + i * 40
        x_start = center_x - 150 + i * 10
        x_end = center_x + 150 - i * 10
        if 0 <= y < height:
            img[y:min(y+3, height), max(0, x_start):min(x_end, width)] = 200

    # Lung fields
    left_lung = np.random.normal(30, 15, (150, 120)).astype(np.uint8)
    right_lung = np.random.normal(30, 15, (150, 120)).astype(np.uint8)

    img[center_y-75:center_y+75, center_x-180:center_x-60] = np.clip(
        img[center_y-75:center_y+75, center_x-180:center_x-60] + left_lung, 0, 255
    )
    img[center_y-75:center_y+75, center_x+60:center_x+180] = np.clip(
        img[center_y-75:center_y+75, center_x+60:center_x+180] + right_lung, 0, 255
    )

    # Add condition-specific patterns
    if condition == "Pneumonia":
        # Add cloudy infiltrate
        infiltrate = np.random.normal(80, 30, (80, 100)).astype(np.uint8)
        img[center_y-40:center_y+40, center_x-130:center_x-30] = np.clip(
            img[center_y-40:center_y+40, center_x-130:center_x-30] + infiltrate, 0, 255
        )
    elif condition == "Cardiomegaly":
        # Enlarged heart shadow
        heart = np.random.normal(120, 20, (100, 80)).astype(np.uint8)
        img[center_y-10:center_y+90, center_x-40:center_x+40] = np.clip(
            img[center_y-10:center_y+90, center_x-40:center_x+40] + heart, 0, 255
        )

    return img

def create_sample_images():
    """Create three sample X-ray images"""

    samples_dir = "/Users/johnathanhorner/medical-image-triage/samples"

    # Sample 1: Normal chest X-ray
    normal_img = create_chest_xray_pattern(condition="Normal")
    normal_pil = Image.fromarray(normal_img, mode='L')
    normal_pil.save(os.path.join(samples_dir, "normal_chest_xray.png"))

    # Sample 2: Pneumonia
    pneumonia_img = create_chest_xray_pattern(condition="Pneumonia")
    pneumonia_pil = Image.fromarray(pneumonia_img, mode='L')
    pneumonia_pil.save(os.path.join(samples_dir, "pneumonia_chest_xray.png"))

    # Sample 3: Cardiomegaly
    cardio_img = create_chest_xray_pattern(condition="Cardiomegaly")
    cardio_pil = Image.fromarray(cardio_img, mode='L')
    cardio_pil.save(os.path.join(samples_dir, "cardiomegaly_chest_xray.png"))

    print("Sample medical images created successfully!")
    print("Created files:")
    print("- normal_chest_xray.png")
    print("- pneumonia_chest_xray.png")
    print("- cardiomegaly_chest_xray.png")

if __name__ == "__main__":
    create_sample_images()