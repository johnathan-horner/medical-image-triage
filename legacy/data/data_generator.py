"""
Data generator for creating synthetic medical images for demonstration.
In production, this would use real NIH Chest X-Ray images.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SyntheticMedicalImageGenerator:
    """Generates synthetic medical images for demonstration purposes."""

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.conditions = ["Normal", "Pneumonia", "Pneumothorax", "Infiltration", "Mass"]

    def generate_base_chest_xray(self) -> Image.Image:
        """Generate a base chest X-ray like structure."""
        width, height = self.image_size

        # Create base grayscale image
        img = Image.new('L', self.image_size, color=40)
        draw = ImageDraw.Draw(img)

        # Draw chest cavity outline (simplified rib cage)
        chest_width = int(width * 0.7)
        chest_height = int(height * 0.8)
        chest_x = (width - chest_width) // 2
        chest_y = int(height * 0.1)

        # Draw oval for chest cavity
        draw.ellipse([chest_x, chest_y, chest_x + chest_width, chest_y + chest_height],
                    fill=80, outline=120, width=2)

        # Add lung regions
        lung_width = int(chest_width * 0.35)
        lung_height = int(chest_height * 0.6)

        # Left lung
        left_lung_x = chest_x + int(chest_width * 0.1)
        left_lung_y = chest_y + int(chest_height * 0.2)
        draw.ellipse([left_lung_x, left_lung_y,
                     left_lung_x + lung_width, left_lung_y + lung_height],
                    fill=100, outline=130)

        # Right lung
        right_lung_x = chest_x + chest_width - lung_width - int(chest_width * 0.1)
        right_lung_y = left_lung_y
        draw.ellipse([right_lung_x, right_lung_y,
                     right_lung_x + lung_width, right_lung_y + lung_height],
                    fill=100, outline=130)

        # Add some texture and noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        return img

    def add_condition_features(self, img: Image.Image, condition: str) -> Image.Image:
        """Add condition-specific features to the image."""
        draw = ImageDraw.Draw(img)
        width, height = img.size

        if condition == "Pneumonia":
            # Add cloudy patches for pneumonia
            for _ in range(random.randint(2, 5)):
                x = random.randint(width//4, 3*width//4)
                y = random.randint(height//3, 2*height//3)
                size = random.randint(20, 40)
                draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2],
                           fill=60, outline=None)

        elif condition == "Pneumothorax":
            # Add dark area indicating collapsed lung
            lung_x = width//4 + random.randint(-20, 20)
            lung_y = height//3 + random.randint(-20, 20)
            lung_size = random.randint(30, 50)
            draw.ellipse([lung_x, lung_y, lung_x + lung_size, lung_y + lung_size],
                        fill=20, outline=None)

        elif condition == "Infiltration":
            # Add streaky patterns
            for _ in range(random.randint(3, 8)):
                x1 = random.randint(width//4, 3*width//4)
                y1 = random.randint(height//3, 2*height//3)
                x2 = x1 + random.randint(-30, 30)
                y2 = y1 + random.randint(-30, 30)
                draw.line([x1, y1, x2, y2], fill=70, width=random.randint(2, 4))

        elif condition == "Mass":
            # Add distinct mass
            mass_x = random.randint(width//3, 2*width//3)
            mass_y = random.randint(height//3, 2*height//3)
            mass_size = random.randint(15, 25)
            draw.ellipse([mass_x-mass_size//2, mass_y-mass_size//2,
                         mass_x+mass_size//2, mass_y+mass_size//2],
                        fill=30, outline=50, width=2)

        # Normal condition gets no additional features

        return img

    def generate_image(self, condition: str, add_noise: bool = True) -> Image.Image:
        """Generate a synthetic medical image with specified condition."""
        # Generate base chest X-ray
        img = self.generate_base_chest_xray()

        # Add condition-specific features
        img = self.add_condition_features(img, condition)

        # Add realistic noise
        if add_noise:
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

        return img

    def generate_dataset_images(self, data_dir: Path, metadata_df) -> None:
        """Generate synthetic images for the dataset."""
        logger.info("Generating synthetic medical images...")

        generated_count = 0
        for _, row in metadata_df.iterrows():
            condition = row['Primary_Condition']
            split = row['Split']
            image_name = row['Image Index']

            # Generate image
            img = self.generate_image(condition)

            # Save image
            img_path = data_dir / "raw" / split / condition / image_name
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)

            generated_count += 1
            if generated_count % 50 == 0:
                logger.info(f"Generated {generated_count} images...")

        logger.info(f"Generated {generated_count} synthetic medical images")


def main():
    """Generate synthetic dataset images."""
    import pandas as pd

    # Load metadata
    metadata_path = Path("data/raw/processed_metadata.csv")
    if not metadata_path.exists():
        logger.error("Metadata file not found. Run download_dataset.py first.")
        return

    metadata_df = pd.read_csv(metadata_path)

    # Generate images
    generator = SyntheticMedicalImageGenerator()
    generator.generate_dataset_images(Path("data"), metadata_df)


if __name__ == "__main__":
    main()