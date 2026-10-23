"""
Dataset download and preprocessing script for NIH Chest X-Ray dataset.
Downloads a subset of the dataset for demonstration purposes.
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
import pandas as pd
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NIHChestXrayDataset:
    """Handler for NIH Chest X-Ray dataset download and preprocessing."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # For demo purposes, we'll create synthetic labels for common conditions
        self.conditions = [
            "Normal",
            "Pneumonia",
            "Pneumothorax",
            "Infiltration",
            "Mass"
        ]

    def create_demo_dataset(self) -> Tuple[pd.DataFrame, Path]:
        """
        Creates a demo dataset structure for development.
        In production, this would download the actual NIH dataset.
        """
        logger.info("Creating demo dataset structure...")

        # Create directories
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"

        for split_dir in [train_dir, val_dir, test_dir]:
            split_dir.mkdir(exist_ok=True)
            for condition in self.conditions:
                (split_dir / condition).mkdir(exist_ok=True)

        # Create metadata CSV
        metadata_rows = []
        for split, count in [("train", 100), ("val", 20), ("test", 30)]:
            for condition in self.conditions:
                for i in range(count // len(self.conditions)):
                    image_name = f"{condition.lower()}_{split}_{i:03d}.jpg"
                    metadata_rows.append({
                        "Image Index": image_name,
                        "Finding Labels": condition,
                        "Split": split,
                        "Patient ID": f"P{split}_{condition}_{i:03d}"
                    })

        metadata_df = pd.DataFrame(metadata_rows)
        metadata_path = self.data_dir / "Data_Entry_2017.csv"
        metadata_df.to_csv(metadata_path, index=False)

        logger.info(f"Demo dataset created with {len(metadata_df)} samples")
        logger.info(f"Metadata saved to: {metadata_path}")

        return metadata_df, self.data_dir

    def preprocess_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the metadata for training."""

        # Clean finding labels - take primary condition for multi-label cases
        metadata_df['Primary_Condition'] = metadata_df['Finding Labels'].apply(
            lambda x: x.split('|')[0] if isinstance(x, str) else 'Normal'
        )

        # Filter to our target conditions
        metadata_df = metadata_df[
            metadata_df['Primary_Condition'].isin(self.conditions)
        ].copy()

        # Create label encoding
        label_mapping = {condition: idx for idx, condition in enumerate(self.conditions)}
        metadata_df['Label'] = metadata_df['Primary_Condition'].map(label_mapping)

        logger.info(f"Preprocessed metadata: {len(metadata_df)} samples")
        logger.info(f"Class distribution:\n{metadata_df['Primary_Condition'].value_counts()}")

        return metadata_df


def main():
    """Download and preprocess the dataset."""
    downloader = NIHChestXrayDataset()

    # Create demo dataset
    metadata_df, data_path = downloader.create_demo_dataset()

    # Preprocess metadata
    processed_df = downloader.preprocess_metadata(metadata_df)

    # Save processed metadata
    processed_path = data_path / "processed_metadata.csv"
    processed_df.to_csv(processed_path, index=False)

    logger.info(f"Dataset preparation complete!")
    logger.info(f"Raw data: {data_path}")
    logger.info(f"Processed metadata: {processed_path}")

    return data_path, processed_path


if __name__ == "__main__":
    main()