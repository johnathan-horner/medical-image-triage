"""PneumoniaMNIST loading (MedMNIST v2, 64x64 images, CC BY 4.0)."""

from __future__ import annotations

import numpy as np

from triage.config import DATA_CACHE, IMAGE_SIZE


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 N x 64 x 64, labels int N) for 'train', 'val' or 'test'.

    Labels: 0 = normal, 1 = pneumonia. Downloads once into .data/.
    """
    from medmnist import PneumoniaMNIST

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    ds = PneumoniaMNIST(split=split, download=True, root=str(DATA_CACHE), size=IMAGE_SIZE)
    return ds.imgs.astype(np.uint8), ds.labels.reshape(-1).astype(np.int64)
