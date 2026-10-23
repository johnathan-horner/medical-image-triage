"""ResNet18 classifier for PneumoniaMNIST and single-image inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from triage.config import CLASS_NAMES, IMAGE_SIZE, MODEL_VERSION, MODEL_WEIGHTS

# ImageNet statistics, because the backbone was pretrained on ImageNet.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    # One logit: probability of pneumonia after a sigmoid.
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 array (N, H, W) grayscale -> normalized float tensor (N, 3, H, W)."""
    x = torch.from_numpy(np.array(images, copy=True)).float().div(255.0).unsqueeze(1)
    if x.shape[-1] != IMAGE_SIZE or x.shape[-2] != IMAGE_SIZE:
        x = nn.functional.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - _MEAN) / _STD


def image_to_array(image: Image.Image) -> np.ndarray:
    """Any PIL image -> uint8 grayscale array at the model's input size."""
    gray = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(gray, dtype=np.uint8)


@dataclass(frozen=True)
class Prediction:
    label: str
    pneumonia_probability: float
    confidence: float
    model_version: str


class Classifier:
    """Loads trained weights once and scores images."""

    def __init__(self, weights_path: Path = MODEL_WEIGHTS):
        self.model = build_model(pretrained=False)
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict({k: v.float() for k, v in state.items()})
        self.model.eval()
        self.version = MODEL_VERSION

    @torch.no_grad()
    def probabilities(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        out = []
        for start in range(0, len(images), batch_size):
            logits = self.model(to_tensor(images[start:start + batch_size])).squeeze(1)
            out.append(torch.sigmoid(logits).numpy())
        return np.concatenate(out) if out else np.empty(0)

    def predict(self, image: Image.Image) -> Prediction:
        p = float(self.probabilities(image_to_array(image)[None])[0])
        return make_prediction(p, self.version)


def make_prediction(pneumonia_probability: float, version: str = MODEL_VERSION) -> Prediction:
    p = pneumonia_probability
    label = CLASS_NAMES[1] if p >= 0.5 else CLASS_NAMES[0]
    return Prediction(label=label, pneumonia_probability=p, confidence=max(p, 1.0 - p), model_version=version)
