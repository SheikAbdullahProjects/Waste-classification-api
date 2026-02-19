from typing import List

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int, model_name: str) -> nn.Module:
    if not hasattr(models, model_name):
        raise ValueError(f"Unsupported model: {model_name}")

    model_builder = getattr(models, model_name)
    try:
        weights = models.get_model_weights(model_builder).DEFAULT
    except Exception:
        weights = None
    model = model_builder(weights=weights)

    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Model does not expose a standard fc head")

    return model


def save_model(model: nn.Module, class_names: List[str], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }, path)


def load_model(path, device: torch.device, model_name: str) -> tuple[nn.Module, List[str]]:
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint["class_names"]
    model = create_model(len(class_names), model_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
