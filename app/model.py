from typing import List

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_model(model: nn.Module, class_names: List[str], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }, path)


def load_model(path, device: torch.device) -> tuple[nn.Module, List[str]]:
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint["class_names"]
    model = create_model(len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
