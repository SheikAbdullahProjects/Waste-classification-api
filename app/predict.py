from PIL import Image
import torch
from torchvision import transforms

from .config import IMAGE_SIZE


def build_infer_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_image(model, class_names, image: Image.Image, device: torch.device):
    tf = build_infer_transform()
    tensor = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probs, dim=0)
    return class_names[top_idx.item()], float(top_prob.item())
