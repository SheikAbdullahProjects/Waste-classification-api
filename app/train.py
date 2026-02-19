from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import (
    DATASET_DIR,
    MODEL_PATH,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_TRAIN_STEPS,
    MAX_VAL_STEPS,
    MODEL_NAME,
    FREEZE_BACKBONE_EPOCHS,
    LABEL_SMOOTHING,
)
from .dataset import build_dataloaders
from .model import create_model, save_model


def train():
    print(torch.cuda.is_available())
    
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = build_dataloaders(DATASET_DIR)
    model = create_model(len(class_names), MODEL_NAME).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        if epoch <= FREEZE_BACKBONE_EPOCHS:
            for name, param in model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
        train_loss = 0.0

        train_steps = 0
        train_count = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Train"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_count += images.size(0)
            train_steps += 1
            if MAX_TRAIN_STEPS and train_steps >= MAX_TRAIN_STEPS:
                break

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_steps = 0
        val_count = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Val"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_count += labels.size(0)
                val_steps += 1
                if MAX_VAL_STEPS and val_steps >= MAX_VAL_STEPS:
                    break

        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        val_acc = correct / total if total else 0.0

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, class_names, MODEL_PATH)

        scheduler.step()

        print(
            f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}"
        )
        print(torch.cuda.is_available())

    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
