from collections import Counter

import torch
from torch.utils.data import random_split, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_SAMPLES,
    VAL_SPLIT,
    RANDOM_SEED,
    USE_WEIGHTED_SAMPLER,
)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.15)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


def build_dataloaders(dataset_dir, val_split=VAL_SPLIT):
    train_tf, val_tf = build_transforms()

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_tf)

    if MAX_SAMPLES and len(full_dataset) > MAX_SAMPLES:
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        indices = torch.randperm(len(full_dataset), generator=generator)[:MAX_SAMPLES].tolist()
        full_dataset = Subset(full_dataset, indices)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_ds.dataset.transform = val_tf

    train_sampler = None
    if USE_WEIGHTED_SAMPLER:
        if isinstance(full_dataset, Subset):
            targets = [full_dataset.dataset.targets[i] for i in full_dataset.indices]
        else:
            targets = full_dataset.targets

        if isinstance(train_ds, Subset):
            train_targets = [targets[i] for i in train_ds.indices]
        else:
            train_targets = targets

        counts = Counter(train_targets)
        class_weights = {cls: 1.0 / count for cls, count in counts.items()}
        sample_weights = [class_weights[t] for t in train_targets]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    class_names = full_dataset.dataset.classes if isinstance(full_dataset, Subset) else full_dataset.classes
    return train_loader, val_loader, class_names
