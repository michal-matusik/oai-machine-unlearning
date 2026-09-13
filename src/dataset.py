"""Fashion-MNIST loading utilities, filtered by class.

This module is lightly-adapted competition-provided infrastructure
(originally marked "DO NOT MODIFY" in the source notebook): it builds a
forget-set loader (only the target class) and a retain-set loader (every
other class), using the exact normalization constants from the source
notebook.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NORMALIZE_MEAN = (0.2860,)
NORMALIZE_STD = (0.3530,)


def _transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


class FilteredFashionMNIST(datasets.FashionMNIST):
    """Fashion-MNIST restricted to a subset of class labels."""

    def __init__(self, *args, classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_kept = classes
        if classes is not None:
            self.data, self.targets = self._filter_classes(self.data, self.targets, classes)

    @staticmethod
    def _filter_classes(data, targets, classes):
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for c in classes:
            mask = mask | (targets == c)
        return data[mask], targets[mask]


def setup_data(
    target_class: int,
    data_root: str = "./data",
    batch_size: int = 64,
    num_classes: int = 10,
) -> Tuple[DataLoader, DataLoader]:
    """Build (forget_loader, retain_loader) for a given target class.

    `forget_loader` yields only images labeled `target_class`; `retain_loader`
    yields every other class. Both use the training split of Fashion-MNIST.
    """
    os.makedirs(data_root, exist_ok=True)
    transform = _transform()

    forget_dataset = FilteredFashionMNIST(
        root=data_root, train=True, download=True, transform=transform, classes=[target_class]
    )
    retain_classes = [c for c in range(num_classes) if c != target_class]
    retain_dataset = FilteredFashionMNIST(
        root=data_root, train=True, download=True, transform=transform, classes=retain_classes
    )

    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    return forget_loader, retain_loader
