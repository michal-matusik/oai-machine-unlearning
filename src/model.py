"""LeNet architecture used by the base (pre-unlearning) model.

This is the competition-provided architecture (originally marked
"NIE ZMIENIAJ TEJ KOMORKI" / "DO NOT MODIFY" in the source notebook): a
classic LeNet-style CNN trained on all ten Fashion-MNIST classes. The
unlearning solution in `src/train.py` must not change the shape or values of
`fc2` (the final classification layer) -- only the feature-extractor layers
(`block1`, `block2`, `fc`, `fc1`) may be modified.
"""

import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        latent_dim = 256
        self.fc = nn.Linear(latent_dim, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def load_pretrained(checkpoint_path: str, num_classes: int = 10) -> LeNet:
    """Load the officially-provided base LeNet checkpoint."""
    model = LeNet(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def has_same_last_layer(model1: LeNet, model2: LeNet) -> bool:
    """True if fc2 (the classification head) is identical between two models."""
    return all(torch.equal(p1, p2) for p1, p2 in zip(model1.fc2.parameters(), model2.fc2.parameters()))


def has_same_architecture(model: LeNet) -> bool:
    expected = {
        "block1.0.weight", "block1.0.bias", "block1.1.weight", "block1.1.bias",
        "block2.0.weight", "block2.0.bias", "block2.1.weight", "block2.1.bias",
        "fc.weight", "fc.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
    }
    return {name for name, _ in model.named_parameters()} == expected
