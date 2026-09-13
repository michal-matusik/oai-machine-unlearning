"""Smoke test for the reconstructed machine-unlearning pipeline.

Uses a freshly-initialized LeNet and small synthetic tensors (no download
required) to check that: (1) `unlearn` leaves the frozen `fc2` layer
byte-identical, as the competition hard-requires, and (2) the metrics module
produces sane values on hand-built inputs. Not an official score -- the
hidden evaluation harness is unavailable.
"""

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.metrics import compute_final_score, measure_l2_distance, measure_target_uniformity
from src.model import LeNet, has_same_last_layer
from src.train import unlearn
from src.utils import set_seed


def _synthetic_loader(label: int, num_images: int = 8, num_classes: int = 10) -> DataLoader:
    images = torch.randn(num_images, 1, 28, 28)
    labels = torch.full((num_images,), label, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def _mixed_loader(exclude: int, num_images: int = 16, num_classes: int = 10) -> DataLoader:
    images = torch.randn(num_images, 1, 28, 28)
    labels = torch.randint(0, num_classes - 1, (num_images,))
    labels = torch.where(labels >= exclude, labels + 1, labels)
    return DataLoader(TensorDataset(images, labels), batch_size=4)


class UnlearningSmokeTest(unittest.TestCase):
    def test_unlearn_freezes_fc2(self):
        set_seed(0)
        model = LeNet(num_classes=10)
        forget_loader = _synthetic_loader(label=3)
        retain_loader = _mixed_loader(exclude=3)

        unlearned = unlearn(model, forget_loader, retain_loader, steps=3, learning_rate=1e-3)

        self.assertTrue(has_same_last_layer(model, unlearned))

    def test_metrics_behave_sensibly(self):
        model = LeNet(num_classes=10)
        initial_state_dict = {name: param.detach().clone() for name, param in model.named_parameters()}

        forget_loader = _synthetic_loader(label=3)
        kl = measure_target_uniformity(model, forget_loader, num_classes=10)
        self.assertGreaterEqual(kl, 0.0)

        l2 = measure_l2_distance(model, initial_state_dict)
        self.assertAlmostEqual(l2, 0.0, places=5)

        # A model that fully forgot (near-chance forget accuracy, uniform
        # predictions, high retain accuracy, tiny drift) should score highly.
        self.assertGreater(compute_final_score(kl_after=0.1, acc_rest_after=0.95, l2_dist=1.0, acc_target_after=0.1), 90.0)
        # A model that did nothing (perfect forget accuracy, no drift) should score 0.
        self.assertEqual(compute_final_score(kl_after=0.1, acc_rest_after=0.95, l2_dist=0.0, acc_target_after=0.95), 0.0)


if __name__ == "__main__":
    unittest.main()
