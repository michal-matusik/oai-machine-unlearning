"""Evaluation metrics and competition scoring.

This module is competition-provided evaluation infrastructure (originally
marked "DO NOT MODIFY" in the source notebook, used to grade submissions),
translated to English with the exact thresholds/weights preserved. It is not
part of the authored unlearning solution in `src/train.py`.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

WEIGHTS = [0.25, 0.25, 0.25, 0.25]

ACC_FORGET_THRESHOLDS = [0.09, 0.3]
ACC_OTHER_THRESHOLDS = [0.87, 0.90]
DIST_THRESHOLDS = [1.3, 3.0]
DKL_THRESHOLDS = [0.2, 0.5]

ACC_FORGET_ABSOLUTE_THRESHOLD = 0.5
ACC_TARGET_ABSOLUTE_THRESHOLD = 0.75
DISTANCE_ABSOLUTE_THRESHOLD = 8.0
DKL_ABSOLUTE_THRESHOLD = 1.75


def evaluate_target(model: nn.Module, data_loader: DataLoader, device: str = "cpu"):
    """Cross-entropy loss and accuracy of `model` on a labeled data loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return total_loss / total, correct / total


def measure_target_uniformity(model: nn.Module, loader_target: DataLoader, device: str = "cpu", num_classes: int = 10) -> float:
    """Mean KL(p(x) || Uniform) of the model's predictions on the forget-class loader.

    Lower is better: it means predictions on the forgotten class are close to
    a uniform (maximally uninformative) distribution over classes, which is
    the behaviour expected of a model that never saw that data.
    """
    model.eval()
    kl_sum, total = 0.0, 0
    with torch.no_grad():
        for images, _ in loader_target:
            images = images.to(device)
            log_probs = F.log_softmax(model(images), dim=1)
            probs = torch.exp(log_probs)
            entropy_term = (probs * log_probs).sum(dim=1)
            kl_batch = entropy_term + math.log(num_classes)
            kl_sum += kl_batch.sum().item()
            total += images.size(0)
    return kl_sum / total


def measure_l2_distance(model: nn.Module, initial_state_dict: Dict[str, torch.Tensor], device: str = "cpu") -> float:
    """L2 distance between the model's current parameters and its initial checkpoint."""
    l2_distance = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_param = initial_state_dict[name]
            l2_distance += torch.sum((param.to(device) - initial_param.to(device)) ** 2).item()
    return l2_distance ** 0.5


def _scale_metric(value: float, thresholds, maximize: bool = True) -> float:
    if maximize:
        if value >= thresholds[1]:
            return 100.0
        if value <= thresholds[0]:
            return 0.0
        return 100.0 * (value - thresholds[0]) / (thresholds[1] - thresholds[0])
    if value <= thresholds[0]:
        return 100.0
    if value >= thresholds[1]:
        return 0.0
    return 100.0 * (thresholds[1] - value) / (thresholds[1] - thresholds[0])


def compute_final_score(kl_after: float, acc_rest_after: float, l2_dist: float, acc_target_after: float) -> float:
    """Combine the four raw metrics into the official 0-100 competition score."""
    if (
        acc_rest_after < ACC_TARGET_ABSOLUTE_THRESHOLD
        or acc_target_after > ACC_FORGET_ABSOLUTE_THRESHOLD
        or l2_dist > DISTANCE_ABSOLUTE_THRESHOLD
        or kl_after > DKL_ABSOLUTE_THRESHOLD
    ):
        return 0.0

    kl_score = _scale_metric(kl_after, DKL_THRESHOLDS, maximize=False)
    acc_rest_score = _scale_metric(acc_rest_after, ACC_OTHER_THRESHOLDS, maximize=True)
    l2_dist_score = _scale_metric(l2_dist, DIST_THRESHOLDS, maximize=False)
    acc_target_score = _scale_metric(acc_target_after, ACC_FORGET_THRESHOLDS, maximize=False)

    return (
        WEIGHTS[0] * kl_score
        + WEIGHTS[1] * acc_rest_score
        + WEIGHTS[2] * l2_dist_score
        + WEIGHTS[3] * acc_target_score
    )


def evaluate_model(
    model: nn.Module,
    data_loader_target: DataLoader,
    data_loader_rest: DataLoader,
    initial_state_dict: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute all four raw metrics plus the final 0-100 score for a (candidate) unlearned model."""
    _, acc_target_after = evaluate_target(model, data_loader_target, device=device)
    _, acc_rest_after = evaluate_target(model, data_loader_rest, device=device)
    kl_after = measure_target_uniformity(model, data_loader_target, device=device)
    l2_dist = measure_l2_distance(model, initial_state_dict, device=device)

    score = compute_final_score(kl_after, acc_rest_after, l2_dist, acc_target_after)

    return {
        "kl_to_uniform": kl_after,
        "acc_retain": acc_rest_after,
        "l2_distance": l2_dist,
        "acc_forget": acc_target_after,
        "score": score,
    }
