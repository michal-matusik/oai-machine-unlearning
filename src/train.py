"""Constrained unlearning solution: forget one class while preserving the rest.

The competition provides a pretrained LeNet (`src/model.py`) and asks for a
function that removes its ability to recognize one Fashion-MNIST class,
under a hard constraint (the final classification layer `fc2` must stay
byte-identical) and four scored objectives (minimize forget-class accuracy,
maximize retain-class accuracy, minimize L2 weight drift, and push
forget-class predictions toward a uniform distribution to resist a
membership-inference attack). This is the authored solution.
"""

from copy import deepcopy
from itertools import cycle
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import LeNet


def _feature_extractor_parameters(model: LeNet):
    """All learnable parameters except the frozen classification head (fc2)."""
    return [param for name, param in model.named_parameters() if not name.startswith("fc2")]


def _infinite(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    return iter(cycle(loader))


def unlearn(
    model: LeNet,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    steps: int = 100,
    learning_rate: float = 2e-4,
    alpha: float = 0.0,
    beta: float = 10.0,
    gamma: float = 6.0,
    delta: float = 0.02,
    grad_clip: float = 5.0,
    device: str = "cpu",
) -> LeNet:
    """Return a copy of `model` that has forgotten `forget_loader`'s class.

    Per step, minimizes:
        -alpha * CE(forget_logits, forget_labels)         # gradient ascent: worsen forget-class fit
        + beta  * KL(forget_probs || uniform)             # push forget predictions toward uniform
        + gamma * KL(teacher_probs(retain_x) || student)   # distill original behaviour on retain set
        + delta * ||theta - theta_0||^2                   # proximal term bounding total weight drift

    Only feature-extractor parameters (`block1`, `block2`, `fc`, `fc1`) are
    optimized; `fc2` is frozen throughout and is therefore guaranteed to be
    byte-identical to the input model's `fc2`, matching the competition's
    hard requirement.
    """
    initial_state = {name: param.detach().clone() for name, param in model.named_parameters()}

    teacher = deepcopy(model).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = deepcopy(model).to(device)
    student.train()
    for param in student.fc2.parameters():
        param.requires_grad_(False)

    trainable_params = _feature_extractor_parameters(student)
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    num_classes = student.fc2.out_features
    log_num_classes = torch.log(torch.tensor(float(num_classes), device=device))

    forget_iter = _infinite(forget_loader)
    retain_iter = _infinite(retain_loader)

    for _ in range(steps):
        forget_x, forget_y = next(forget_iter)
        retain_x, _ = next(retain_iter)
        forget_x, forget_y = forget_x.to(device), forget_y.to(device)
        retain_x = retain_x.to(device)

        optimizer.zero_grad()

        forget_logits = student(forget_x)
        ascent_loss = F.cross_entropy(forget_logits, forget_y)

        forget_log_probs = F.log_softmax(forget_logits, dim=1)
        forget_probs = forget_log_probs.exp()
        uniform_kl = (forget_probs * forget_log_probs).sum(dim=1).mean() + log_num_classes

        with torch.no_grad():
            teacher_logits = teacher(retain_x)
        retain_kl = F.kl_div(
            F.log_softmax(student(retain_x), dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )

        proximal = sum(
            torch.sum((param - initial_state[name].to(device)) ** 2)
            for name, param in student.named_parameters()
            if param.requires_grad
        )

        loss = -alpha * ascent_loss + beta * uniform_kl + gamma * retain_kl + delta * proximal

        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

    student.eval()
    assert all(
        torch.equal(p1, p2) for p1, p2 in zip(model.fc2.parameters(), student.fc2.parameters())
    ), "fc2 must remain unchanged by the unlearning procedure."
    return student
