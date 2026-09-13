# Machine Unlearning on Fashion-MNIST

Make a pretrained LeNet forget one Fashion-MNIST class — without touching its final
classification layer, without hurting accuracy on the other nine classes, and without leaving an
inference-attackable fingerprint of the forgotten class in its predictions.

This repository is a reconstructed reference solution for the final-stage Machine Unlearning task
in the Polish Artificial Intelligence Olympiad (Olympiad II). It is not the author's original
competition submission.

![Machine-unlearning task illustration](assets/task-machine-unlearning.jpg)

*Task illustration referenced by the official Polish AI Olympiad II final notebook.*

## Overview

Machine unlearning asks: given a model trained on some dataset, can we make it behave as if a
specific part of that dataset had never been used to train it — without paying the cost of a
full retrain? This matters for privacy regulation (the "right to be forgotten"), for scrubbing
harmful learned concepts, and for correcting mistakes in training data.

This task provides a **pretrained LeNet classifier** trained on all ten Fashion-MNIST classes,
and asks for a function that makes it forget one target class. The catch: the model's **final
classification layer must stay completely unchanged** — all the unlearning has to happen in the
feature extractor — and a good solution must satisfy four *competing* objectives at once (see
below), not just tank accuracy on the forgotten class.

## Competition Task

**Goal:** implement `unlearn(model, data, target_class)` that modifies a pretrained LeNet so it:

1. **Forgets the target class** — classification accuracy on that class should drop sharply.
2. **Keeps its skill on every other class** — accuracy on the retained classes should stay high.
3. **Changes as little as possible** — the L2 distance between the new and original weights
   should be small.
4. **Doesn't leak evidence of having ever trained on the class** — predictions on the forgotten
   class should look close to a random guess (uniform over classes), which is what a model that
   truly never saw that data would look like. This resists *membership inference attacks*.

**Hard constraint:** the final layer (`fc2`) must be byte-identical before and after — only the
feature-extractor layers (`block1`, `block2`, `fc`, `fc1`) may change.

**Data:** the full Fashion-MNIST training set, pre-split by class into a "forget" loader (only
the target class) and a "retain" loader (every other class), plus an officially-provided
pretrained checkpoint. The worked example in the source notebook forgets class 9 (ankle boot);
the actual hidden test set grades an undisclosed different class.

**Evaluation metric:** the final score (0–100) combines four sub-scores (each weighted 25%),
computed from: KL-divergence of forget-class predictions to uniform (lower is better), retained-
class accuracy (higher is better), L2 weight distance (lower is better), and forget-class
accuracy (lower is better). Each is linearly scaled between fixed thresholds, and the whole score
is **zeroed** if any metric crosses a hard absolute threshold (e.g. retain accuracy dropping below
75%, or forget accuracy staying above 50%) — see [Model Architecture](#model-architecture) below
and `src/metrics.py: compute_final_score` for the exact numbers.

## Approach

- **Base model (`src/model.py: LeNet`):** the competition-provided architecture, loaded from the
  officially-provided checkpoint (`data/lenet_base_final.pt`, fetched via
  `scripts/download_data.py`).
- **Data (`src/dataset.py`):** `torchvision.datasets.FashionMNIST` filtered per class into a
  forget loader (target class only) and a retain loader (the other nine), using the notebook's
  exact normalization (`mean=0.2860, std=0.3530`).
- **Unlearning procedure (`src/train.py: unlearn`):** a small number of gradient steps on a
  combined objective, optimizing only the feature-extractor parameters (`fc2` is excluded from
  the optimizer and therefore provably unchanged):
  1. **Forget-uniformity term** — directly minimizes `KL(forget_predictions || uniform)`,
     pushing predictions on the forgotten class toward a maximally uninformative distribution.
     This simultaneously drives down forget-class accuracy (a uniform 10-way distribution scores
     ~10% accuracy) *and* targets the anti-membership-inference objective, rather than treating
     them as separate goals.
  2. **Retain-distillation term** — KL-divergence between the current model's predictions and
     the *original* (frozen teacher) model's predictions on retain-set batches, which keeps
     retain-class behavior close to the pretrained model throughout the procedure.
  3. **Proximal term** — an explicit L2 penalty between current and initial weights, directly
     bounding the third scored metric rather than relying only on a small learning rate and few
     steps to keep drift small.
  4. *(Available but disabled by default, `alpha=0`)* An explicit forget-class gradient-ascent
     term (`-alpha * CE(forget_logits, forget_labels)`) — kept in the implementation since it is
     a common unlearning technique, but the uniformity term alone already drives forget accuracy
     down effectively without the risk of ascent producing *confidently wrong* (rather than
     uniform) forget predictions.
- **Reproducibility:** a fixed seed (101, matching the source notebook) is set for Python, NumPy,
  and PyTorch.

## Model Architecture

```text
Input [B, 1, 28, 28]
  |
  |-- Feature extractor (only these layers may change) -----------
  |     block1: Conv5x5(1->6) -> BN -> ReLU -> MaxPool2x2   [B, 6, 12, 12]
  |     block2: Conv5x5(6->16) -> BN -> ReLU -> MaxPool2x2  [B, 16, 4, 4]
  |     flatten -> fc(256->120) -> ReLU -> fc1(120->84) -> ReLU
  |
  |-- Classification head (frozen throughout unlearning) ----------
        fc2(84 -> 10)  -> class logits
```

## Unlearning Procedure

| Parameter | Value |
|---|---:|
| Steps | 100 |
| Learning rate | 2e-4 (Adam, feature-extractor parameters only) |
| Forget-uniformity weight (`beta`) | 10.0 |
| Retain-distillation weight (`gamma`) | 6.0 |
| Proximal (L2-to-initial) weight (`delta`) | 0.02 |
| Forget gradient-ascent weight (`alpha`) | 0.0 (disabled) |
| Seed | 101 |

## Results

Obtained by running `python -m src.evaluate` (which downloads Fashion-MNIST automatically and
loads the pretrained checkpoint fetched via `scripts/download_data.py`; CPU, seed 101) for the
notebook's worked example, **target class 9** (ankle boot):

```text
Target class:                  9
KL(forget || uniform):         0.1842
Retain-set accuracy:           0.9017
L2 weight distance:            1.3698
Forget-set accuracy:           0.1808
Estimated task score (0-100):  88.16
```

Before unlearning, the pretrained model scores `forget accuracy = 0.973`, `retain accuracy =
0.931`, `KL to uniform = 2.23` (i.e. very confident, non-uniform predictions), `L2 distance = 0`
— confirming the unlearning procedure meaningfully changes forget-class behavior while barely
touching retain-class accuracy.

Since the actual hidden test set grades an **undisclosed** target class, the same hyperparameters
were also checked against four other target classes with no per-class retuning:

| Target class | KL to uniform | Retain acc. | L2 distance | Forget acc. | Score |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2218 | 0.8994 | 1.4728 | 0.0717 | 95.1 |
| 3 | 0.1186 | 0.9088 | 1.4540 | 0.4028 | 72.7 |
| 5 | 0.5393 | 0.9053 | 1.4556 | 0.2272 | 56.4 |
| 7 | 0.3103 | 0.9059 | 1.3186 | 0.6805 | 0.0 |
| **9** (worked example) | 0.1842 | 0.9017 | 1.3698 | 0.1808 | **88.2** |

Scores vary considerably by class (some classes are visually/semantically closer to others in
Fashion-MNIST, e.g. shirt/coat/pullover, making them harder to forget without hurting retain
accuracy) — see [Limitations](#limitations-1) below. These are all local re-runs, not the hidden
test-set score the graded submission would receive.

## Repository Structure

```text
.
├── README.md                       # This file
├── SOLUTION.md                     # Detailed technical write-up
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment specification
├── configs/
│   └── config.yaml                 # Hyperparameters
├── docs/
│   └── task_en.md                  # English task summary
├── assets/
│   └── task-machine-unlearning.jpg
├── scripts/
│   └── download_data.py            # Fetches the pretrained checkpoint from Google Drive
├── src/
│   ├── dataset.py                  # Fashion-MNIST loading, filtered by class (competition-provided)
│   ├── model.py                    # LeNet architecture + fc2-preservation checks (competition-provided)
│   ├── metrics.py                  # KL/L2/accuracy metrics + competition scoring (competition-provided)
│   ├── train.py                    # `unlearn()` (authored solution)
│   ├── evaluate.py                 # Evaluation entry point
│   └── utils.py                    # Seeding + device selection
├── tests/
│   └── test_smoke.py               # Synthetic-tensor smoke test (fc2 preservation, metric sanity)
└── notebooks/
    └── original_submission.ipynb   # Original competition notebook, kept for provenance
```

`data/` (the pretrained checkpoint and the downloaded Fashion-MNIST archive) is not tracked in
git — fetch the checkpoint with `python scripts/download_data.py`; Fashion-MNIST itself downloads
automatically the first time `src.dataset.setup_data` runs.

## Running the Project

```bash
pip install -r requirements.txt

python scripts/download_data.py   # downloads data/lenet_base_final.pt

python -m src.evaluate              # unlearns configs/config.yaml's target_class and scores it
python -m src.evaluate --target-class 3   # try a different class
```

Or, to run the smoke test on synthetic tensors (no download required):

```bash
python -m unittest discover -s tests -v
```

## Technical Highlights

- A single KL-to-uniform loss term simultaneously targets two of the four scored objectives
  (forget-class accuracy and anti-membership-inference uniformity), rather than treating them as
  separate loss terms that could pull in different directions.
- Retain-set knowledge distillation against a frozen copy of the original model directly protects
  the retain-accuracy objective throughout the unlearning procedure.
- An explicit proximal term bounds weight drift directly, rather than relying solely on a small
  learning rate/step count to keep the L2-distance metric under control.
- `fc2` is provably unchanged: it is excluded from the optimizer's parameter list entirely
  (verified by an assertion inside `unlearn`, and by the `test_unlearn_freezes_fc2` smoke test),
  rather than relying on a post-hoc equality check.

## Limitations

- Hyperparameters (`steps`, `learning_rate`, `beta`, `gamma`, `delta`) were tuned against the
  notebook's own worked example, target class 9. As the cross-class table above shows, the same
  fixed hyperparameters do not perform equally well on every class (e.g. class 7 fails the hard
  absolute-threshold check entirely) — a more robust solution would adapt the loss weights or
  stopping criterion per target class, e.g. based on how quickly the forget-class KL crosses a
  threshold, rather than using a fixed step count.
- The gradient-ascent term (`alpha`) is implemented but disabled by default; a hybrid schedule
  (e.g. ascent early, uniformity-only later) was not explored.
- No held-out "hidden test" class was available to validate against; all numbers above are
  local re-runs on the full Fashion-MNIST training split, not the actual grading data.

## Potential Improvements

- Make the unlearning procedure adaptive: stop as soon as the forget-KL and forget-accuracy cross
  their full-credit thresholds, rather than running a fixed step count, to reduce unnecessary
  retain-accuracy and L2-distance cost.
- Sweep hyperparameters per-class (or find a single robust setting) using several target classes
  rather than only the notebook's worked example.
- Explore alternating (rather than jointly-weighted) forget/retain update steps.
- Track experiment metrics (e.g. with TensorBoard) instead of print-based logging.

---

### Resume Version

**Project title**

Constrained Machine Unlearning on Fashion-MNIST (PyTorch)

**One-line description**

Designed a constrained-optimization unlearning procedure that removes a pretrained classifier's
ability to recognize one class while provably preserving its classification head and minimizing
collateral damage to unrelated classes.

**Resume bullets**

- Designed a multi-term loss (forget-class KL-to-uniform, retain-set knowledge distillation,
  proximal weight-drift penalty) that jointly optimizes four competing unlearning objectives in a
  single training loop, rather than treating them as independent post-hoc adjustments.
- Enforced a hard architectural constraint (an untouched classification layer) by excluding it
  from the optimizer entirely, verified with an in-code assertion and a dedicated unit test.
- Evaluated the solution across multiple target classes to characterize where a fixed
  hyperparameter setting generalizes and where it does not, rather than reporting a single
  cherry-picked result.
