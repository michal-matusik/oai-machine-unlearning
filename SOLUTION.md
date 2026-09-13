# Technical Write-Up

This document explains the reasoning and engineering decisions behind the solution in
`src/train.py` (`unlearn`). It distinguishes facts directly supported by the notebook/implementation
from inferred interpretation of *why* certain choices were made — the original notebook's
`unlearn` function is a stub (`return model`), so the entire approach below is authored for this
reconstruction, with design rationale flagged as inferred where it goes beyond what the grading
code itself implies.

## 1. Problem Formulation

Given a pretrained `LeNet` classifier trained on all ten Fashion-MNIST classes, and a target
class `i`, produce a modified model `f'` such that:

- `f'.fc2` is byte-identical to `f.fc2` (hard constraint, checked by
  `has_same_last_layer`/`has_same_architecture` in the source notebook).
- Accuracy of `f'` on class-`i` examples is low (forgetting).
- Accuracy of `f'` on all other classes is high (utility preservation).
- `||f'.params - f.params||_2` is small (minimal intervention).
- The predicted-class distribution of `f'` on class-`i` examples is close to uniform over the 10
  classes (resistance to membership inference — see the notebook's own framing, cell 8: the goal
  is not just low accuracy but behaving as if the class had never been trained on).

## 2. Dataset and Preprocessing

`src/dataset.py` loads `torchvision.datasets.FashionMNIST` and splits the training set by label
into a forget loader (only the target class) and a retain loader (every other class), using the
notebook's own normalization constants (`mean=0.2860, std=0.3530`, cell 37). This matches the
source notebook's `FilteredFashionMNIST`/`get_data_dict` cells exactly, translated from a
notebook-global dict of ten precomputed loaders into a function that builds the two loaders
needed for one target class on demand.

## 3. Initial Considerations

The notebook explicitly separates two things that are easy to conflate: "the model can no longer
correctly classify the forgotten class" versus "the model behaves as if it never saw that class"
(cell 8). A naive approach — e.g. plain gradient ascent on the forget-class cross-entropy loss,
sketched informally as a starting idea — pushes accuracy down, but *(inferred)* has no reason to
land on a *uniform* prediction distribution: ascent can just as easily converge to confidently
predicting a *different specific wrong class*, which is arguably even easier for a membership-
inference attacker to detect (a suspiciously consistent wrong answer is itself a signal) than
low-confidence, spread-out predictions. This was confirmed empirically during development: an
early version of this solution using only gradient ascent produced a forget-class KL-to-uniform
of ~2.3 (near its theoretical maximum, `log(10) = 2.303` — i.e. *maximally* non-uniform,
confidently-wrong predictions) rather than the desired near-zero value.

## 4. Chosen Approach

The implemented solution (`src/train.py: unlearn`) directly optimizes a weighted combination of
three differentiable terms, over a small number of gradient steps on the feature-extractor
parameters only (`fc2` excluded from the optimizer's parameter list, and therefore provably
unchanged — not merely restored after the fact):

1. **Forget-uniformity term** (weight `beta`): `KL(softmax(forget_logits) || Uniform)`, computed
   with the same formula used for grading (`measure_target_uniformity` in `src/metrics.py`), so
   the training objective directly targets the *actual* scored quantity rather than a proxy for
   it. This single term simultaneously drives down forget-class accuracy (since a uniform
   10-class distribution is correct only ~10% of the time) and drives the uniformity metric
   toward its ideal value, addressing two of the four scored objectives at once.
2. **Retain-distillation term** (weight `gamma`): KL-divergence between the *current* model's
   predictions and a frozen copy of the *original* model's predictions on retain-set batches
   (standard knowledge distillation). This is the same idea as the notebook's own draft
   `unlearning_step` cell, kept because distilling against the original model's own retain-set
   behavior is a more direct way to preserve utility than merely regularizing weights, since it
   optimizes the actual retain-accuracy-relevant quantity (output distribution agreement) rather
   than a weight-space proxy for it.
3. **Proximal term** (weight `delta`): an explicit `||params - initial_params||^2` penalty over
   the trainable (non-`fc2`) parameters. *(Inferred)* Added beyond the notebook's own draft
   because the L2-distance metric is one of the four scored quantities with a fairly tight
   full-credit threshold (`< 1.3`); relying only on "small learning rate, few steps" to keep drift
   bounded is indirect and harder to reason about than penalizing drift directly in the loss.
4. **Gradient-ascent term** (weight `alpha`, default `0.0`): `-CE(forget_logits, forget_labels)`,
   kept in the implementation (it is a standard unlearning building block, and the notebook's own
   informal framing gestures at it) but disabled by default per the finding in Section 3 above —
   with `beta` already targeting the uniformity objective directly, adding an ascent term on top
   was found during tuning to just as often accelerate collapse toward a confident-wrong
   prediction as toward a uniform one, without clearly improving the scored metrics.

## 5. Hyperparameter Selection

*(This entire section is empirical, arrived at by sweeping — no equivalent tuning existed in the
source notebook.)* Starting from an equal-weight combination (`alpha=1, beta=1, gamma=2`), a
first pass produced badly overfit-to-ascent behavior: forget accuracy hit 0 but retain accuracy
also collapsed to ~0.42 and KL-to-uniform stayed near its non-uniform maximum. Disabling ascent
(`alpha=0`) and searching over `(steps, learning_rate, beta, gamma, delta)` on the notebook's
worked-example target class (9) found that:

- Learning rate needed to be in a fairly narrow band (`~2e-4`): much lower made no visible
  progress within a reasonable step budget; much higher destroyed retain accuracy before the
  uniformity term could act.
- Increasing `gamma` (retain-distillation weight) relative to `beta` (uniformity weight) was
  necessary to keep retain accuracy above the `0.90` full-credit threshold once enough steps were
  used to meaningfully reduce forget-class KL.
- A small explicit proximal weight (`delta=0.02`) kept L2 distance in the `1.3`-ish range
  (comfortably inside the `<3.0` non-zero-scoring band, close to the `<1.3` full-credit band)
  without needing to further shrink the learning rate or step count.

The final configuration (`configs/config.yaml`: `steps=100, learning_rate=2e-4, beta=10, gamma=6,
delta=0.02`) reaches an estimated score of 88.16/100 on target class 9. See
[README's Results section](README.md#results) for the full breakdown and a cross-class check.

## 6. Evaluation Methodology

`src/metrics.py` reconstructs the notebook's `evaluate_target`/`measure_target_uniformity`/
`measure_l2_distance`/`compute_final_score` cells verbatim, including the exact threshold lists
(`acc_forget_thresholds`, `acc_other_thresholds`, `dist_thresholds`, `d_kl_thresholds`) and hard
absolute-threshold zeroing conditions (e.g. the whole score is 0 if retain accuracy drops below
0.75, regardless of how good the other three metrics are). This means small changes near a
threshold boundary can have an outsized effect on the reported score — visible in the cross-class
results table, where class 7 scores 0 entirely because forget accuracy (0.68) exceeds the 0.5
absolute cutoff, despite the other three metrics being reasonable.

## 7. Results

See [README's Results section](README.md#results): target class 9 (the notebook's worked
example) scores 88.16/100 locally; four additional classes were checked with the same fixed
hyperparameters and scored between 0 and 95.1, illustrating that this fixed configuration is not
uniformly robust across all ten classes (see Limitations).

## 8. Why the Approach Works

*(Inferred, based on standard machine-learning reasoning — not stated in any source material.)*
Directly minimizing `KL(forget_predictions || uniform)` as a differentiable loss term is more
targeted than minimizing forget-class accuracy alone, because accuracy is a discontinuous,
non-differentiable proxy for "the model behaves as if it never saw this data" — many very
different prediction distributions can share the same (low) accuracy, but only a distribution
close to uniform also satisfies the anti-membership-inference objective the notebook cares about.
Distilling against the frozen original model on the retain set gives the optimizer a much richer
training signal for utility preservation than a plain classification loss would (it matches the
full output distribution, not just the argmax), which matters when only a handful of gradient
steps are available before weight drift (and therefore the L2 metric) becomes too large.

## 9. Limitations

- The forget/retain/proximal loss weights were tuned on one target class and do not generalize
  uniformly to all ten (see the cross-class table in the README); a production-quality solution
  would need either per-class tuning or an adaptive stopping rule.
- The proximal term (`delta`) is a heuristic weight, not derived from the actual `1.3`/`3.0` L2
  thresholds in `src/metrics.py` — it was tuned qualitatively rather than analytically.
- No formal membership-inference attack was run against the "unlearned" model; the KL-to-uniform
  metric is a reasonable but indirect proxy for actual attack resistance.
- The hidden competition test set grades an undisclosed class different from the notebook's
  worked example (9); the reported numbers are local re-runs on the public Fashion-MNIST training
  split, not the actual graded result.

## 10. Potential Improvements

See the [README's Potential Improvements section](README.md#potential-improvements) — these are
proposed next steps, not part of the current solution.
