# Reconstructed Machine Unlearning - Polish AI Olympiad II Final

This repository is a reconstructed reference solution for the final-stage Machine Unlearning task in the Polish Artificial Intelligence Olympiad.
It is not the author's original competition submission.

The reference pipeline uses a retain/forget split, performs a small number of ascent steps on the forget set, and constrains drift through distillation on the retain set.
This is a practical approximation to retraining without the forgotten data and documents the crucial utility-forgetting tradeoff.

## Quick start

`python -m unittest discover -s tests -v`

## Validation

The smoke test verifies deterministic forget/retain partitioning and forgetting-score calculation.
It is not an official score because the official hidden evaluation is unavailable.

## Provenance

The original Polish task notebook is retained as `notebooks/original_submission.ipynb`.
`docs/task_en.md` is an English task summary.
