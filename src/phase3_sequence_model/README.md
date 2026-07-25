# Phase 3 — Sequence-Aware Detection Model

## Purpose
Capture temporal dependencies in access sequences that a static baseline
cannot — e.g., "this user never accesses resource X immediately after resource Y."

## Architecture
- BiLSTM Autoencoder (primary), Transformer encoder (alternative).
- Reconstruction MSE per event → anomaly score.

## Training Setup
- 70/15/15 chronological split, normal-only training.
- Adam, lr=1e-3, 20 epochs, early stopping.

## Results
The performance of the models evaluated on the chronological test split (15% test set) is summarized below:

| Model | ROC-AUC | PR-AUC | P@1% |
|-------|---------|--------|------|
| BiLSTM-AE | 0.7809 | 0.3312 | 0.5507 |
| Transformer | 0.0000 | 0.0000 | 0.0000 |
| Baseline (Phase 2) | 0.9683 | 0.6602 | 0.8094 |

**Selected Best Sequence Model:** `BILSTM`

## Concept-Drift Handling
- Weekly rolling retrain hook exposed in train.py (`--incremental` flag).
- Score normalization uses trailing-30-day rolling min-max.
