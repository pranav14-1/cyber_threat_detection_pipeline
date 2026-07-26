# Phase 2 — Baseline Profiling Model

## Purpose
Establish a per-entity representation of "normal" so any large deviation
produces an initial anomaly score. Serves as a competitive baseline and a
warm-start for the sequence model.

## Models Evaluated
- Statistical Profile (KDE + distance scores)
- Isolation Forest
- One-Class SVM

## Best Model & Metrics
The model performance was evaluated using a chronological split (70% train / 15% validation / 15% test). The metrics are listed below:

| Model | ROC-AUC | PR-AUC | FPR@1% |
|-------|---------|--------|--------|
| StatProfile | 0.9233 | 0.2013 | 0.0069 |
| IForest | 0.7690 | 0.1115 | 0.0079 |
| OC-SVM | 0.7342 | 0.2730 | 0.0063 |

**Selected Best Model:** `StatProfile`

**Test Set Evaluation Metrics (Best Model):**
- **ROC-AUC:** 0.9070
- **PR-AUC:** 0.2246
- **FPR@1%:** 0.0054

## Cold-Start Policy
Entities with <10 events in the training data use a fallback global population profile, logging a `"cold_start=True"` audit trail flag.

## Artifacts
- `models/baseline.pkl`
- `models/entity_profiles.pkl`
