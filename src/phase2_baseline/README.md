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
| StatProfile | 0.9062 | 0.3834 | 0.0038 |
| IForest | 0.9133 | 0.2880 | 0.0067 |
| OC-SVM | 0.9755 | 0.7717 | 0.0013 |

**Selected Best Model:** `OC-SVM`

**Test Set Evaluation Metrics (Best Model):**
- **ROC-AUC:** 0.9637
- **PR-AUC:** 0.6285
- **FPR@1%:** 0.0021

## Cold-Start Policy
Entities with <10 events in the training data use a fallback global population profile, logging a `"cold_start=True"` audit trail flag.

## Artifacts
- `models/baseline.pkl`
- `models/entity_profiles.pkl`
