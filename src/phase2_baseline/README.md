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
| StatProfile | 0.9232 | 0.1995 | 0.0069 |
| IForest | 0.7796 | 0.1257 | 0.0080 |
| OC-SVM | 0.7367 | 0.2734 | 0.0063 |

**Selected Best Model:** `StatProfile`

**Test Set Evaluation Metrics (Best Model):**
- **ROC-AUC:** 0.9072
- **PR-AUC:** 0.2258
- **FPR@1%:** 0.0053

## Cold-Start Policy
Entities with <10 events in the training data use a fallback global population profile, logging a `"cold_start=True"` audit trail flag.

## Artifacts
- `models/baseline.pkl`
- `models/entity_profiles.pkl`
