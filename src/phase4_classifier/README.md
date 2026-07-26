# Phase 4 — Anomaly Classification

## Purpose
Move from "anomalous / not anomalous" to naming the attack category so security analysts
know how to respond (brute force ≠ insider drift ≠ impossible travel).

## Model
XGBoost Multi-Class (8 classes: 7 attack types + `unknown`), balanced sample weights, `max_delta_step=1`, `subsample=0.8`, `learning_rate=0.08`, `n_estimators=150`,
5-fold stratified CV with deterministic rule assist.

## Deterministic Overrides
| Rule | Trigger | Assigned Class |
|------|---------|----------------|
| `geo_velocity_kmh > 900 AND geo_distance_prev_km > 500` | strictly satisfied | `impossible_travel` |
| `failed_auth_count_5min > 30` | strictly satisfied | `brute_force` |

## Results (5-Fold Stratified Cross-Validation)
| Attack Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| `brute_force` | 0.9974 | 1.0000 | 0.9987 |
| `credential_stuffing` | 1.0000 | 0.9969 | 0.9984 |
| `device_spoofing` | 1.0000 | 1.0000 | 1.0000 |
| `impossible_travel` | 0.9930 | 1.0000 | 0.9965 |
| `insider_drift` | 0.6724 | 0.9435 | 0.7852 |
| `lateral_movement` | 0.8662 | 0.9173 | 0.8910 |
| `low_slow_exfil` | 0.9742 | 1.0000 | 0.9869 |
| `unknown` | 0.9945 | 0.9660 | 0.9800 |

**Macro F1-Score:** `0.9546`

## Artifacts
- `models/classifier.pkl`
- `reports/figures/confusion_matrix.png`
- `data/processed/phase4_predictions.csv`
