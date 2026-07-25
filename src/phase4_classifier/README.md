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
| `brute_force` | 1.0000 | 1.0000 | 1.0000 |
| `credential_stuffing` | 1.0000 | 0.9904 | 0.9952 |
| `device_spoofing` | 1.0000 | 0.9928 | 0.9964 |
| `impossible_travel` | 0.2443 | 0.9980 | 0.3926 |
| `insider_drift` | 0.8667 | 0.9630 | 0.9123 |
| `lateral_movement` | 0.9828 | 0.9902 | 0.9865 |
| `low_slow_exfil` | 0.9933 | 0.9886 | 0.9910 |
| `unknown` | 0.9979 | 0.7131 | 0.8318 |

**Macro F1-Score:** `0.8882`

## Artifacts
- `models/classifier.pkl`
- `reports/figures/confusion_matrix.png`
- `data/processed/phase4_predictions.csv`
