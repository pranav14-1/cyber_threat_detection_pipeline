# Phase 4 — Anomaly Classification

## Purpose
Move from "anomalous / not anomalous" to naming the attack category so security analysts
know how to respond (brute force ≠ insider drift ≠ impossible travel).

## Model
LightGBM / GBDT multi-class (8 classes: 7 attack types + `unknown`), balanced class weights,
5-fold stratified CV with deterministic rule assist.

## Deterministic Overrides
| Rule | Trigger | Assigned Class |
|------|---------|----------------|
| `geo_velocity_kmh > 1000` | always | `impossible_travel` |
| `failed_auth_count_5min > 30` | always | `brute_force` |

## Results (5-Fold Stratified Cross-Validation)
| Attack Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| `brute_force` | 1.0000 | 1.0000 | 1.0000 |
| `credential_stuffing` | 0.9991 | 0.9912 | 0.9952 |
| `device_spoofing` | 1.0000 | 0.9928 | 0.9964 |
| `impossible_travel` | 0.2460 | 0.9980 | 0.3947 |
| `insider_drift` | 0.9099 | 0.9537 | 0.9313 |
| `lateral_movement` | 0.9892 | 0.9935 | 0.9913 |
| `low_slow_exfil` | 0.9971 | 0.9905 | 0.9938 |
| `unknown` | 0.9971 | 0.7184 | 0.8351 |

**Macro F1-Score:** `0.8922`

## Artifacts
- `models/classifier.pkl`
- `reports/figures/confusion_matrix.png`
