# Phase 1 — Synthetic Data Generator

## Purpose
Generate labeled access-log data because real cybersecurity intrusion datasets
are scarce, privacy-restricted, and domain-specific.

## Behavioural Assumptions
- Users have habitual work-hour Gaussian distributions.
- Geo-location is stable per entity (home region ± noise).
- Resources accessed follow a Zipf-like preference distribution.
- 98% of events are benign; 2% split across 7 anomaly types.

## Injected Attack Taxonomy
| Type | Rate | Signature |
|------|------|-----------|
| brute_force | 0.4% | high-freq fail from 1 IP |
| impossible_travel | 0.3% | geo-velocity > 1000 km/h |
| credential_stuffing | 0.3% | many IDs, few IPs, high fail |
| lateral_movement | 0.3% | new-resource burst |
| device_spoofing | 0.2% | MAC/OS mismatch |
| low_slow_exfil | 0.3% | off-hours accumulation |
| insider_drift | 0.2% | edge case, gradual expansion |

## Outputs
- `data/raw/logs.csv` — N events
- `data/raw/labels.csv` — ground truth (hidden from model at inference)
- `data/raw/entity_profiles.json`

## Reproducibility
```bash
python -m src.phase1_data_gen.generate --config config.yaml
```
