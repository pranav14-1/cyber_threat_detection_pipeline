# Phase 4: Threat Signature Classification

This module classifies detected anomalies and sequences into specific threat categories (e.g., Insider Threat, Brute Force, Privilege Escalation, or Data Exfiltration) corresponding to MITRE ATT&CK techniques.

## 🎯 Objectives
- **Supervised Classification:** Train supervised classifiers (e.g., XGBoost, LightGBM, Random Forest, or Multi-Layer Perceptrons) on labeled anomaly datasets.
- **MITRE ATT&CK Mapping:** Assign specific technique/tactic tags to classified threat alerts to give context to security operators.
- **Confidence Thresholding:** Implement class probability thresholds to ensure low-confidence predictions are flagged as "Unknown/Generic Anomaly" for manual triage.

## 🚀 Getting Started
To train the classifier:
```bash
python train_classifier.py --features ../../data/processed/features_labeled.csv --model xgboost
```
