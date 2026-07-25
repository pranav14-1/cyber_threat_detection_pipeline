# Phase 2: Unsupervised Anomaly Detection Baseline

This module establishes a baseline detection capability using classic unsupervised ML algorithms and statistical outlier detection methods. It consumes processed feature vectors from Phase 1 and flags unusual behaviors without requiring labeled historical threat data.

## 🎯 Objectives
- **Establish Baselines:** Implement classical unsupervised algorithms (e.g., Isolation Forest, Local Outlier Factor, One-Class SVM, and PyOD autoencoders) to evaluate baseline detection metrics.
- **Unsupervised Evaluation:** Evaluate the false positive rate (FPR) vs. true positive rate (TPR) when logs are passed directly through unsupervised models.
- **Feature Importance Mapping:** Capture basic outlier scores and feature attribution for identified anomalies to pass downstream.

## 🚀 Getting Started
To train the baseline models and generate outlier scores:
```bash
python train_baseline.py --input ../../data/processed/features.csv --model isolation_forest
```
