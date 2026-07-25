# Phase 3: Deep Sequence Modeling for Threat Detection

This module focuses on capturing temporal relationships and patterns across sequences of system events. Cyber attacks often manifest not as a single outlier event, but as a specific, ordered sequence of actions (e.g., login -> lateral movement -> file compression -> egress traffic).

## 🎯 Objectives
- **Temporal Modeling:** Implement sequence-aware architectures (e.g., LSTMs, GRUs, or Transformer encoders) to model log event sequences.
- **Next-Event Prediction / Sequence Anomaly:** Train models to predict the next log action or identify out-of-order/anomalous execution paths.
- **Attention Mapping:** Utilize attention mechanisms to identify which step in an event sequence contributed most to the anomaly designation.

## 🚀 Getting Started
To train the sequence model:
```bash
python train_sequence.py --input ../../data/processed/sequences/ --epochs 20
```
