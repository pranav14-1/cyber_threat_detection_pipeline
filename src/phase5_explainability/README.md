# Phase 5: Explainability & Interpretability (XAI)

This module provides explanations for the anomaly and threat classifier model predictions. A security model is only as useful as the trust it builds with SOC analysts; this phase ensures every alert comes with explicit reason attributions.

## 🎯 Objectives
- **Local Feature Attribution:** Implement SHAP (SHapley Additive exPlanations) or LIME to explain individual log and event anomaly scores.
- **Sequence Attributions:** Extract and visualize attention weights from Phase 3 sequence models to pinpoint the exact sequence step causing the alert.
- **Analyst-Readable Cards:** Translate complex mathematical feature attributions into natural-language explanation cards (e.g., "Alert triggered due to 10x normal egress bytes to an external IP").

## 🚀 Getting Started
To generate explanation data for an alert:
```bash
python generate_explanations.py --model_path ../../models/classifier.pkl --alert_id 1042
```
