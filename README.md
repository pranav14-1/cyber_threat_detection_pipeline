# AI-Powered Behavioral Anomaly Detection System for Cybersecurity Access Logs

This repository hosts a multi-stage machine learning pipeline designed to ingest, process, detect, classify, explain, and visualize behavioral anomalies and threat tactics in enterprise cybersecurity access logs.

---

## 🔒 Problem Statement

Enterprise environments generate vast quantities of access, authentication, and network logs daily. Traditional Signature-based Intrusion Detection Systems (IDS) fail to identify zero-day exploits, slow-and-low Advanced Persistent Threats (APTs), and insider threat behaviors that do not match predefined signatures. 

This project implements an **AI-powered Behavioral Anomaly Detection pipeline**. By combining statistical baseline models, deep temporal sequence modeling, supervised threat classifiers, and model explainability techniques, the system enables security operations center (SOC) analysts to detect anomalous event sequences and map them to known MITRE ATT&CK techniques with high confidence and transparency.

---

## 📂 7-Deliverable Project Index

1. **[Phase 1: Synthetic Data Generation & Threat Injection](src/phase1_data_gen/README.md)**
   * Simulates enterprise logs and programmatically injects multi-stage attack scenarios.
2. **[Phase 2: Unsupervised Anomaly Detection Baseline](src/phase2_baseline/README.md)**
   * Establishes detection baseline using unsupervised models (e.g., Isolation Forest, LOF).
3. **[Phase 3: Deep Sequence Modeling](src/phase3_sequence_model/README.md)**
   * Detects sequential anomalies using recurrent (LSTM/GRU) or attention-based architectures.
4. **[Phase 4: Threat Signature Classification](src/phase4_classifier/README.md)**
   * Classifies anomalies into specific threat categories mapped to the MITRE ATT&CK framework.
5. **[Phase 5: Model Explainability & Interpretability](src/phase5_explainability/README.md)**
   * Pinpoints why an anomaly was flagged using SHAP/LIME and attention attribution.
6. **[Phase 6: Incident Triage & Explainability Dashboard](src/phase6_dashboard/README.md)**
   * Interactive Streamlit dashboard for real-time alert triage and explanation visualization.
7. **[Phase 7: Exploratory Notebooks & Reports](notebooks/README.md)** (Or check [notebooks/](notebooks/) / [reports/](reports/) / [docs/](docs/))
   * Houses EDA notebooks, evaluation metrics comparisons, and final pipeline design reports.

---

## ✅ Evaluation Criteria Checklist

- [ ] **Data Quality & Realism:** Access logs resemble authentic enterprise metadata (timestamps, IP/port pairs, user IDs, event success flags).
- [ ] **Threat Coverage:** Pipeline injects and successfully triggers alerts for at least 3 distinct MITRE ATT&CK techniques.
- [ ] **Detection Rate:** Unsupervised baselines flag anomalies with a False Positive Rate (FPR) ≤ 5%.
- [ ] **Temporal Sensitivity:** Sequence model successfully flags out-of-order execution flows (e.g., download without prior auth).
- [ ] **Classification Accuracy:** Supervised classifier achieves a Macro F1-score ≥ 85% on labeled threat sequences.
- [ ] **Explainability Coherence:** SHAP values highlight relevant network traffic/activity attributes as primary alert causes.
- [ ] **Dashboard Usability:** Streamlit app loads, renders alert timelines, and displays explanation cards in under 2 seconds.

---

## 🔄 Reproducibility Instructions

Execute the pipeline stages sequentially from the project root using Python:

1. **Generate Synthetic Logs:**
   ```bash
   python -m src.phase1_data_gen.generate
   ```
2. **Train Unsupervised Baseline Models:**
   ```bash
   python -m src.phase2_baseline.train
   ```
3. **Train Temporal Sequence Model:**
   ```bash
   python -m src.phase3_sequence_model.train
   ```
4. **Train Supervised Threat Classifier:**
   ```bash
   python -m src.phase4_classifier.train
   ```
5. **Compute Model Explanations:**
   ```bash
   python -m src.phase5_explainability.explain
   ```
6. **Run Interactive Dashboard:**
   ```bash
   python -m src.phase6_dashboard.app
   ```
