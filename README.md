# Cyber Threat Detection & Anomaly Detection Pipeline

An end-to-end Machine Learning pipeline for detecting anomalies and potential cyber threats in system/network logs. The system ingests raw system/network events, processes them into feature vectors, baseline anomalies using unsupervised methods, models temporal event sequences, classifies specific threat signatures, and provides explainability insights through a real-time security dashboard.

---

## 📂 Repository Structure

```directory
anomaly-detection-cybersec/
├── README.md                          # Master index
├── requirements.txt                   # Dependency list
├── data/
│   ├── raw/                           # Generated synthetic logs (.json, .csv)
│   └── processed/                     # Engineered features and normalized datasets
├── src/
│   ├── phase1_data_gen/               # Synthetic event generator & threat injector
│   │   └── README.md
│   ├── phase2_baseline/               # Unsupervised anomaly baseline models (Isolation Forest, etc.)
│   │   └── README.md
│   ├── phase3_sequence_model/         # Deep sequence modeling (LSTMs / Transformers)
│   │   └── README.md
│   ├── phase4_classifier/             # Supervised threat signature classification
│   │   └── README.md
│   ├── phase5_explainability/         # Model interpretability (SHAP / LIME / Integrated Gradients)
│   │   └── README.md
│   └── phase6_dashboard/              # Real-time incident response dashboard
│       └── README.md
├── notebooks/                         # Exploratory Data Analysis (EDA) and experimental notebooks
├── models/                            # Saved model checkpoints and serialized artifacts
├── reports/                           # Final performance reports and presentation materials
└── docs/                              # System architecture diagrams and technical documentation
```

---

## 🚀 Pipeline Phases

### [Phase 1: Synthetic Data Generation](src/phase1_data_gen/README.md)
Generates high-fidelity synthetic system events (e.g., authentication logs, network flows, process executions) with customizable user personas and injected threat scenarios (e.g., brute-force, data exfiltration, lateral movement).

### [Phase 2: Baseline Unsupervised Models](src/phase2_baseline/README.md)
Establishes a baseline detection rate using statistical thresholds and classical unsupervised models like Isolation Forest, local outlier factor (LOF), or Autoencoders.

### [Phase 3: Deep Sequence Modeling](src/phase3_sequence_model/README.md)
Captures temporal context and sequential patterns in security events to detect slow-and-low attacks and out-of-order execution anomalies using LSTMs, GRUs, or attention-based models.

### [Phase 4: Threat Classification](src/phase4_classifier/README.md)
Classifies identified anomalies and alerts into specific threat categories (MITRE ATT&CK mappings) using semi-supervised or supervised classifiers.

### [Phase 5: Explainability & Interpretability](src/phase5_explainability/README.md)
Explains model predictions by highlighting the features, events, or steps that contributed most to a high anomaly score, aiding security analysts in investigation.

### [Phase 6: Incident Response Dashboard](src/phase6_dashboard/README.md)
A visual dashboard presenting real-time log ingestion, alert timelines, threat classification, and interactive explainability cards.

---

## 🛠️ Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone <repo-url>
   cd cyber_threat_detection_pipeline
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
