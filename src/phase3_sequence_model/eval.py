"""
Phase 3: Model Evaluation
Evaluates the trained sequence-aware model against the Phase 2 baseline.
Computes ROC-AUC, PR-AUC, and Precision@top-1%, and generates the ensembled score output.
"""

import os
import sys
import yaml
import joblib
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

from src.phase3_sequence_model.dataset import SequenceFeatureExtractor, AccessLogSequenceDataset
from src.phase3_sequence_model.model import BiLSTMAutoencoder, TransformerAutoencoder
from src.phase3_sequence_model.score import compute_timestep_scores, calibrate_min_max, apply_static_normalization, apply_rolling_normalization
from src.phase2_baseline.train import PipelineBaselineTrainer, StatisticalProfiler, safe_json_loads

# Bind StatisticalProfiler on current __main__ namespace to resolve any baseline.pkl pickle stream references
setattr(sys.modules["__main__"], "StatisticalProfiler", StatisticalProfiler)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_precision_at_top_1(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate precision at the top-1% alert budget threshold."""
    threshold = np.percentile(y_scores, 99.0)
    y_pred = (y_scores >= threshold).astype(int)
    
    # TP / (TP + FP)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    total_flagged = tp + fp
    return float(tp / total_flagged) if total_flagged > 0 else 0.0

def evaluate_predictions(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, float]:
    """Compute ROC-AUC, PR-AUC, and Precision@top-1%."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    roc_auc = float(roc_auc_score(y_true, y_scores))
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = float(auc(recall, precision))
    p_at_1 = compute_precision_at_top_1(y_true, y_scores)
    
    return roc_auc, pr_auc, p_at_1

def update_readme(bilstm_metrics: Tuple[float, float, float],
                  transformer_metrics: Tuple[float, float, float],
                  baseline_metrics: Tuple[float, float, float],
                  best_sequence_model: str):
    """Dynamically update Phase 3 README.md with the comparative evaluation results."""
    readme_path = "src/phase3_sequence_model/README.md"
    
    markdown_content = f"""# Phase 3 — Sequence-Aware Detection Model

## Purpose
Capture temporal dependencies in access sequences that a static baseline
cannot — e.g., "this user never accesses resource X immediately after resource Y."

## Architecture
- BiLSTM Autoencoder (primary), Transformer encoder (alternative).
- Reconstruction MSE per event → anomaly score.

## Training Setup
- 70/15/15 chronological split, normal-only training.
- Adam, lr=1e-3, 20 epochs, early stopping.

## Results
The performance of the models evaluated on the chronological test split (15% test set) is summarized below:

| Model | ROC-AUC | PR-AUC | P@1% |
|-------|---------|--------|------|
| BiLSTM-AE | {bilstm_metrics[0]:.4f} | {bilstm_metrics[1]:.4f} | {bilstm_metrics[2]:.4f} |
| Transformer | {transformer_metrics[0]:.4f} | {transformer_metrics[1]:.4f} | {transformer_metrics[2]:.4f} |
| Baseline (Phase 2) | {baseline_metrics[0]:.4f} | {baseline_metrics[1]:.4f} | {baseline_metrics[2]:.4f} |

**Selected Best Sequence Model:** `{best_sequence_model}`

## Concept-Drift Handling
- Weekly rolling retrain hook exposed in train.py (`--incremental` flag).
- Score normalization uses trailing-30-day rolling min-max.
"""
    with open(readme_path, "w") as f:
        f.write(markdown_content)
    print(f"Updated {readme_path} with Phase 3 comparative performance metrics.")

def main():
    config = load_config()
    
    # Load dataset
    print("Loading data for evaluation...")
    df_logs = pd.read_csv("data/raw/logs.csv")
    df_labels = pd.read_csv("data/raw/labels.csv")
    df = pd.merge(df_logs, df_labels, on="event_id")
    
    # Parse nested columns for baseline compatibility
    df["geo_location_parsed"] = df["geo_location"].apply(safe_json_loads)
    df["device_fingerprint_parsed"] = df["device_fingerprint"].apply(safe_json_loads)
    df["lat"] = df["geo_location_parsed"].apply(lambda x: x.get("lat", 0.0))
    df["lon"] = df["geo_location_parsed"].apply(lambda x: x.get("lon", 0.0))
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Split chronologically
    n = len(df)
    train_idx = int(0.70 * n)
    val_idx = int(0.85 * n)
    
    df_train = df.iloc[:train_idx].copy()
    df_val = df.iloc[train_idx:val_idx].copy()
    df_test = df.iloc[val_idx:].copy()
    
    y_test = (df_test["label"] != "normal").astype(int).values
    
    # --------------------------------------------------------------------------
    # 1. Evaluate Current Sequence Model
    # --------------------------------------------------------------------------
    checkpoint_path = "models/seq_ae.pt"
    extractor_path = "models/seq_ae_extractor.pkl"
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(extractor_path):
        print("ERROR: Checkpoint or extractor not found. Run training first.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    extractor = joblib.load(extractor_path)
    
    num_res = len(extractor.res_vocab)
    num_auth = len(extractor.auth_vocab)
    num_os = len(extractor.os_vocab)
    
    model_type = checkpoint.get("model_type", "bilstm")
    print(f"Loading sequence model checkpoint type: {model_type.upper()}")
    
    if model_type == "bilstm":
        hidden_dim = config.get("lstm_hidden_dim", 128)
        model = BiLSTMAutoencoder(num_res, num_auth, num_os, hidden_dim=hidden_dim)
    else:
        heads = config.get("transformer_heads", 4)
        layers = config.get("transformer_layers", 3)
        model = TransformerAutoencoder(num_res, num_auth, num_os, num_layers=layers, nhead=heads)
        
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    seq_len = config.get("seq_length", 32)
    val_dataset = AccessLogSequenceDataset(df_val, extractor, seq_len)
    test_dataset = AccessLogSequenceDataset(df_test, extractor, seq_len)
    
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Compute raw reconstruction scores
    print("Scoring validation set...")
    _, val_raw_scores = compute_timestep_scores(model, val_loader, device)
    
    print("Scoring test set...")
    _, test_raw_scores = compute_timestep_scores(model, test_loader, device)
    
    # Calibrate static normalization on validation set
    min_val, max_val = calibrate_min_max(val_raw_scores)
    test_scores_seq = apply_static_normalization(test_raw_scores, min_val, max_val)
    
    # Evaluate sequence model
    roc_seq, pr_seq, p1_seq = evaluate_predictions(y_test, test_scores_seq)
    
    # --------------------------------------------------------------------------
    # 2. Evaluate Phase 2 Baseline
    # --------------------------------------------------------------------------
    print("Loading Phase 2 baseline model for comparison...")
    baseline_payload = joblib.load("models/baseline.pkl")
    entity_profiles = joblib.load("models/entity_profiles.pkl")
    
    baseline_trainer = PipelineBaselineTrainer()
    _, global_profile = baseline_trainer.build_profiles(df_train)
    
    X_test_base, _ = baseline_trainer.extract_features(df_test, entity_profiles, global_profile)
    
    if baseline_payload["model_type"] == "StatProfile":
        test_scores_base = baseline_payload["estimator"].predict_score(X_test_base, df_test)
    elif baseline_payload["model_type"] == "IForest":
        test_scores_base = -baseline_payload["estimator"].score_samples(X_test_base)
    else:
        # OC-SVM
        X_test_base_scaled = baseline_payload["scaler"].transform(X_test_base)
        test_scores_base = -baseline_payload["estimator"].score_samples(X_test_base_scaled)
        
    # Evaluate baseline
    roc_base, pr_base, p1_base = evaluate_predictions(y_test, test_scores_base)
    
    # --------------------------------------------------------------------------
    # 3. Evaluate Ensemble
    # --------------------------------------------------------------------------
    w_seq = config.get("sequence_model_weight", 0.5)
    w_base = config.get("baseline_model_weight", 0.5)
    print(f"Computing ensembled scores (Sequence weight: {w_seq}, Baseline weight: {w_base})...")
    
    # Normalize baseline test scores to [0,1] to ensure equal scaling in ensemble
    base_min = np.min(test_scores_base)
    base_max = np.max(test_scores_base)
    denom = base_max - base_min if abs(base_max - base_min) > 1e-8 else 1.0
    test_scores_base_norm = (test_scores_base - base_min) / denom
    
    test_scores_ensemble = w_seq * test_scores_seq + w_base * test_scores_base_norm
    roc_ens, pr_ens, p1_ens = evaluate_predictions(y_test, test_scores_ensemble)
    
    # Output metrics summary table
    print("\n" + "="*60)
    print("COMPARATIVE TEST SET PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model / Architecture':<25} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'P@1%':<9}")
    print("-"*60)
    
    # Distinguish based on loaded type
    bilstm_metrics = (0.0, 0.0, 0.0)
    transformer_metrics = (0.0, 0.0, 0.0)
    
    if model_type == "bilstm":
        bilstm_metrics = (roc_seq, pr_seq, p1_seq)
        print(f"{'BiLSTM-AE (Current)':<25} | {roc_seq:.4f}    | {pr_seq:.4f}    | {p1_seq:.4f}")
        print(f"{'Transformer (Alt)':<25} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9}")
    else:
        transformer_metrics = (roc_seq, pr_seq, p1_seq)
        print(f"{'BiLSTM-AE (Primary)':<25} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9}")
        print(f"{'Transformer (Current)':<25} | {roc_seq:.4f}    | {pr_seq:.4f}    | {p1_seq:.4f}")
        
    print(f"{'Baseline (Phase 2)':<25} | {roc_base:.4f}    | {pr_base:.4f}    | {p1_base:.4f}")
    print(f"{'Ensemble (Seq + Base)':<25} | {roc_ens:.4f}    | {pr_ens:.4f}    | {p1_ens:.4f}")
    print("="*60 + "\n")
    
    # Save the ensembled scores for downstream processing (Phase 4, 5, 6)
    # Output schema needs to align with access_logs reference and classification inputs
    print("Generating ensembled anomaly scores for all events...")
    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
    all_dataset = AccessLogSequenceDataset(df_all, extractor, seq_len)
    all_loader = DataLoader(all_dataset, batch_size=128, shuffle=False)
    
    # Score all events with sequence model
    _, all_raw_scores_seq = compute_timestep_scores(model, all_loader, device)
    all_scores_seq = apply_static_normalization(all_raw_scores_seq, min_val, max_val)
    
    # Score all events with baseline model
    X_all_base, _ = baseline_trainer.extract_features(df_all, entity_profiles, global_profile)
    if baseline_payload["model_type"] == "StatProfile":
        all_scores_base = baseline_payload["estimator"].predict_score(X_all_base, df_all)
    elif baseline_payload["model_type"] == "IForest":
        all_scores_base = -baseline_payload["estimator"].score_samples(X_all_base)
    else:
        X_all_base_scaled = baseline_payload["scaler"].transform(X_all_base)
        all_scores_base = -baseline_payload["estimator"].score_samples(X_all_base_scaled)
        
    all_scores_base_norm = (all_scores_base - np.min(all_scores_base)) / (np.max(all_scores_base) - np.min(all_scores_base) + 1e-8)
    
    # Combine
    all_scores_ensemble = w_seq * all_scores_seq + w_base * all_scores_base_norm
    
    # Calculate rolling min-max normalization to handle concept drift as per README
    # We construct a helper df for rolling calculation
    df_roll = pd.DataFrame({
        "timestamp": df_all["timestamp"].values,
        "raw_score": all_scores_ensemble
    })
    df_roll_norm = apply_rolling_normalization(df_roll, window_size="30D")
    
    # Calibrate alert flagging threshold using validation ensemble scores
    val_len = len(df_val)
    # Extract validation slice from combined scores (chronological train then val then test)
    val_start = train_idx
    val_end = val_idx
    val_ensemble_scores = all_scores_ensemble[val_start:val_end]
    
    # We flag using config alert threshold percentile (e.g. 95th percentile, top 5% flags)
    alert_percentile = float(config.get("alert_threshold_percentile", 0.95)) * 100.0
    ensemble_threshold = np.percentile(val_ensemble_scores, alert_percentile)
    is_anomaly_flags = (all_scores_ensemble >= ensemble_threshold).astype(bool)
    
    df_out_scores = pd.DataFrame({
        "event_id": df_all["event_id"].values,
        "score_seq": all_scores_seq,
        "score_base": all_scores_base_norm,
        "score_ensemble": all_scores_ensemble,
        "score_rolling_ensemble": df_roll_norm["normalized_score"].values,
        "threshold": ensemble_threshold,
        "is_anomaly": is_anomaly_flags
    })
    
    scores_out_path = "data/processed/phase3_scores.csv"
    os.makedirs(os.path.dirname(scores_out_path), exist_ok=True)
    df_out_scores.to_csv(scores_out_path, index=False)
    print(f"Successfully exported Phase 3 ensembled anomaly scores to {scores_out_path}")
    
    # Update README
    update_readme(bilstm_metrics, transformer_metrics, (roc_base, pr_base, p1_base), model_type.upper())

if __name__ == "__main__":
    main()
