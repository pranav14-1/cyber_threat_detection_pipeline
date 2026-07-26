"""
Phase 4: Threat Signature Classification
Builds a multi-class LightGBM/GBDT classifier with deterministic rule-based assist to categorize
flagged behavioral anomalies into specific MITRE ATT&CK threat tactics.
"""

import os
import json
import joblib
import yaml
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime

# ML & Evaluation imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch sequence scoring breakdown
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.phase3_sequence_model.dataset import SequenceFeatureExtractor, AccessLogSequenceDataset
from src.phase3_sequence_model.model import BiLSTMAutoencoder, TransformerAutoencoder
from src.phase2_baseline.train import PipelineBaselineTrainer, safe_json_loads, haversine_distance

# XGBoost & Scikit-Learn imports
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


class ThreatClassifier:
    """
    Inference & Wrapper class for the Phase 4 Threat Classifier model.
    Combines machine learning multi-class probabilities with deterministic rule-based overrides.
    """
    def __init__(self,
                 model: Any,
                 label_encoder: LabelEncoder,
                 feature_names: List[str],
                 entity_profiles: Dict[str, Any],
                 global_profile: Dict[str, Any]):
        self.model = model
        self.label_encoder = label_encoder
        self.feature_names = feature_names
        self.entity_profiles = entity_profiles
        self.global_profile = global_profile

    def check_deterministic_rules(self, geo_velocity_kmh: float, failed_auth_count_5min: int, geo_distance_km: float = 0.0) -> Tuple[str, float, str]:
        """
        Check deterministic rule overrides:
        1. geo_velocity_kmh > 900 AND geo_distance_km > 500 -> impossible_travel
        2. failed_auth_count_5min > 30 -> brute_force
        """
        if geo_velocity_kmh > 900.0 and geo_distance_km > 500.0:
            return "impossible_travel", 1.0, "Rule Override: geo_velocity_kmh > 900 km/h AND geo_distance > 500 km"
        if failed_auth_count_5min > 30:
            return "brute_force", 1.0, "Rule Override: failed_auth_count_5min > 30"
        return None, 0.0, None

    def predict(self, feature_dict_or_df: Any) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict attack type, confidence score, and top contributing features.
        
        Args:
            feature_dict_or_df: Dictionary or DataFrame row containing input event features.
        Returns:
            Tuple of (attack_type, confidence, top_contributing_features)
        """
        if isinstance(feature_dict_or_df, dict):
            df_feat = pd.DataFrame([feature_dict_or_df])
        elif isinstance(feature_dict_or_df, pd.Series):
            df_feat = pd.DataFrame([feature_dict_or_df.to_dict()])
        else:
            df_feat = feature_dict_or_df.copy()

        # Extract deterministic rule triggers if available
        geo_vel = float(df_feat["geo_velocity_kmh"].iloc[0]) if "geo_velocity_kmh" in df_feat else 0.0
        geo_dist = float(df_feat["geo_distance_prev_km"].iloc[0]) if "geo_distance_prev_km" in df_feat else 0.0
        failed_auth = int(df_feat["failed_auth_count_5min"].iloc[0]) if "failed_auth_count_5min" in df_feat else 0

        override_class, confidence, rule_reason = self.check_deterministic_rules(geo_vel, failed_auth, geo_dist)
        if override_class is not None:
            trigger_feature = "geo_velocity_kmh" if override_class == "impossible_travel" else "failed_auth_count_5min"
            top_features = [(trigger_feature, 1.0), ("rule_override_trigger", 1.0)]
            return override_class, confidence, top_features

        # ML Model Prediction
        X = df_feat[self.feature_names].values
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        attack_type = str(self.label_encoder.inverse_transform([pred_idx])[0])

        # Extract top feature contributions
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            row_vals = np.abs(X[0])
            scores = row_vals * importances
            top_indices = np.argsort(scores)[::-1][:3]
            top_features = [(self.feature_names[i], float(scores[i])) for i in top_indices]
        else:
            top_features = [(self.feature_names[0], 0.33), (self.feature_names[1], 0.33), (self.feature_names[2], 0.33)]

        return attack_type, confidence, top_features


class PipelineThreatClassifierTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get("random_seed", 42)
        np.random.seed(self.seed)

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load logs, ground-truth labels, and Phase 3 ensembled anomaly scores."""
        print("Loading datasets for Phase 4 classification...")
        df_logs = pd.read_csv("data/raw/logs.csv")
        df_labels = pd.read_csv("data/raw/labels.csv")
        df_p3 = pd.read_csv("data/processed/phase3_scores.csv")

        # Merge on event_id (left join to preserve all log events)
        df = pd.merge(df_logs, df_labels, on="event_id")
        df = pd.merge(df, df_p3[["event_id", "score_ensemble", "threshold", "is_anomaly"]], on="event_id", how="left")
        df["score_ensemble"] = df["score_ensemble"].fillna(0.5)
        df["threshold"] = df["threshold"].fillna(0.8)
        df["is_anomaly"] = df["is_anomaly"].fillna(0)

        # Parse timestamps and sort chronologically
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df = df.sort_values(by="timestamp").reset_index(drop=True)

        # Parse nested records
        df["geo_location_parsed"] = df["geo_location"].apply(safe_json_loads)
        df["device_fingerprint_parsed"] = df["device_fingerprint"].apply(safe_json_loads)
        df["lat"] = df["geo_location_parsed"].apply(lambda x: x.get("lat", 0.0))
        df["lon"] = df["geo_location_parsed"].apply(lambda x: x.get("lon", 0.0))
        df["mac"] = df["device_fingerprint_parsed"].apply(lambda x: x.get("mac", ""))
        df["device_os"] = df["device_fingerprint_parsed"].apply(lambda x: x.get("os", ""))

        return df, df_logs, df_labels

    def compute_sliding_window_features(self, df: pd.DataFrame, entity_profiles: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute sliding-window aggregate features:
        - failed_auth_count_5min
        - distinct_ip_count_1h
        - new_resource_count_1h
        - geo_velocity_kmh
        """
        print("Engineering sliding-window temporal aggregates...")
        failed_auth_5m = np.zeros(len(df), dtype=int)
        distinct_ip_1h = np.zeros(len(df), dtype=int)
        new_resource_1h = np.zeros(len(df), dtype=int)
        geo_velocity = np.zeros(len(df), dtype=float)
        geo_distance_prev = np.zeros(len(df), dtype=float)

        entity_groups = df.groupby("entity_id")

        for ent_id, group in entity_groups:
            indices = group.index.values
            ts_sec = group["timestamp"].view('int64').values // 10**9
            ips = group["source_ip"].values
            resources = group["resource_accessed"].values
            lats = group["lat"].values
            lons = group["lon"].values
            durs = group["session_duration"].values
            auths = group["auth_method"].values

            profile = entity_profiles.get(ent_id, {})
            typical_res = set(profile.get("typical_resources", []))

            # 1. Failed auths mask (session_duration == 0 or failed password login)
            is_failed = ((durs == 0) | (auths == "password")).astype(int)

            n_ev = len(group)
            for i in range(n_ev):
                curr_idx = indices[i]
                t_curr = ts_sec[i]

                # 5-min window (300 seconds)
                t_5m_start = t_curr - 300
                idx_5m_start = np.searchsorted(ts_sec[:i+1], t_5m_start, side='left')
                failed_auth_5m[curr_idx] = int(np.sum(is_failed[idx_5m_start:i+1]))

                # 1-hour window (3600 seconds)
                t_1h_start = t_curr - 3600
                idx_1h_start = np.searchsorted(ts_sec[:i+1], t_1h_start, side='left')

                # Distinct IPs in 1h
                distinct_ip_1h[curr_idx] = len(set(ips[idx_1h_start:i+1]))

                # Novel resources accessed in 1h
                window_res = resources[idx_1h_start:i+1]
                novel_in_win = set(res for res in window_res if res not in typical_res)
                new_resource_1h[curr_idx] = len(novel_in_win)

                # Geo-velocity from previous event
                if i > 0:
                    dt_hours = (t_curr - ts_sec[i-1]) / 3600.0
                    dist_km = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
                    geo_velocity[curr_idx] = dist_km / dt_hours if dt_hours > 0.0 else 0.0
                    geo_distance_prev[curr_idx] = dist_km
                else:
                    geo_velocity[curr_idx] = 0.0
                    geo_distance_prev[curr_idx] = 0.0

        df["failed_auth_count_5min"] = failed_auth_5m
        df["distinct_ip_count_1h"] = distinct_ip_1h
        df["new_resource_count_1h"] = new_resource_1h
        df["geo_velocity_kmh"] = geo_velocity
        df["geo_distance_prev_km"] = geo_distance_prev

        return df

    def compute_phase3_reconstruction_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute itemized reconstruction errors per input feature from Phase 3 sequence AE."""
        print("Extracting Phase 3 sequence model reconstruction error breakdown...")
        checkpoint_path = "models/seq_ae.pt"
        extractor_path = "models/seq_ae_extractor.pkl"

        if not os.path.exists(checkpoint_path) or not os.path.exists(extractor_path):
            print("WARNING: Sequence model checkpoint not found. Filling reconstruction breakdowns with 0.0.")
            for col in ["rec_err_duration", "rec_err_hour_sin", "rec_err_hour_cos", "rec_err_geo_dist", "rec_err_resource", "rec_err_auth", "rec_err_os"]:
                df[col] = 0.0
            return df

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        extractor = joblib.load(extractor_path)

        num_res = len(extractor.res_vocab)
        num_auth = len(extractor.auth_vocab)
        num_os = len(extractor.os_vocab)

        model_type = checkpoint.get("model_type", "bilstm")
        if model_type == "bilstm":
            hidden_dim = self.config.get("lstm_hidden_dim", 128)
            seq_model = BiLSTMAutoencoder(num_res, num_auth, num_os, hidden_dim=hidden_dim)
        else:
            heads = self.config.get("transformer_heads", 4)
            layers = self.config.get("transformer_layers", 3)
            seq_model = TransformerAutoencoder(num_res, num_auth, num_os, num_layers=layers, nhead=heads)

        seq_model.load_state_dict(checkpoint["model_state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_model = seq_model.to(device)
        seq_model.eval()

        seq_len = self.config.get("seq_length", 32)
        dataset = AccessLogSequenceDataset(df, extractor, seq_len)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        criterion_mse_none = nn.MSELoss(reduction="none")
        criterion_ce_none = nn.CrossEntropyLoss(reduction="none", ignore_index=0)

        event_error_map = {}
        with torch.no_grad():
            for cat_seq, cont_seq, ev_ids, _ in loader:
                cat_seq = cat_seq.to(device)
                cont_seq = cont_seq.to(device)

                outputs = seq_model(cat_seq, cont_seq)
                pred_cont, pred_res, pred_auth, pred_os = outputs

                idx = cat_seq.shape[1] - 1  # final timestep

                loss_cont_all = criterion_mse_none(pred_cont[:, idx, :], cont_seq[:, idx, :]).cpu().numpy()
                loss_res = criterion_ce_none(pred_res[:, idx, :], cat_seq[:, idx, 0]).cpu().numpy()
                loss_auth = criterion_ce_none(pred_auth[:, idx, :], cat_seq[:, idx, 1]).cpu().numpy()
                loss_os = criterion_ce_none(pred_os[:, idx, :], cat_seq[:, idx, 2]).cpu().numpy()

                for b in range(len(ev_ids)):
                    event_error_map[ev_ids[b]] = (
                        float(loss_cont_all[b, 0]),
                        float(loss_cont_all[b, 1]),
                        float(loss_cont_all[b, 2]),
                        float(loss_cont_all[b, 3]),
                        float(loss_res[b]),
                        float(loss_auth[b]),
                        float(loss_os[b])
                    )

        err_cols = ["rec_err_duration", "rec_err_hour_sin", "rec_err_hour_cos", "rec_err_geo_dist", "rec_err_resource", "rec_err_auth", "rec_err_os"]
        err_data = [event_error_map.get(ev_id, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) for ev_id in df["event_id"].values]
        df_errs = pd.DataFrame(err_data, columns=err_cols, index=df.index)

        for col in err_cols:
            df[col] = df_errs[col]

        return df

    def run(self):
        """Execute Phase 4 pipeline training, cross-validation, plotting, and model export."""
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports/figures", exist_ok=True)

        df, df_logs, df_labels = self.load_and_prepare_data()

        # Load entity profiles and Phase 2 baseline trainer
        entity_profiles = joblib.load("models/entity_profiles.pkl") if os.path.exists("models/entity_profiles.pkl") else {}
        baseline_trainer = PipelineBaselineTrainer()
        if "_global" in entity_profiles:
            global_profile = entity_profiles["_global"]
        else:
            _, global_profile = baseline_trainer.build_profiles(df)

        # 1. Phase 2 handcrafted features
        print("Extracting Phase 2 handcrafted features...")
        X_base, _ = baseline_trainer.extract_features(df, entity_profiles, global_profile)

        # Merge base features with main df without creating duplicate column names
        for col in X_base.columns:
            if col != "session_duration":
                df[col] = X_base[col]

        # 2. Sliding window aggregates
        df = self.compute_sliding_window_features(df, entity_profiles)

        # 3. Phase 3 reconstruction breakdown
        df = self.compute_phase3_reconstruction_breakdown(df)

        # Filter flagged anomalies subset (top ~5% flagged by Phase 3)
        print("Filtering flagged anomalies subset (is_anomaly == True)...")
        is_anom_mask = (df["is_anomaly"].astype(str).str.lower() == "true") | (df["score_ensemble"] >= df["threshold"])
        df_flagged = df[is_anom_mask].copy().reset_index(drop=True)
        print(f"Total flagged anomaly events for classification: {len(df_flagged)}")

        # Assign ground-truth multi-class labels (normal events flagged by detector map to 'unknown')
        target_labels = df_flagged["label"].apply(lambda x: "unknown" if x == "normal" else x).values
        df_flagged["target_class"] = target_labels

        feature_cols = [
            # Phase 2 features & cold start / decay markers
            "hour_of_day", "day_of_week", "is_weekend", "session_duration", "norm_session_duration",
            "geo_distance_from_centroid_km", "resource_novelty", "device_fingerprint_hash_novelty",
            "auth_password", "auth_token", "auth_certificate", "auth_biometric", "cold_start",
            "is_cold_start", "profile_decay_factor",
            # Phase 3 error breakdown
            "rec_err_duration", "rec_err_hour_sin", "rec_err_hour_cos", "rec_err_geo_dist",
            "rec_err_resource", "rec_err_auth", "rec_err_os",
            # Sliding window aggregates & distance
            "failed_auth_count_5min", "distinct_ip_count_1h", "new_resource_count_1h", "geo_velocity_kmh", "geo_distance_prev_km"
        ]

        X = df_flagged[feature_cols].values
        le = LabelEncoder()
        y = le.fit_transform(df_flagged["target_class"].values)
        class_names = le.classes_.tolist()

        print(f"\nMulti-Class Target Taxonomy ({len(class_names)} classes): {class_names}")

        # Perform 5-fold Stratified CV
        print("\n" + "="*60)
        print("PERFORMING 5-FOLD STRATIFIED CROSS-VALIDATION")
        print("="*60)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof_preds = np.zeros(len(df_flagged), dtype=int)
        rule_override_count = 0

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            sw_tr = compute_sample_weight('balanced', y_tr)
            clf = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                max_delta_step=1,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            y_pred_fold = clf.predict(X_va)

            # Apply deterministic rule-based assist overrides on val set
            df_val_sub = df_flagged.iloc[val_idx]
            geo_vels = df_val_sub["geo_velocity_kmh"].values
            geo_dists = df_val_sub["geo_distance_prev_km"].values
            failed_auths = df_val_sub["failed_auth_count_5min"].values

            imp_travel_idx = le.transform(["impossible_travel"])[0] if "impossible_travel" in class_names else -1
            brute_force_idx = le.transform(["brute_force"])[0] if "brute_force" in class_names else -1

            for k in range(len(val_idx)):
                if geo_vels[k] > 900.0 and geo_dists[k] > 500.0 and imp_travel_idx != -1:
                    y_pred_fold[k] = imp_travel_idx
                    rule_override_count += 1
                elif failed_auths[k] > 30 and brute_force_idx != -1:
                    y_pred_fold[k] = brute_force_idx
                    rule_override_count += 1

            oof_preds[val_idx] = y_pred_fold

        print(f"Total Deterministic Rule Overrides Executed: {rule_override_count}")

        # Compute Out-Of-Fold Evaluation Metrics
        report_dict = classification_report(y, oof_preds, target_names=class_names, output_dict=True)
        print("\n" + classification_report(y, oof_preds, target_names=class_names))

        # Final model fit on entire flagged subset
        print("Training final XGBoost threat classifier on all flagged anomaly data...")
        sw_full = compute_sample_weight('balanced', y)
        final_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            max_delta_step=1,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        final_model.fit(X, y, sample_weight=sw_full)

        # Export predictions CSV
        print("Exporting Phase 4 predictions to data/processed/phase4_predictions.csv...")
        df_flagged["predicted_attack_type"] = le.inverse_transform(oof_preds)

        rule_reasons = []
        for i in range(len(df_flagged)):
            gv = df_flagged["geo_velocity_kmh"].iloc[i]
            gd = df_flagged["geo_distance_prev_km"].iloc[i]
            fa = df_flagged["failed_auth_count_5min"].iloc[i]
            if gv > 900.0 and gd > 500.0:
                rule_reasons.append("Rule Override: geo_velocity_kmh > 900 km/h AND geo_distance > 500 km")
            elif fa > 30:
                rule_reasons.append("Rule Override: failed_auth_count_5min > 30")
            else:
                rule_reasons.append("XGBoost ML Classifier")
        df_flagged["prediction_source"] = rule_reasons

        pred_cols = ["event_id", "timestamp", "entity_id", "label", "target_class", "predicted_attack_type", "prediction_source"]
        df_flagged[pred_cols].to_csv("data/processed/phase4_predictions.csv", index=False)

        # Plot & Save Confusion Matrix
        cm = confusion_matrix(y, oof_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Phase 4: Threat Classification Confusion Matrix (5-Fold CV OOF)')
        plt.xlabel('Predicted Threat Class')
        plt.ylabel('Ground Truth Threat Class')
        plt.tight_layout()

        cm_path = "reports/figures/confusion_matrix.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix figure to {cm_path}")

        # Package classifier instance & export artifact
        classifier_payload = ThreatClassifier(
            model=final_model,
            label_encoder=le,
            feature_names=feature_cols,
            entity_profiles=entity_profiles,
            global_profile=global_profile
        )

        joblib.dump(classifier_payload, "models/classifier.pkl")
        print("Exported trained classifier artifact to models/classifier.pkl")

        # Update Phase 4 README with metrics table
        self.update_readme(report_dict, class_names)

    def update_readme(self, report_dict: Dict[str, Any], class_names: List[str]):
        """Dynamically update Phase 4 README.md with final cross-validation metrics table."""
        readme_path = "src/phase4_classifier/README.md"

        table_rows = []
        for cls_name in class_names:
            if cls_name in report_dict:
                prec = report_dict[cls_name]["precision"]
                rec = report_dict[cls_name]["recall"]
                f1 = report_dict[cls_name]["f1-score"]
                table_rows.append(f"| `{cls_name}` | {prec:.4f} | {rec:.4f} | {f1:.4f} |")

        table_str = "\n".join(table_rows)
        macro_f1 = report_dict.get("macro avg", {}).get("f1-score", 0.0)

        markdown_content = f"""# Phase 4 — Anomaly Classification

## Purpose
Move from "anomalous / not anomalous" to naming the attack category so security analysts
know how to respond (brute force ≠ insider drift ≠ impossible travel).

## Model
XGBoost Multi-Class ({len(class_names)} classes: 7 attack types + `unknown`), balanced sample weights, `max_delta_step=1`, `subsample=0.8`, `learning_rate=0.08`, `n_estimators=150`,
5-fold stratified CV with deterministic rule assist.

## Deterministic Overrides
| Rule | Trigger | Assigned Class |
|------|---------|----------------|
| `geo_velocity_kmh > 900 AND geo_distance_prev_km > 500` | strictly satisfied | `impossible_travel` |
| `failed_auth_count_5min > 30` | strictly satisfied | `brute_force` |

## Results (5-Fold Stratified Cross-Validation)
| Attack Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
{table_str}

**Macro F1-Score:** `{macro_f1:.4f}`

## Artifacts
- `models/classifier.pkl`
- `reports/figures/confusion_matrix.png`
- `data/processed/phase4_predictions.csv`
"""
        with open(readme_path, "w") as f:
            f.write(markdown_content)
        print(f"Updated {readme_path} with final classification results table.")


if __name__ == "__main__":
    trainer = PipelineThreatClassifierTrainer()
    trainer.run()
