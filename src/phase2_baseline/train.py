"""
Phase 2: Baseline Profiling Model
Extracts per-entity habits, trains and compares anomaly detection baselines:
- Statistical Profile (KDE + distance scores)
- Isolation Forest
- One-Class SVM
Selects the best model and exports artifacts.
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

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import scipy.stats as stats

# Pre-defined list of global cities with lat/lon for distance verification
GLOBAL_CITIES = {
    "US": {"lat": 40.7128, "lon": -74.0060},  # New York
    "GB": {"lat": 51.5074, "lon": -0.1278},   # London
    "JP": {"lat": 35.6762, "lon": 139.6503},  # Tokyo
    "DE": {"lat": 50.1109, "lon": 8.6821},    # Frankfurt
    "AU": {"lat": -33.8688, "lon": 151.2093}, # Sydney
    "SG": {"lat": 1.3521, "lon": 103.8198},   # Singapore
    "IN": {"lat": 12.9716, "lon": 77.5946},   # Bangalore
    "BR": {"lat": -23.5505, "lon": -46.6333}, # São Paulo
    "ZA": {"lat": -33.9249, "lon": 18.4241},  # Cape Town
    "CA": {"lat": 43.6532, "lon": -79.3832}   # Toronto
}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def safe_json_loads(val: Any) -> Dict[str, Any]:
    """Safely parse a JSON string or return an empty dictionary."""
    if pd.isna(val):
        return {}
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val if isinstance(val, dict) else {}

class StatisticalProfiler:
    def __init__(self, entity_profiles: Dict[str, Any], global_profile: Dict[str, Any]):
        """Initialize the custom profiling baseline scorer."""
        self.entity_profiles = entity_profiles
        self.global_profile = global_profile
        self._fit_kdes()
        
    def _fit_kdes(self):
        """Fit KDE models for entity and global log hour densities."""
        self.kdes = {}
        for ent_id, prof in self.entity_profiles.items():
            if prof.get("num_events", 0) >= 10:
                hours = prof.get("hours", [])
                if len(hours) > 0 and np.var(hours) >= 1e-4:
                    try:
                        self.kdes[ent_id] = stats.gaussian_kde(hours, bw_method=0.2)
                    except Exception:
                        self.kdes[ent_id] = None
                else:
                    self.kdes[ent_id] = None
                        
        global_hours = self.global_profile.get("hours", [])
        if len(global_hours) > 0:
            try:
                self.global_kde = stats.gaussian_kde(global_hours, bw_method=0.2)
            except Exception:
                self.global_kde = None
        else:
            self.global_kde = None

    def __getstate__(self):
        """Custom pickling state exclusion for scipy.stats.gaussian_kde lambda functions."""
        state = self.__dict__.copy()
        state["kdes"] = {}
        state["global_kde"] = None
        return state

    def __setstate__(self, state):
        """Reconstruct KDE models upon unpickling."""
        self.__dict__.update(state)
        self._fit_kdes()
            
    def predict_score(self, df_features: pd.DataFrame, df_raw: pd.DataFrame) -> np.ndarray:
        """Calculate custom anomaly risk scores bounded in the range [0.0, 1.0]."""
        scores = []
        entity_ids = df_raw["entity_id"].values
        timestamps = pd.to_datetime(df_raw["timestamp"], format="ISO8601")
        hours = (timestamps.dt.hour + timestamps.dt.minute / 60.0).values
        
        geo_dists = df_features["geo_distance_from_centroid_km"].values
        res_novs = df_features["resource_novelty"].values
        dev_novs = df_features["device_fingerprint_hash_novelty"].values
        norm_durs = df_features["norm_session_duration"].values
        cold_starts = df_features["cold_start"].values
        
        for idx in range(len(df_raw)):
            ent_id = entity_ids[idx]
            hour = hours[idx]
            geo_dist = geo_dists[idx]
            res_nov = res_novs[idx]
            dev_nov = dev_novs[idx]
            norm_dur = norm_durs[idx]
            cold_start = cold_starts[idx]
            
            # 1. Geo distance anomaly scoring (threshold at 150 km)
            score_geo = min(1.0, geo_dist / 150.0)
            
            # 2. Session duration anomaly scoring
            score_dur = 1.0 - np.exp(-abs(norm_dur) / 2.0)
            
            # 3. Time anomaly scoring using KDE likelihood
            score_time = 0.5
            kde = self.kdes.get(ent_id, None) if cold_start == 0 else self.global_kde
            if kde is not None:
                try:
                    p = float(kde.evaluate([hour])[0])
                    # Higher probability density means lower anomaly score
                    score_time = max(0.0, min(1.0, 1.0 - (p / 0.25)))
                except Exception:
                    pass
            
            # Combine scores with weights (higher weights to structural novelties)
            score = (score_geo + score_dur + score_time + 2.5 * res_nov + 2.5 * dev_nov) / 8.0
            scores.append(score)
            
        return np.array(scores)

import sys
setattr(sys.modules["__main__"], "StatisticalProfiler", StatisticalProfiler)

class PipelineBaselineTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration parameters."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.seed = self.config.get("random_seed", 42)
        np.random.seed(self.seed)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw access log data and join with ground-truth labels."""
        print("Loading dataset...")
        df_logs = pd.read_csv("data/raw/logs.csv")
        df_labels = pd.read_csv("data/raw/labels.csv")
        df = pd.merge(df_logs, df_labels, on="event_id")
        
        # Sort chronologically
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        
        # Parse nested columns
        print("Parsing geo-location and device fingerprint records...")
        df["geo_location_parsed"] = df["geo_location"].apply(safe_json_loads)
        df["device_fingerprint_parsed"] = df["device_fingerprint"].apply(safe_json_loads)
        
        # Extract lat/lon for vector processing
        df["lat"] = df["geo_location_parsed"].apply(lambda x: x.get("lat", 0.0))
        df["lon"] = df["geo_location_parsed"].apply(lambda x: x.get("lon", 0.0))
        
        # Split into Train (70%), Val (15%), Test (15%)
        n = len(df)
        train_idx = int(0.70 * n)
        val_idx = int(0.85 * n)
        
        df_train = df.iloc[:train_idx].copy()
        df_val = df.iloc[train_idx:val_idx].copy()
        df_test = df.iloc[val_idx:].copy()
        
        return df_train, df_val, df_test
        
    def build_profiles(self, df_train: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract baseline habits for each entity in the training set."""
        print("Building per-entity and global population profiles...")
        df_train_normal = df_train[df_train["label"] == "normal"].copy()
        
        # Group normal activities
        entity_groups = df_train_normal.groupby("entity_id")
        entity_profiles = {}
        
        for entity_id, group in entity_groups:
            num_evs = len(group)
            lats = group["lat"].values
            lons = group["lon"].values
            mean_lat = float(np.mean(lats))
            mean_lon = float(np.mean(lons))
            
            # Radius calculation
            dists = [haversine_distance(mean_lat, mean_lon, lat, lon) for lat, lon in zip(lats, lons)]
            avg_radius = float(np.mean(dists)) if dists else 10.0
            
            # Resources (top 5 accessed)
            top_res = set(group["resource_accessed"].value_counts().head(5).index.tolist())
            
            # Unique devices
            macs = set()
            for dev in group["device_fingerprint_parsed"]:
                mac = dev.get("mac", "")
                if mac:
                    macs.add(mac)
                    
            # Session duration limits
            durs = group["session_duration"].values
            dur_mean = float(np.mean(durs))
            dur_std = float(np.std(durs)) if len(durs) > 1 else 10.0
            
            # Time of day
            hours = (group["timestamp"].dt.hour + group["timestamp"].dt.minute / 60.0).values
            
            entity_profiles[entity_id] = {
                "num_events": num_evs,
                "home_lat_lon": (mean_lat, mean_lon),
                "avg_radius": avg_radius,
                "typical_resources": top_res,
                "typical_devices": macs,
                "session_duration_mean": dur_mean,
                "session_duration_std": dur_std,
                "hours": hours.tolist()
            }
            
        # Global Population Profile (Fallback)
        g_lats = df_train_normal["lat"].values
        g_lons = df_train_normal["lon"].values
        global_lat = float(np.mean(g_lats))
        global_lon = float(np.mean(g_lons))
        global_res = set(df_train_normal["resource_accessed"].value_counts().head(10).index.tolist())
        global_macs = set()
        for dev in df_train_normal["device_fingerprint_parsed"]:
            mac = dev.get("mac", "")
            if mac:
                global_macs.add(mac)
        global_dur_mean = float(np.mean(df_train_normal["session_duration"].values))
        global_dur_std = float(np.std(df_train_normal["session_duration"].values))
        global_hours = (df_train_normal["timestamp"].dt.hour + df_train_normal["timestamp"].dt.minute / 60.0).tolist()
        
        global_profile = {
            "num_events": len(df_train_normal),
            "home_lat_lon": (global_lat, global_lon),
            "avg_radius": 1000.0,
            "typical_resources": global_res,
            "typical_devices": global_macs,
            "session_duration_mean": global_dur_mean,
            "session_duration_std": global_dur_std,
            "hours": global_hours
        }
        
        return entity_profiles, global_profile

    def extract_features(self, df: pd.DataFrame, entity_profiles: Dict[str, Any], global_profile: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """Engineer per-entity features for ML models."""
        features = []
        audit_trail = []
        
        entity_ids = df["entity_id"].values
        timestamps = pd.to_datetime(df["timestamp"], format="ISO8601")
        hours = (timestamps.dt.hour + timestamps.dt.minute / 60.0).values
        days_of_week = timestamps.dt.dayofweek.values
        is_weekends = (days_of_week >= 5).astype(int)
        durations = df["session_duration"].values
        lats = df["lat"].values
        lons = df["lon"].values
        resources = df["resource_accessed"].values
        auth_methods = df["auth_method"].values
        
        macs = [dev.get("mac", "") for dev in df["device_fingerprint_parsed"]]
        
        for idx in range(len(df)):
            entity_id = entity_ids[idx]
            hour = hours[idx]
            dow = days_of_week[idx]
            is_we = is_weekends[idx]
            dur = durations[idx]
            lat = lats[idx]
            lon = lons[idx]
            res = resources[idx]
            auth = auth_methods[idx]
            mac = macs[idx]
            
            # Fetch profile with Cold-Start policy
            profile = entity_profiles.get(entity_id, None)
            if profile is not None and profile["num_events"] >= 10:
                cold_start = 0
                audit_trail.append("cold_start=False")
            else:
                profile = global_profile
                cold_start = 1
                audit_trail.append("cold_start=True")
                
            # Compute distances & novelties
            p_lat, p_lon = profile["home_lat_lon"]
            geo_dist = haversine_distance(p_lat, p_lon, lat, lon)
            res_novel = 1 if res not in profile["typical_resources"] else 0
            dev_novel = 1 if mac not in profile["typical_devices"] else 0
            
            # Auth One-Hot
            auth_pw = 1 if auth == "password" else 0
            auth_tok = 1 if auth == "token" else 0
            auth_cert = 1 if auth == "certificate" else 0
            auth_bio = 1 if auth == "biometric" else 0
            
            # Session duration standard scaling
            mean_dur = profile["session_duration_mean"]
            std_dur = profile["session_duration_std"]
            norm_dur = (dur - mean_dur) / (std_dur + 1e-5)
            
            features.append([
                hour,
                dow,
                is_we,
                dur,
                norm_dur,
                geo_dist,
                res_novel,
                dev_novel,
                auth_pw,
                auth_tok,
                auth_cert,
                auth_bio,
                cold_start
            ])
            
        feature_cols = [
            "hour_of_day", "day_of_week", "is_weekend", "session_duration", "norm_session_duration",
            "geo_distance_from_centroid_km", "resource_novelty", "device_fingerprint_hash_novelty",
            "auth_password", "auth_token", "auth_certificate", "auth_biometric", "cold_start"
        ]
        return pd.DataFrame(features, columns=feature_cols), audit_trail

    def evaluate_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, float]:
        """Calculate ROC-AUC, PR-AUC, and False Positive Rate at top 1% alert budget threshold."""
        roc_auc = float(roc_auc_score(y_true, y_scores))
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = float(auc(recall, precision))
        
        # Calculate FPR at top-1% alert budget
        threshold = np.percentile(y_scores, 99.0)
        y_pred = (y_scores >= threshold).astype(int)
        
        # False Positives / Total Actual Negatives
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fpr_at_1 = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        
        return roc_auc, pr_auc, fpr_at_1

    def run(self):
        """Train and evaluate the baseline models."""
        os.makedirs("models", exist_ok=True)
        
        df_train, df_val, df_test = self.load_data()
        
        # Build profiles from training normal rows
        entity_profiles, global_profile = self.build_profiles(df_train)
        
        # Extract features for train/val/test
        print("Extracting features for training set...")
        X_train, _ = self.extract_features(df_train, entity_profiles, global_profile)
        
        print("Extracting features for validation set...")
        X_val, val_audit = self.extract_features(df_val, entity_profiles, global_profile)
        
        print("Extracting features for test set...")
        X_test, test_audit = self.extract_features(df_test, entity_profiles, global_profile)
        
        # Separate normal training rows for semi-supervised fitting
        normal_mask = (df_train["label"] == "normal").values
        X_train_normal = X_train.iloc[normal_mask].copy()
        
        # Target vectors for validation and test (1 = anomaly, 0 = normal)
        y_val = (df_val["label"] != "normal").astype(int).values
        y_test = (df_test["label"] != "normal").astype(int).values
        
        results = {}
        
        # ==============================================================================
        # Model 1: Statistical Profile
        # ==============================================================================
        print("Fitting Statistical Profile Model...")
        stat_model = StatisticalProfiler(entity_profiles, global_profile)
        val_scores_stat = stat_model.predict_score(X_val, df_val)
        test_scores_stat = stat_model.predict_score(X_test, df_test)
        results["StatProfile"] = self.evaluate_metrics(y_val, val_scores_stat)
        
        # ==============================================================================
        # Model 2: Isolation Forest
        # ==============================================================================
        print("Fitting Isolation Forest Model...")
        if_model = IsolationForest(n_estimators=100, contamination=0.02, random_state=self.seed)
        # Train on normal events
        if_model.fit(X_train_normal)
        # Decision function outputs lower values for outliers. We invert to get high = anomaly.
        val_scores_if = -if_model.score_samples(X_val)
        test_scores_if = -if_model.score_samples(X_test)
        results["IForest"] = self.evaluate_metrics(y_val, val_scores_if)
        
        # ==============================================================================
        # Model 3: One-Class SVM
        # ==============================================================================
        print("Fitting One-Class SVM Model...")
        scaler = StandardScaler()
        X_train_normal_scaled = scaler.fit_transform(X_train_normal)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Keep nu small since training set is mostly normal (clean)
        oc_svm = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
        # Downsample for SVM fitting to avoid quadratic O(N^2) complexity bottlenecks
        if len(X_train_normal_scaled) > 20000:
            rng = np.random.default_rng(self.seed)
            sample_idx = rng.choice(len(X_train_normal_scaled), 20000, replace=False)
            oc_svm.fit(X_train_normal_scaled[sample_idx])
        else:
            oc_svm.fit(X_train_normal_scaled)
        
        val_scores_svm = -oc_svm.score_samples(X_val_scaled)
        test_scores_svm = -oc_svm.score_samples(X_test_scaled)
        results["OC-SVM"] = self.evaluate_metrics(y_val, val_scores_svm)
        
        # ==============================================================================
        # Comparison & Artifact Export
        # ==============================================================================
        print("\n" + "="*50)
        print("MODEL PERFORMANCE COMPARISON (VALIDATION SET)")
        print("="*50)
        print(f"{'Model':<15} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'FPR@1%':<9}")
        print("-"*50)
        for model_name, metrics in results.items():
            print(f"{model_name:<15} | {metrics[0]:.4f}    | {metrics[1]:.4f}    | {metrics[2]:.4f}")
        print("="*50 + "\n")
        
        # Choose the best model based on validation ROC-AUC
        best_model_name = max(results, key=lambda k: results[k][0])
        print(f"Best model selected: {best_model_name}")
        
        # Assign best outputs
        if best_model_name == "StatProfile":
            best_model = stat_model
            test_metrics = self.evaluate_metrics(y_test, test_scores_stat)
            # Wrap best estimator structure
            export_payload = {
                "model_type": "StatProfile",
                "estimator": stat_model,
                "scaler": None
            }
        elif best_model_name == "IForest":
            best_model = if_model
            test_metrics = self.evaluate_metrics(y_test, test_scores_if)
            export_payload = {
                "model_type": "IForest",
                "estimator": if_model,
                "scaler": None
            }
        else:
            best_model = oc_svm
            test_metrics = self.evaluate_metrics(y_test, test_scores_svm)
            export_payload = {
                "model_type": "OC-SVM",
                "estimator": oc_svm,
                "scaler": scaler
            }
            
        print(f"Test Set Metrics for Best Model ({best_model_name}):")
        print(f"  - ROC-AUC: {test_metrics[0]:.4f}")
        print(f"  - PR-AUC:  {test_metrics[1]:.4f}")
        print(f"  - FPR@1%:  {test_metrics[2]:.4f}\n")
        
        # Save Profiles and Best Scorer to models/
        joblib.dump(export_payload, "models/baseline.pkl")
        joblib.dump(entity_profiles, "models/entity_profiles.pkl")
        print("Model artifacts successfully saved to models/ directory.")

        # Recombine dataset to score all events for downstream DB loading and models
        print("Generating baseline anomaly scores for all events...")
        df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
        X_all, _ = self.extract_features(df_all, entity_profiles, global_profile)

        if best_model_name == "StatProfile":
            all_scores = stat_model.predict_score(X_all, df_all)
            threshold = np.percentile(val_scores_stat, 99.0)
        elif best_model_name == "IForest":
            all_scores = -if_model.score_samples(X_all)
            threshold = np.percentile(val_scores_if, 99.0)
        else:
            X_all_scaled = scaler.transform(X_all)
            all_scores = -oc_svm.score_samples(X_all_scaled)
            threshold = np.percentile(val_scores_svm, 99.0)

        is_anomaly_flags = (all_scores >= threshold).astype(bool)

        df_scores = pd.DataFrame({
            "event_id": df_all["event_id"].values,
            "score": all_scores,
            "threshold": threshold,
            "is_anomaly": is_anomaly_flags,
            "cold_start": X_all["cold_start"].values.astype(bool)
        })

        os.makedirs("data/processed", exist_ok=True)
        scores_csv_path = "data/processed/phase2_scores.csv"
        df_scores.to_csv(scores_csv_path, index=False)
        print(f"Baseline anomaly scores for all events successfully saved to {scores_csv_path}")
        
        # Update Phase 2 README with the actual metrics table
        self.update_readme(results, best_model_name, test_metrics)
        
    def update_readme(self, results: Dict[str, Tuple[float, float, float]], best_model_name: str, test_metrics: Tuple[float, float, float]):
        """Dynamically update Phase 2 README.md file with the actual validation metrics."""
        readme_path = "src/phase2_baseline/README.md"
        
        markdown_content = f"""# Phase 2 — Baseline Profiling Model

## Purpose
Establish a per-entity representation of "normal" so any large deviation
produces an initial anomaly score. Serves as a competitive baseline and a
warm-start for the sequence model.

## Models Evaluated
- Statistical Profile (KDE + distance scores)
- Isolation Forest
- One-Class SVM

## Best Model & Metrics
The model performance was evaluated using a chronological split (70% train / 15% validation / 15% test). The metrics are listed below:

| Model | ROC-AUC | PR-AUC | FPR@1% |
|-------|---------|--------|--------|
| StatProfile | {results["StatProfile"][0]:.4f} | {results["StatProfile"][1]:.4f} | {results["StatProfile"][2]:.4f} |
| IForest | {results["IForest"][0]:.4f} | {results["IForest"][1]:.4f} | {results["IForest"][2]:.4f} |
| OC-SVM | {results["OC-SVM"][0]:.4f} | {results["OC-SVM"][1]:.4f} | {results["OC-SVM"][2]:.4f} |

**Selected Best Model:** `{best_model_name}`

**Test Set Evaluation Metrics (Best Model):**
- **ROC-AUC:** {test_metrics[0]:.4f}
- **PR-AUC:** {test_metrics[1]:.4f}
- **FPR@1%:** {test_metrics[2]:.4f}

## Cold-Start Policy
Entities with <10 events in the training data use a fallback global population profile, logging a `"cold_start=True"` audit trail flag.

## Artifacts
- `models/baseline.pkl`
- `models/entity_profiles.pkl`
"""
        with open(readme_path, "w") as f:
            f.write(markdown_content)
        print(f"Updated {readme_path} with final validation performance metrics.")

if __name__ == "__main__":
    trainer = PipelineBaselineTrainer()
    trainer.run()
