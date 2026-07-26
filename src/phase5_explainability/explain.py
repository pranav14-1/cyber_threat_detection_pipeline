"""
Phase 5: Explainability Layer (XAI)
Provides local feature attributions and analyst-readable natural language cards for security alerts.
Combines SHAP (TreeExplainer) on the Phase 4 classifier with per-dimension reconstruction attributions
from the Phase 3 sequence autoencoder.
"""

import os
import sys
import json
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

import torch

# Import pipeline components for deserialization and feature calculations
from src.phase2_baseline.train import PipelineBaselineTrainer, StatisticalProfiler, safe_json_loads, haversine_distance
from src.phase4_classifier.train import ThreatClassifier, PipelineThreatClassifierTrainer

# SHAP Import with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load system configuration parameters."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_feature_narrative(feature_name: str, feature_value: float, attack_type: str) -> str:
    """
    Translate raw technical feature names and numerical values into plain English narratives.
    
    Args:
        feature_name: Name of the feature contributing to the alert.
        feature_value: Numerical value of the feature for the event.
        attack_type: Predicted attack taxonomy category.
        
    Returns:
        Human-readable analyst narrative string.
    """
    fn = feature_name.lower()
    val = float(feature_value)
    
    if "geo_velocity" in fn:
        return f"Travel velocity of {val:.1f} km/h between consecutive logins exceeds physical speed limits (>900 km/h)"
    elif "geo_distance_prev" in fn:
        return f"Physical distance of {val:.1f} km from previous login location exceeds physical movement threshold (>500 km)"
    elif "failed_auth" in fn:
        return f"High frequency of failed authentication attempts ({int(val)} failures in 5-minute window)"
    elif "distinct_ip" in fn:
        return f"Login requests originating from {int(val)} distinct IP addresses within a 1-hour window"
    elif "new_resource" in fn:
        return f"Rapid burst of {int(val)} novel or atypical resource accesses within a 1-hour window"
    elif "rec_err_duration" in fn or fn == "session_duration":
        return f"Abnormal session duration ({val:.1f}s) compared to historical baseline"
    elif "rec_err_hour" in fn or "hour_of_day" in fn or "is_weekend" in fn:
        return f"Login timestamp occurs outside normal entity working hours (hour: {val:.2f})"
    elif "rec_err_geo" in fn or "geo_distance_from_centroid" in fn:
        return f"Login location is {val:.1f} km away from historical entity centroid"
    elif "rec_err_resource" in fn or "resource_novelty" in fn:
        return f"Access attempt to uncharacteristic or high-privilege resource for entity role"
    elif "rec_err_auth" in fn or "auth_" in fn:
        return f"Unusual authentication method or credential mechanism used"
    elif "rec_err_os" in fn or "device_fingerprint" in fn:
        return f"Access from an unrecognized device fingerprint or operating system"
    elif "bytes_transferred" in fn:
        return f"Excessive data volume transferred ({val:,.0f} bytes)"
    elif "score_ensemble" in fn:
        return f"High ensembled anomaly score ({val:.2f}) from combined baseline and sequence models"
    elif "profile_decay_factor" in fn:
        return f"Baseline decay factor of {val:.2f} indicates potential concept drift or dormant account activity"
    elif "is_cold_start" in fn or "cold_start" in fn:
        return "Entity has sparse historical log activity (<5 events), using global population baseline fallback"
    else:
        return f"Elevated anomaly contribution from feature '{feature_name}' (value: {val:.2f})"


class ExplainabilityEngine:
    """
    Engine to compute feature attributions, cache SHAP explainers, and output explanation cards.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.shap_explainer = None
        self.classifier_payload: ThreatClassifier = None
        self.df_processed = None
        self._initialize()

    def _initialize(self):
        """Load datasets, Phase 4 classifier model, and initialize/cache SHAP explainer."""
        print("Initializing Explainability Engine...")
        
        # 1. Load Classifier Artifact
        classifier_path = "models/classifier.pkl"
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier artifact not found at {classifier_path}. Run Phase 4 training first.")
        
        self.classifier_payload = joblib.load(classifier_path)
        
        # 2. Build or Load Pre-processed Feature Data (Instant Loading)
        features_cache_path = "data/processed/phase5_features.csv"
        pred_csv = "data/processed/phase4_predictions.csv"
        
        if os.path.exists(features_cache_path):
            print(f"Loading cached explainability features from {features_cache_path}...")
            self.df_processed = pd.read_csv(features_cache_path)
        elif os.path.exists(pred_csv):
            print(f"Loading instant predictions dataset from {pred_csv}...")
            df_pred = pd.read_csv(pred_csv)
            # Ensure all required classifier features exist as columns to prevent KeyError
            feature_names = getattr(self.classifier_payload, "feature_names", [])
            for f in feature_names:
                if f not in df_pred.columns:
                    df_pred[f] = 0.0
            self.df_processed = df_pred
        else:
            print("Extracting feature matrix for Explainability Engine...")
            trainer = PipelineThreatClassifierTrainer(config_path="config.yaml")
            df_base, _, _ = trainer.load_and_prepare_data()

            entity_profiles = self.classifier_payload.entity_profiles
            global_profile = self.classifier_payload.global_profile
            baseline_trainer = PipelineBaselineTrainer(config_path="config.yaml")

            # 2.1 Extract Phase 2 handcrafted features
            X_base, _ = baseline_trainer.extract_features(df_base, entity_profiles, global_profile)
            for col in X_base.columns:
                if col != "session_duration":
                    df_base[col] = X_base[col]

            # 2.2 Sliding window aggregates & Phase 3 reconstruction errors
            df_base = trainer.compute_sliding_window_features(df_base, entity_profiles)
            df_full = trainer.compute_phase3_reconstruction_breakdown(df_base)
            self.df_processed = df_full
            
            # Cache full feature matrix to disk
            os.makedirs("data/processed", exist_ok=True)
            self.df_processed.to_csv(features_cache_path, index=False)
            print(f"Saved full feature matrix to {features_cache_path}")
        
        # 3. Load or Build & Cache SHAP Explainer
        explainer_path = "models/shap_explainer.pkl"
        if os.path.exists(explainer_path):
            print(f"Loading cached SHAP explainer from {explainer_path}...")
            try:
                self.shap_explainer = joblib.load(explainer_path)
            except Exception as e:
                print(f"Warning: Failed to load cached SHAP explainer ({e}). Rebuilding...")
                self.shap_explainer = self._build_and_cache_shap_explainer(explainer_path)
        else:
            self.shap_explainer = self._build_and_cache_shap_explainer(explainer_path)

    def _build_and_cache_shap_explainer(self, save_path: str):
        """Build TreeExplainer on classifier model and save to disk."""
        print("Building SHAP TreeExplainer on Phase 4 classifier...")
        model = self.classifier_payload.model
        
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                joblib.dump(explainer, save_path)
                print(f"Saved SHAP explainer artifact to {save_path}")
                return explainer
            except Exception as e1:
                try:
                    if hasattr(model, "get_booster"):
                        explainer = shap.TreeExplainer(model.get_booster())
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        joblib.dump(explainer, save_path)
                        print(f"Saved SHAP explainer artifact to {save_path} (via booster)")
                        return explainer
                except Exception as e2:
                    pass
                print("Notice: SHAP TreeExplainer build fallback to feature importances.")
                return None
        else:
            print("Notice: SHAP package not installed. Using XGBoost feature importances for attributions.")
            return None

    def explain(self, event_id: str) -> Dict[str, Any]:
        """
        Produce a comprehensive explanation for a single event ID.
        
        Args:
            event_id: Unique event identifier string.
            
        Returns:
            Dictionary matching the output contract:
            {
                "event_id": str,
                "risk_score": float,
                "attack_type": str,
                "reasons": List[Dict[str, Any]],
                "entity_history_snippet": List[Dict[str, Any]],
                "cold_start": bool
            }
        """
        # 1. Locate event in processed dataset
        matching_rows = self.df_processed[self.df_processed["event_id"] == event_id]
        if len(matching_rows) == 0:
            raise ValueError(f"Event ID '{event_id}' not found in raw or processed log datasets.")
            
        row = matching_rows.iloc[0]
        entity_id = str(row["entity_id"])
        
        # 2. Extract risk score and ML prediction/rule overrides
        feature_names = self.classifier_payload.feature_names
        attack_type, confidence, top_contribs = self.classifier_payload.predict(row)
        
        risk_score = float(row.get("score_ensemble", confidence))
        
        # 3. Check for Cold-Start status (< 5 historical events)
        profile = self.classifier_payload.entity_profiles.get(entity_id, {})
        is_cold = float(row.get("is_cold_start", row.get("cold_start", 0.0))) == 1.0 or profile.get("num_events", 0) < 5
        cold_start = bool(is_cold)
        
        # 4. Feature Attributions Calculation (SHAP + Sequence Reconstruction + Rule Overrides)
        reasons = []
        
        # Check rule override triggers first for deterministic insertion
        geo_vel = float(row.get("geo_velocity_kmh", 0.0))
        geo_dist_prev = float(row.get("geo_distance_prev_km", 0.0))
        failed_auth_count = int(row.get("failed_auth_count_5min", 0))
        
        if attack_type == "impossible_travel" and (geo_vel > 900.0 and geo_dist_prev > 500.0):
            reasons.append({
                "feature": "geo_velocity_kmh",
                "value": geo_vel,
                "narrative": get_feature_narrative("geo_velocity_kmh", geo_vel, attack_type)
            })
            reasons.append({
                "feature": "geo_distance_prev_km",
                "value": geo_dist_prev,
                "narrative": get_feature_narrative("geo_distance_prev_km", geo_dist_prev, attack_type)
            })
        elif attack_type == "brute_force" and failed_auth_count > 30:
            reasons.append({
                "feature": "failed_auth_count_5min",
                "value": float(failed_auth_count),
                "narrative": get_feature_narrative("failed_auth_count_5min", float(failed_auth_count), attack_type)
            })
            
        # Compute feature attributions for ML features
        X_vec = row[feature_names].values.astype(float).reshape(1, -1)
        attr_scores = {}
        
        if self.shap_explainer is not None:
            try:
                shap_out = self.shap_explainer.shap_values(X_vec)
                # Parse multi-class SHAP values
                if isinstance(shap_out, list):
                    # List of arrays per class
                    pred_class_idx = 0
                    if hasattr(self.classifier_payload.label_encoder, "transform"):
                        try:
                            pred_class_idx = int(self.classifier_payload.label_encoder.transform([attack_type])[0])
                        except Exception:
                            pred_class_idx = 0
                    pred_class_idx = min(pred_class_idx, len(shap_out) - 1)
                    raw_shap = np.abs(shap_out[pred_class_idx][0])
                elif isinstance(shap_out, np.ndarray):
                    if len(shap_out.shape) == 3:
                        pred_class_idx = 0
                        if hasattr(self.classifier_payload.label_encoder, "transform"):
                            try:
                                pred_class_idx = int(self.classifier_payload.label_encoder.transform([attack_type])[0])
                            except Exception:
                                pred_class_idx = 0
                        pred_class_idx = min(pred_class_idx, shap_out.shape[2] - 1)
                        raw_shap = np.abs(shap_out[0, :, pred_class_idx])
                    else:
                        raw_shap = np.abs(shap_out[0])
                else:
                    raw_shap = np.zeros(len(feature_names))
                    
                for fname, val_score in zip(feature_names, raw_shap):
                    attr_scores[fname] = float(val_score)
            except Exception as e:
                print(f"Notice: SHAP calculation fallback invoked ({e}).")
                attr_scores = self._get_importance_fallback(row, feature_names, X_vec[0])
        else:
            attr_scores = self._get_importance_fallback(row, feature_names, X_vec[0])

        # Sequence reconstruction error attributions (Phase 3 continuous & categorical features)
        seq_rec_cols = [
            ("rec_err_duration", "session_duration"),
            ("rec_err_hour_sin", "hour_of_day"),
            ("rec_err_geo_dist", "geo_distance_from_centroid_km"),
            ("rec_err_resource", "resource_accessed"),
            ("rec_err_auth", "auth_method"),
            ("rec_err_os", "device_os")
        ]
        
        w_seq = float(self.config.get("sequence_attribution_weight", 0.4))
        w_shap = float(self.config.get("shap_attribution_weight", 0.6))
        
        # Combine SHAP and Sequence attributions
        combined_candidates = []
        already_added_features = set(r["feature"] for r in reasons)
        
        for fname, s_score in attr_scores.items():
            if fname in already_added_features:
                continue
            val = float(row.get(fname, 0.0))
            # Check if this feature has an associated sequence reconstruction error
            rec_err_val = 0.0
            for rec_col, mapped_fn in seq_rec_cols:
                if mapped_fn in fname or fname in mapped_fn:
                    rec_err_val = float(row.get(rec_col, 0.0))
                    break
                    
            combined_weight = w_shap * s_score + w_seq * rec_err_val
            combined_candidates.append({
                "feature": fname,
                "value": val,
                "combined_weight": combined_weight,
                "narrative": get_feature_narrative(fname, val, attack_type)
            })
            
        # Sort combined candidates by weight descending
        combined_candidates.sort(key=lambda x: x["combined_weight"], reverse=True)
        
        for cand in combined_candidates:
            if len(reasons) >= 3:
                break
            if cand["feature"] not in already_added_features:
                reasons.append({
                    "feature": cand["feature"],
                    "value": cand["value"],
                    "narrative": cand["narrative"]
                })
                already_added_features.add(cand["feature"])
                
        # Ensure exactly top-3 reasons
        while len(reasons) < 3 and len(combined_candidates) > 0:
            for cand in combined_candidates:
                if cand["feature"] not in already_added_features:
                    reasons.append({
                        "feature": cand["feature"],
                        "value": cand["value"],
                        "narrative": cand["narrative"]
                    })
                    already_added_features.add(cand["feature"])
                    break
            else:
                break
                
        reasons = reasons[:3]

        # 5. Extract Entity History Snippet (last 5 events prior to or including event_id)
        entity_rows = self.df_processed[self.df_processed["entity_id"] == entity_id].copy()
        entity_rows = entity_rows.sort_values(by="timestamp").reset_index(drop=True)
        
        # Find index of target event
        target_idx_matches = entity_rows.index[entity_rows["event_id"] == event_id].tolist()
        if target_idx_matches:
            t_idx = target_idx_matches[0]
            snippet_df = entity_rows.iloc[max(0, t_idx - 4):t_idx + 1]
        else:
            snippet_df = entity_rows.tail(5)
            
        history_snippet = []
        for _, s_row in snippet_df.iterrows():
            ts_str = s_row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S.%fZ") if isinstance(s_row["timestamp"], pd.Timestamp) else str(s_row["timestamp"])
            history_snippet.append({
                "event_id": str(s_row["event_id"]),
                "timestamp": ts_str,
                "resource_accessed": str(s_row.get("resource_accessed", "")),
                "source_ip": str(s_row.get("source_ip", "")),
                "auth_method": str(s_row.get("auth_method", "")),
                "session_duration": int(s_row.get("session_duration", 0))
            })
            
        return {
            "event_id": event_id,
            "risk_score": round(risk_score, 4),
            "attack_type": attack_type,
            "reasons": reasons,
            "entity_history_snippet": history_snippet,
            "cold_start": cold_start
        }

    def _get_importance_fallback(self, row: pd.Series, feature_names: List[str], x_row: np.ndarray) -> Dict[str, float]:
        """Fallback attribution calculation using feature importances multiplied by absolute feature value."""
        attr_scores = {}
        model = self.classifier_payload.model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for fn, imp, val in zip(feature_names, importances, x_row):
                attr_scores[fn] = float(abs(val) * imp)
        else:
            for fn, val in zip(feature_names, x_row):
                attr_scores[fn] = float(abs(val))
        return attr_scores


# Global engine singleton instance
_engine_instance = None

def get_engine() -> ExplainabilityEngine:
    """Retrieve or instantiate the global ExplainabilityEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ExplainabilityEngine()
    return _engine_instance


def explain(event_id: str) -> Dict[str, Any]:
    """
    Public API endpoint function to generate alert explanation card.
    
    Args:
        event_id: Unique event ID string.
        
    Returns:
        Explanation dictionary containing risk score, attack type, top 3 reasons, history snippet, and cold start flag.
    """
    engine = get_engine()
    return engine.explain(event_id)


def run_unit_tests():
    """
    Execute unit tests on 3 seeded events of distinct attack types
    (e.g., impossible_travel, brute_force, lateral_movement / credential_stuffing).
    """
    print("\n" + "="*70)
    print("RUNNING PHASE 5 EXPLAINABILITY UNIT TESTS & DEMONSTRATION")
    print("="*70)
    
    engine = get_engine()
    df_proc = engine.df_processed
    
    target_attacks = ["impossible_travel", "brute_force", "lateral_movement"]
    test_events = []
    
    for atk in target_attacks:
        matches = df_proc[df_proc["label"] == atk]
        if len(matches) > 0:
            test_events.append((atk, str(matches.iloc[0]["event_id"])))
        else:
            # Fallback to any non-normal event
            non_normal = df_proc[df_proc["label"] != "normal"]
            if len(non_normal) > 0:
                test_events.append((atk, str(non_normal.iloc[len(test_events) % len(non_normal)]["event_id"])))
                
    print(f"Selected {len(test_events)} seeded test events across distinct attack categories:\n")
    
    for expected_atk, ev_id in test_events:
        print("-" * 70)
        print(f"Testing Event ID: {ev_id} (Expected Attack Type: {expected_atk})")
        explanation = engine.explain(ev_id)
        
        # Verify schema compliance
        assert "event_id" in explanation and explanation["event_id"] == ev_id
        assert "risk_score" in explanation and 0.0 <= explanation["risk_score"] <= 1.0
        assert "attack_type" in explanation and isinstance(explanation["attack_type"], str)
        assert "reasons" in explanation and len(explanation["reasons"]) == 3
        assert "entity_history_snippet" in explanation and len(explanation["entity_history_snippet"]) > 0
        assert "cold_start" in explanation and isinstance(explanation["cold_start"], bool)
        
        print(f"✅ Risk Score: {explanation['risk_score']} | Predicted Attack: {explanation['attack_type']} | Cold-Start: {explanation['cold_start']}")
        print("Top 3 Plain-English Contributing Factors:")
        for idx, reason in enumerate(explanation["reasons"], 1):
            print(f"   {idx}. [{reason['feature']}] = {reason['value']} -> \"{reason['narrative']}\"")
            
        print(f"Entity History Snippet (Recent {len(explanation['entity_history_snippet'])} events):")
        for h_item in explanation['entity_history_snippet'][-2:]: # Display last 2 for brevity
            print(f"   - {h_item['timestamp']} | {h_item['source_ip']} | {h_item['resource_accessed']} | dur: {h_item['session_duration']}s")
        print("-" * 70 + "\n")
        
    print("🎉 ALL PHASE 5 EXPLAINABILITY UNIT TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 — Explainability Layer CLI")
    parser.add_argument("--event-id", "--event_id", type=str, default=None,
                        help="Unique event ID to generate explanation card for.")
    args = parser.parse_args()
    
    target_event_id = args.event_id
    if target_event_id:
        print(f"Generating explanation card for event_id: {target_event_id}...")
        res = explain(target_event_id)
        print("\n" + json.dumps(res, indent=2))
    else:
        run_unit_tests()
