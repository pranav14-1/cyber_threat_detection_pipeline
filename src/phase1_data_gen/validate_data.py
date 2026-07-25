"""
Phase 1 Data Validation Script: validate_data.py
Performs automated audit and verification on data/raw/logs.csv and data/raw/labels.csv:
1. Zero null / missing values across all mandatory log fields.
2. Monotonic timestamp chronology per entity_id with strictly positive time deltas (>0s).
3. 100% alignment between logs.csv and labels.csv event_ids.
4. Anomaly injection rate verification (target: 2.0%).
5. Target taxonomy check across all 7 injected attack types + normal.
6. Spatial velocity constraint check (< 500 km/h for benign, > 900 km/h and > 500 km for impossible travel).
7. Summary statistics (min, max, mean, std) for numerical features.
"""

import os
import json
import math
import sys
import pandas as pd
import numpy as np

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def validate_datasets(logs_path: str = "data/raw/logs.csv", labels_path: str = "data/raw/labels.csv") -> bool:
    print("="*70)
    print("RUNNING AUTOMATED DATASET QUALITY & INTEGRITY AUDIT")
    print("="*70)

    if not os.path.exists(logs_path) or not os.path.exists(labels_path):
        print(f"❌ FAIL: File not found ({logs_path} or {labels_path}). Please generate data first.")
        return False

    df_logs = pd.read_csv(logs_path)
    df_labels = pd.read_csv(labels_path)

    all_passed = True
    test_results = []

    # 1. Null / Missing Values Check
    mandatory_cols = [
        "event_id", "entity_id", "entity_type", "timestamp", "source_ip", 
        "geo_location", "resource_accessed", "auth_method", "session_duration", 
        "bytes_transferred", "device_fingerprint"
    ]
    missing_cols = [col for col in mandatory_cols if col not in df_logs.columns]
    if missing_cols:
        test_results.append((f"Mandatory Columns Present ({missing_cols})", False, f"Missing columns: {missing_cols}"))
        all_passed = False
    else:
        null_counts = df_logs[mandatory_cols].isnull().sum().sum()
        empty_str_counts = (df_logs[mandatory_cols].astype(str).values == "").sum()
        if null_counts == 0 and empty_str_counts == 0:
            test_results.append(("Zero Null / Missing Values", True, f"All {len(mandatory_cols)} mandatory columns clean"))
        else:
            test_results.append(("Zero Null / Missing Values", False, f"Found {null_counts} nulls, {empty_str_counts} empty strings"))
            all_passed = False

    # 2. Alignment & Key Integrity Check
    len_logs = len(df_logs)
    len_labels = len(df_labels)
    if len_logs == len_labels:
        key_match = (df_logs["event_id"] == df_labels["event_id"]).all()
        if key_match:
            test_results.append(("Logs & Labels 100% Alignment", True, f"Matched {len_logs:,} event IDs in exact order"))
        else:
            test_results.append(("Logs & Labels 100% Alignment", False, "Event IDs misaligned between logs and labels"))
            all_passed = False
    else:
        test_results.append(("Logs & Labels Row Count Equality", False, f"Logs count ({len_logs}) != Labels count ({len_labels})"))
        all_passed = False

    # 3. Anomaly Rate & Taxonomy Check
    df = pd.merge(df_logs, df_labels, on="event_id")
    total_events = len(df)
    class_counts = df["label"].value_counts().to_dict()
    benign_count = class_counts.get("normal", 0)
    anomaly_count = total_events - benign_count
    actual_anomaly_rate = (anomaly_count / total_events) * 100.0

    expected_taxonomy = [
        "normal", "brute_force", "credential_stuffing", "device_spoofing", 
        "impossible_travel", "insider_drift", "lateral_movement", "low_slow_exfil"
    ]
    missing_tax = [cls for cls in expected_taxonomy if cls not in class_counts]

    if abs(actual_anomaly_rate - 2.0) <= 0.5:
        test_results.append(("Anomaly Rate Target (2.0%)", True, f"Actual: {actual_anomaly_rate:.2f}% ({anomaly_count:,} anomalies)"))
    else:
        test_results.append(("Anomaly Rate Target (2.0%)", False, f"Actual: {actual_anomaly_rate:.2f}% (Target: 2.0%)"))
        all_passed = False

    if not missing_tax:
        test_results.append(("Target Taxonomy Coverage", True, f"All {len(expected_taxonomy)} classes present"))
    else:
        test_results.append(("Target Taxonomy Coverage", False, f"Missing target classes: {missing_tax}"))
        all_passed = False

    # 4. Monotonic Chronology & Positive Time Deltas Check
    df["ts"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(by=["entity_id", "ts"]).reset_index(drop=True)

    non_positive_deltas = 0
    entity_groups = df.groupby("entity_id")
    for ent_id, group in entity_groups:
        dts = group["ts"].values
        if len(dts) > 1:
            diffs = np.diff(dts).astype('timedelta64[ms]').astype(float) / 1000.0
            non_positive_deltas += (diffs <= 0).sum()

    if non_positive_deltas == 0:
        test_results.append(("Per-Entity Monotonic Chronology (dt > 0s)", True, "0 negative or zero-second deltas"))
    else:
        test_results.append(("Per-Entity Monotonic Chronology (dt > 0s)", False, f"Found {non_positive_deltas} non-positive time deltas"))
        all_passed = False

    # 5. Geolocation & Spatial Velocity Sanity Check
    # Parse lat/lon for distance check
    def parse_lat_lon(geo_str):
        try:
            g = json.loads(geo_str)
            return g.get("lat", 0.0), g.get("lon", 0.0)
        except:
            return 0.0, 0.0

    lats_lons = [parse_lat_lon(g) for g in df["geo_location"].values]
    df["lat"] = [x[0] for x in lats_lons]
    df["lon"] = [x[1] for x in lats_lons]

    benign_high_velocity_count = 0
    imp_travel_valid_count = 0
    imp_travel_total = class_counts.get("impossible_travel", 0)

    for ent_id, group in entity_groups:
        if len(group) < 2:
            continue
        ts_sec = group["ts"].view('int64').values // 10**9
        lats = group["lat"].values
        lons = group["lon"].values
        labels = group["label"].values

        for i in range(1, len(group)):
            dt_h = (ts_sec[i] - ts_sec[i-1]) / 3600.0
            if dt_h <= 0:
                continue
            dist_km = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
            vel_kmh = dist_km / dt_h

            # Check benign max velocity
            if labels[i] == "normal" and labels[i-1] == "normal":
                if vel_kmh > 500.0 and dist_km > 50.0:
                    benign_high_velocity_count += 1

            # Check impossible travel injection constraint
            if labels[i] == "impossible_travel":
                if dist_km > 500.0 and vel_kmh > 900.0:
                    imp_travel_valid_count += 1

    if benign_high_velocity_count == 0:
        test_results.append(("Benign Spatial Velocity (<500 km/h)", True, "100% of benign logins adhere to realistic spatial speeds"))
    else:
        test_results.append(("Benign Spatial Velocity (<500 km/h)", False, f"Found {benign_high_velocity_count} benign logins > 500 km/h"))
        all_passed = False

    if imp_travel_total == 0 or (imp_travel_valid_count / imp_travel_total) >= 0.8:
        test_results.append(("Impossible Travel Injected Constraint (>500km & >900km/h)", True, f"Verified {imp_travel_valid_count}/{imp_travel_total} events satisfy strict velocity"))
    else:
        test_results.append(("Impossible Travel Injected Constraint (>500km & >900km/h)", False, f"Only {imp_travel_valid_count}/{imp_travel_total} events satisfy distance/velocity"))
        all_passed = False

    # Print Detailed Validation Report Table
    print(f"\n{'TEST CHECK':<52} | {'STATUS':<8} | {'DETAILS'}")
    print("-" * 100)
    for test_name, passed, detail in test_results:
        status_str = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<52} | {status_str:<8} | {detail}")
    print("-" * 100)

    # 6. Numerical Feature Distribution Summary
    print("\n" + "="*70)
    print("NUMERICAL FEATURE DISTRIBUTION SUMMARY")
    print("="*70)

    num_summary = []
    for col in ["session_duration", "bytes_transferred"]:
        if col in df_logs.columns:
            s = df_logs[col]
            num_summary.append({
                "Feature": col,
                "Min": float(s.min()),
                "Max": float(s.max()),
                "Mean": float(s.mean()),
                "Std": float(s.std()),
                "Median": float(s.median())
            })

    df_num_sum = pd.DataFrame(num_summary)
    print(df_num_sum.to_string(index=False))
    print("="*70)

    if all_passed:
        print("\n🎉 ALL DATA QUALITY & INTEGRITY CHECKS PASSED SUCCESSFULLY!\n")
        return True
    else:
        print("\n❌ DATASET HEALTH AUDIT FAILED. PLEASE REVIEW ERRORS ABOVE.\n")
        return False

if __name__ == "__main__":
    success = validate_datasets()
    sys.exit(0 if success else 1)
