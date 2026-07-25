"""
Phase 3: Dataset Preparation
Defines sequence dataset and feature extraction tools for access logs.
Handles vocabulary mapping, cyclical hour conversion, normalization, and sequence windowing.
"""

import json
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple

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

class SequenceFeatureExtractor:
    def __init__(self):
        """Initialize feature vocabularies and scale parameters."""
        # 0 = Padding, 1 = Unknown
        self.res_vocab = {"<PAD>": 0, "<UNK>": 1}
        self.auth_vocab = {"<PAD>": 0, "<UNK>": 1}
        self.os_vocab = {"<PAD>": 0, "<UNK>": 1}
        
        # Scaling parameters
        self.mean_dur = 0.0
        self.std_dur = 1.0
        self.mean_dist = 0.0
        self.std_dist = 1.0
        
        # Entity home centroids (fitted from training)
        self.entity_centroids = {}
        self.global_centroid = (0.0, 0.0)
        
    def fit(self, df_train: pd.DataFrame):
        """Fit vocabulary mappings and scaling statistics on the training set."""
        # Build vocabs
        for res in df_train["resource_accessed"].dropna().unique():
            if res not in self.res_vocab:
                self.res_vocab[res] = len(self.res_vocab)
                
        for auth in df_train["auth_method"].dropna().unique():
            if auth not in self.auth_vocab:
                self.auth_vocab[auth] = len(self.auth_vocab)
                
        # Parse device fingerprints to extract OS
        print("Fitting vocabularies and continuous feature scaling statistics...")
        df_train_parsed = df_train.copy()
        df_train_parsed["geo_parsed"] = df_train_parsed["geo_location"].apply(safe_json_loads)
        df_train_parsed["device_parsed"] = df_train_parsed["device_fingerprint"].apply(safe_json_loads)
        
        df_train_parsed["lat"] = df_train_parsed["geo_parsed"].apply(lambda x: x.get("lat", 0.0))
        df_train_parsed["lon"] = df_train_parsed["geo_parsed"].apply(lambda x: x.get("lon", 0.0))
        df_train_parsed["device_os"] = df_train_parsed["device_parsed"].apply(lambda x: x.get("os", ""))
        
        for os_val in df_train_parsed["device_os"].dropna().unique():
            if os_val and os_val not in self.os_vocab:
                self.os_vocab[os_val] = len(self.os_vocab)
                
        # Centroids
        self.global_centroid = (float(df_train_parsed["lat"].mean()), float(df_train_parsed["lon"].mean()))
        entity_groups = df_train_parsed.groupby("entity_id")
        for ent_id, gp in entity_groups:
            self.entity_centroids[ent_id] = (float(gp["lat"].mean()), float(gp["lon"].mean()))
            
        # Distances
        dists = []
        for ent_id, lat, lon in zip(df_train_parsed["entity_id"].values, df_train_parsed["lat"].values, df_train_parsed["lon"].values):
            cent = self.entity_centroids.get(ent_id, self.global_centroid)
            dists.append(haversine_distance(cent[0], cent[1], lat, lon))
        df_train_parsed["geo_distance"] = dists
        
        # Scaling stats (Normal only to keep scale clean)
        df_normal = df_train_parsed[df_train_parsed["label"] == "normal"]
        if len(df_normal) == 0:
            df_normal = df_train_parsed
            
        self.mean_dur = float(df_normal["session_duration"].mean())
        self.std_dur = float(df_normal["session_duration"].std()) if df_normal["session_duration"].std() > 1e-4 else 1.0
        
        self.mean_dist = float(df_normal["geo_distance"].mean())
        self.std_dist = float(df_normal["geo_distance"].std()) if df_normal["geo_distance"].std() > 1e-4 else 1.0
        
    def transform_row_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a dataframe into clean categorical indices and scaled continuous arrays."""
        df_parsed = df.copy()
        df_parsed["geo_parsed"] = df_parsed["geo_location"].apply(safe_json_loads)
        df_parsed["device_parsed"] = df_parsed["device_fingerprint"].apply(safe_json_loads)
        
        df_parsed["lat"] = df_parsed["geo_parsed"].apply(lambda x: x.get("lat", 0.0))
        df_parsed["lon"] = df_parsed["geo_parsed"].apply(lambda x: x.get("lon", 0.0))
        df_parsed["device_os"] = df_parsed["device_parsed"].apply(lambda x: x.get("os", ""))
        
        # Timestamps
        ts = pd.to_datetime(df_parsed["timestamp"], format="ISO8601")
        hours = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
        hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
        
        # Distances
        dists = []
        for ent_id, lat, lon in zip(df_parsed["entity_id"].values, df_parsed["lat"].values, df_parsed["lon"].values):
            cent = self.entity_centroids.get(ent_id, self.global_centroid)
            dists.append(haversine_distance(cent[0], cent[1], lat, lon))
        
        # Scale continuous
        norm_dur = (df_parsed["session_duration"].values - self.mean_dur) / self.std_dur
        norm_dist = (np.array(dists) - self.mean_dist) / self.std_dist
        
        cont_features = np.column_stack([
            norm_dur,
            hour_sin,
            hour_cos,
            norm_dist
        ])
        
        # Map categoricals
        cat_features = []
        for res, auth, os_val in zip(df_parsed["resource_accessed"].values, df_parsed["auth_method"].values, df_parsed["device_os"].values):
            res_idx = self.res_vocab.get(res, self.res_vocab["<UNK>"])
            auth_idx = self.auth_vocab.get(auth, self.auth_vocab["<UNK>"])
            os_idx = self.os_vocab.get(os_val, self.os_vocab["<UNK>"])
            cat_features.append([res_idx, auth_idx, os_idx])
            
        return np.array(cat_features, dtype=np.int64), np.array(cont_features, dtype=np.float32)

class AccessLogSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, extractor: SequenceFeatureExtractor, seq_length: int = 32):
        """Construct sequence dataset containing windows of length seq_length for each event."""
        self.seq_length = seq_length
        self.event_ids = df["event_id"].values
        self.labels = (df["label"] != "normal").astype(int).values
        
        # Generate numeric features for all rows
        self.cat_feats, self.cont_feats = extractor.transform_row_features(df)
        
        # Group indices by entity to maintain separate timelines
        df_temp = df.copy()
        df_temp["global_idx"] = np.arange(len(df))
        entity_groups = df_temp.groupby("entity_id")
        
        # Precompute the slice indices for each event's sequence window
        self.sequence_indices = []
        for ent_id, gp in entity_groups:
            gp_sorted = gp.sort_values(by="timestamp")
            gp_indices = gp_sorted["global_idx"].values
            
            # Slide over each event in chronological order
            for i in range(len(gp_indices)):
                target_idx = gp_indices[i]
                
                # Get the window of indices up to the current event (inclusive)
                window_indices = gp_indices[max(0, i - seq_length + 1) : i + 1]
                
                # Number of padding events needed
                pad_size = seq_length - len(window_indices)
                
                self.sequence_indices.append({
                    "target_idx": target_idx,
                    "window_indices": window_indices,
                    "pad_size": pad_size
                })
                
        # Sort back to match initial dataframe sequence order
        self.sequence_indices = sorted(self.sequence_indices, key=lambda x: x["target_idx"])
        
    def __len__(self) -> int:
        return len(self.sequence_indices)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        info = self.sequence_indices[idx]
        target_idx = info["target_idx"]
        window_idxs = info["window_indices"]
        pad_size = info["pad_size"]
        
        # Fetch actual slices
        cat_win = self.cat_feats[window_idxs]
        cont_win = self.cont_feats[window_idxs]
        
        # Perform left-padding (PAD = 0)
        if pad_size > 0:
            cat_pad = np.zeros((pad_size, 3), dtype=np.int64)
            cont_pad = np.zeros((pad_size, 4), dtype=np.float32)
            cat_seq = np.vstack([cat_pad, cat_win])
            cont_seq = np.vstack([cont_pad, cont_win])
        else:
            cat_seq = cat_win
            cont_seq = cont_win
            
        return (
            torch.tensor(cat_seq, dtype=torch.long),
            torch.tensor(cont_seq, dtype=torch.float32),
            self.event_ids[target_idx],
            self.labels[target_idx]
        )
