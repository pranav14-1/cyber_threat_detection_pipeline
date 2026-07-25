"""
Phase 3: Reconstruction Scoring
Computes timestep-level reconstruction errors, performs calibration on validation set,
and implements a rolling 30-day min-max normalization strategy.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List

def compute_timestep_scores(model: nn.Module,
                             loader: DataLoader,
                             device: torch.device) -> Tuple[List[str], np.ndarray]:
    """Score each event by evaluating reconstruction error at the final timestep of the sequence window."""
    model.eval()
    raw_scores = []
    event_ids = []
    
    criterion_mse_none = nn.MSELoss(reduction="none")
    criterion_ce_none = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
    
    with torch.no_grad():
        for cat_seq, cont_seq, ev_ids, _ in loader:
            cat_seq = cat_seq.to(device)
            cont_seq = cont_seq.to(device)
            
            outputs = model(cat_seq, cont_seq)
            pred_cont, pred_res, pred_auth, pred_os = outputs
            
            # Focus on the final event in the window (index = seq_len - 1)
            idx = cat_seq.shape[1] - 1
            
            # Continuous features reconstruction MSE
            loss_cont = criterion_mse_none(pred_cont[:, idx, :], cont_seq[:, idx, :]).sum(dim=-1)
            
            # Categorical features reconstruction cross-entropy
            loss_res = criterion_ce_none(pred_res[:, idx, :], cat_seq[:, idx, 0])
            loss_auth = criterion_ce_none(pred_auth[:, idx, :], cat_seq[:, idx, 1])
            loss_os = criterion_ce_none(pred_os[:, idx, :], cat_seq[:, idx, 2])
            
            total_loss = loss_cont + loss_res + loss_auth + loss_os
            
            raw_scores.extend(total_loss.cpu().numpy().tolist())
            event_ids.extend(ev_ids)
            
    return event_ids, np.array(raw_scores)

def calibrate_min_max(val_raw_scores: np.ndarray) -> Tuple[float, float]:
    """Calibrate static min-max parameters on validation set scores."""
    min_val = float(np.min(val_raw_scores))
    max_val = float(np.max(val_raw_scores))
    return min_val, max_val

def apply_static_normalization(raw_scores: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Normalize raw scores into range [0.0, 1.0] using static min-max scaling."""
    denom = max_val - min_val
    if abs(denom) < 1e-8:
        denom = 1.0
    norm_scores = (raw_scores - min_val) / denom
    return np.clip(norm_scores, 0.0, 1.0)

def apply_rolling_normalization(df_raw_scores: pd.DataFrame, window_size: str = "30D") -> pd.DataFrame:
    """
    Perform rolling min-max normalization over a trailing window (default 30 days) to handle concept drift.
    df_raw_scores must contain: 'timestamp' (datetime) and 'raw_score' (float).
    """
    df_sorted = df_raw_scores.sort_values(by="timestamp").copy()
    
    # Establish rolling window min/max
    r = df_sorted.rolling(window=window_size, on="timestamp")
    roll_min = r["raw_score"].min()
    roll_max = r["raw_score"].max()
    
    denom = roll_max - roll_min
    # Handle edge case where window min and max are equal
    denom = denom.replace(0.0, 1.0).fillna(1.0)
    
    norm_score = (df_sorted["raw_score"] - roll_min) / denom
    df_sorted["normalized_score"] = norm_score.fillna(0.0).clip(0.0, 1.0)
    
    return df_sorted
