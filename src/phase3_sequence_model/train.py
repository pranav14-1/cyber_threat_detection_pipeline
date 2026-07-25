"""
Phase 3: Model Training
Trains the sequence-aware BiLSTM and Transformer autoencoder models on normal-only sequence logs.
Generates metrics and saves model artifacts.
"""

import os
import argparse
import yaml
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

from src.phase3_sequence_model.dataset import SequenceFeatureExtractor, AccessLogSequenceDataset
from src.phase3_sequence_model.model import BiLSTMAutoencoder, TransformerAutoencoder

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_loss(pred_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 cat_targets: torch.Tensor,
                 cont_targets: torch.Tensor,
                 criterion_mse: nn.Module,
                 criterion_ce: nn.Module) -> torch.Tensor:
    """Joint Reconstruction Loss: MSE for continuous features, Cross-Entropy for categorical features."""
    pred_cont, pred_res, pred_auth, pred_os = pred_outputs
    
    # Continuous MSE loss
    loss_cont = criterion_mse(pred_cont, cont_targets)
    
    # Flatten categorical projections and targets
    res_targets = cat_targets[..., 0].contiguous().view(-1)
    auth_targets = cat_targets[..., 1].contiguous().view(-1)
    os_targets = cat_targets[..., 2].contiguous().view(-1)
    
    pred_res_flat = pred_res.view(-1, pred_res.shape[-1])
    pred_auth_flat = pred_auth.view(-1, pred_auth.shape[-1])
    pred_os_flat = pred_os.view(-1, pred_os.shape[-1])
    
    # Categorical CE losses
    loss_res = criterion_ce(pred_res_flat, res_targets)
    loss_auth = criterion_ce(pred_auth_flat, auth_targets)
    loss_os = criterion_ce(pred_os_flat, os_targets)
    
    # Sum joint losses
    return loss_cont + loss_res + loss_auth + loss_os

def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                criterion_mse: nn.Module,
                criterion_ce: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for cat_seq, cont_seq, _, _ in loader:
        cat_seq = cat_seq.to(device)
        cont_seq = cont_seq.to(device)
        
        optimizer.zero_grad()
        outputs = model(cat_seq, cont_seq)
        
        loss = compute_loss(outputs, cat_seq, cont_seq, criterion_mse, criterion_ce)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(cat_seq)
        
    return total_loss / len(loader.dataset)

def eval_epoch(model: nn.Module,
               loader: DataLoader,
               device: torch.device,
               criterion_mse: nn.Module,
               criterion_ce: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for cat_seq, cont_seq, _, _ in loader:
            cat_seq = cat_seq.to(device)
            cont_seq = cont_seq.to(device)
            
            outputs = model(cat_seq, cont_seq)
            loss = compute_loss(outputs, cat_seq, cont_seq, criterion_mse, criterion_ce)
            total_loss += loss.item() * len(cat_seq)
            
    return total_loss / len(loader.dataset)

def plot_curves(train_losses: List[float], val_losses: List[float], save_path: str):
    """Plot and save training and validation loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="royalblue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Sequence Autoencoder Training curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Sequence-Aware Anomaly Detection Model")
    parser.add_argument("--model_type", type=str, default="bilstm", choices=["bilstm", "transformer"],
                        help="Sequence model architecture style")
    parser.add_argument("--incremental", action="store_true",
                        help="Load existing checkpoint and perform rolling retraining")
    args = parser.parse_args()
    
    config = load_config()
    seed = config.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    print("Loading data...")
    df_logs = pd.read_csv("data/raw/logs.csv")
    df_labels = pd.read_csv("data/raw/labels.csv")
    df = pd.merge(df_logs, df_labels, on="event_id")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Split chronologically
    n = len(df)
    train_idx = int(0.70 * n)
    val_idx = int(0.85 * n)
    
    df_train = df.iloc[:train_idx].copy()
    df_val = df.iloc[train_idx:val_idx].copy()
    df_test = df.iloc[val_idx:].copy()
    
    # Extractor fitting or loading
    extractor_path = "models/seq_ae_extractor.pkl"
    if args.incremental and os.path.exists(extractor_path):
        print(f"Loading existing feature extractor from {extractor_path}...")
        extractor = joblib.load(extractor_path)
    else:
        print("Fitting sequence feature extractor on training data...")
        extractor = SequenceFeatureExtractor()
        extractor.fit(df_train)
        os.makedirs("models", exist_ok=True)
        joblib.dump(extractor, extractor_path)
        
    seq_len = config.get("seq_length", 32)
    batch_size = config.get("batch_size", 128)
    epochs = config.get("epochs", 20)
    lr = float(config.get("learning_rate", 0.001))
    
    # Construct sequence datasets (Training is normal-only)
    print("Preparing sequence datasets...")
    df_train_normal = df_train[df_train["label"] == "normal"].copy()
    df_val_normal = df_val[df_val["label"] == "normal"].copy()
    
    train_dataset = AccessLogSequenceDataset(df_train_normal, extractor, seq_len)
    val_dataset = AccessLogSequenceDataset(df_val_normal, extractor, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Vocab size calculations
    num_res = len(extractor.res_vocab)
    num_auth = len(extractor.auth_vocab)
    num_os = len(extractor.os_vocab)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Instantiate appropriate model
    if args.model_type == "bilstm":
        hidden_dim = config.get("lstm_hidden_dim", 128)
        model = BiLSTMAutoencoder(num_res, num_auth, num_os, hidden_dim=hidden_dim)
    else:
        heads = config.get("transformer_heads", 4)
        layers = config.get("transformer_layers", 3)
        model = TransformerAutoencoder(num_res, num_auth, num_os, num_layers=layers, nhead=heads)
        
    model = model.to(device)
    
    # Define optimization criteria
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    checkpoint_path = "models/seq_ae.pt"
    best_val_loss = float("inf")
    start_epoch = 0
    train_losses = []
    val_losses = []
    
    # Handle Incremental Retrain flag
    if args.incremental and os.path.exists(checkpoint_path):
        print(f"Loading existing checkpoint state for incremental training from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Verify compatibility of model type
            if checkpoint.get("model_type") == args.model_type:
                model.load_state_dict(checkpoint["model_state_dict"])
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                print("Checkpoint successfully loaded. Retraining starting...")
                # Lower learning rate slightly for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 0.2
            else:
                print(f"WARNING: Model type mismatch in checkpoint ({checkpoint.get('model_type')} vs {args.model_type}). Training from scratch.")
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint. Training from scratch. Exception: {e}")
            
    print(f"Fitting {args.model_type.upper()} model for up to {epochs} epochs...")
    
    patience = 3
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion_mse, criterion_ce)
        val_loss = eval_epoch(model, val_loader, device, criterion_mse, criterion_ce)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save state
            torch.save({
                "model_type": args.model_type,
                "model_state_dict": model.state_dict(),
                "vocab_dims": (num_res, num_auth, num_os),
                "best_val_loss": best_val_loss
            }, checkpoint_path)
            print("  --> Saved best model checkpoint.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Save training curves
    curves_path = "reports/figures/phase3_curves.png"
    plot_curves(train_losses, val_losses, curves_path)
    print(f"Training curves saved to {curves_path}")
    
    # Run evaluation script dynamically to report results
    print("\nTraining completed. Executing evaluation script...")
    import subprocess
    subprocess.run(["python", "-m", "src.phase3_sequence_model.eval"])

if __name__ == "__main__":
    main()
