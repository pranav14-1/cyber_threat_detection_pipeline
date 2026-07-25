"""
Phase 3: Deep Learning Sequence Models
Implements FeatureEmbedder, BiLSTM Autoencoder, and Transformer Encoder models.
Features categorical learned embeddings and multi-headed reconstruction projection.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class FeatureEmbedder(nn.Module):
    def __init__(self, num_resources: int, num_auths: int, num_os: int, embed_dims: Tuple[int, int, int] = (16, 8, 8)):
        """Learns low-dimensional continuous representations of categorical access events."""
        super().__init__()
        # +1 for PAD, +1 for UNK
        self.res_embed = nn.Embedding(num_resources + 2, embed_dims[0], padding_idx=0)
        self.auth_embed = nn.Embedding(num_auths + 2, embed_dims[1], padding_idx=0)
        self.os_embed = nn.Embedding(num_os + 2, embed_dims[2], padding_idx=0)
        
        self.total_embed_dim = sum(embed_dims)
        
    def forward(self, cat_features: torch.Tensor, cont_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cat_features: Tensor of shape [batch, seq_len, 3] (Resource, Auth, OS)
            cont_features: Tensor of shape [batch, seq_len, 4]
        Returns:
            x: Concatenated dense vector sequence of shape [batch, seq_len, embed_dim + 4]
        """
        e_res = self.res_embed(cat_features[..., 0])
        e_auth = self.auth_embed(cat_features[..., 1])
        e_os = self.os_embed(cat_features[..., 2])
        
        # Concat embeddings and continuous features
        x = torch.cat([e_res, e_auth, e_os, cont_features], dim=-1)
        return x

class ReconstructionHeads(nn.Module):
    def __init__(self, hidden_dim: int, num_resources: int, num_auths: int, num_os: int):
        """Projects hidden vectors back to the original feature spaces."""
        super().__init__()
        self.proj_cont = nn.Linear(hidden_dim, 4)
        self.proj_res = nn.Linear(hidden_dim, num_resources + 2)
        self.proj_auth = nn.Linear(hidden_dim, num_auths + 2)
        self.proj_os = nn.Linear(hidden_dim, num_os + 2)
        
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            cont_pred: [batch, seq_len, 4]
            res_logits: [batch, seq_len, num_res]
            auth_logits: [batch, seq_len, num_auth]
            os_logits: [batch, seq_len, num_os]
        """
        cont_pred = self.proj_cont(h)
        res_logits = self.proj_res(h)
        auth_logits = self.proj_auth(h)
        os_logits = self.proj_os(h)
        return cont_pred, res_logits, auth_logits, os_logits

class BiLSTMAutoencoder(nn.Module):
    def __init__(self, num_resources: int, num_auths: int, num_os: int, hidden_dim: int = 128, embed_dims: Tuple[int, int, int] = (16, 8, 8)):
        """Sequence-to-sequence bidirectional LSTM autoencoder."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedder = FeatureEmbedder(num_resources, num_auths, num_os, embed_dims)
        
        d_model = self.embedder.total_embed_dim + 4
        
        # Encoder: 2-layer Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Decoder: Unidirectional LSTM (receives projected bottleneck hidden state)
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.reconstruct_heads = ReconstructionHeads(hidden_dim, num_resources, num_auths, num_os)
        
    def forward(self, cat_features: torch.Tensor, cont_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = cat_features.shape
        
        # 1. Embed features
        x = self.embedder(cat_features, cont_features)
        
        # 2. Encode to sequence representation
        _, (h_n, c_n) = self.encoder(x)
        
        # h_n has shape: [num_layers * num_directions, batch_size, hidden_dim] -> [4, batch, 128]
        # We project/average bidirectional states back to unidirectional [2, batch, 128] for the decoder
        h_dec = h_n.view(2, 2, batch_size, self.hidden_dim).mean(dim=1)
        c_dec = c_n.view(2, 2, batch_size, self.hidden_dim).mean(dim=1)
        
        # 3. Decode: Feed a zero tensor input and reconstruct using initial hidden state bottleneck
        decoder_input = torch.zeros((batch_size, seq_len, x.shape[-1]), device=x.device)
        dec_out, _ = self.decoder(decoder_input, (h_dec, c_dec))
        
        # 4. Project back to output spaces
        return self.reconstruct_heads(dec_out)

class TransformerAutoencoder(nn.Module):
    def __init__(self, num_resources: int, num_auths: int, num_os: int, num_layers: int = 3, nhead: int = 4, hidden_dim: int = 128, embed_dims: Tuple[int, int, int] = (16, 8, 8)):
        """Sequence reconstruction model using Transformer Encoder blocks."""
        super().__init__()
        self.embedder = FeatureEmbedder(num_resources, num_auths, num_os, embed_dims)
        d_model = self.embedder.total_embed_dim + 4
        
        # Linear projection to matching transformer dimension if hidden_dim != d_model
        self.input_projection = nn.Linear(d_model, hidden_dim) if d_model != hidden_dim else nn.Identity()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.reconstruct_heads = ReconstructionHeads(hidden_dim, num_resources, num_auths, num_os)
        
    def forward(self, cat_features: torch.Tensor, cont_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Embed features
        x = self.embedder(cat_features, cont_features)
        
        # 2. Project & Encode through self-attention layers
        h = self.input_projection(x)
        out = self.transformer_encoder(h)
        
        # 3. Project back to feature spaces
        return self.reconstruct_heads(out)
