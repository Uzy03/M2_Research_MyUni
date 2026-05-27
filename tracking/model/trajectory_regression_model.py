import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from model.matchvoice_Qformer import BertLMHeadModel
from tracking.encoder import TrackingEncoder


class TrajectoryRegressionModel(nn.Module):
    def __init__(self, K=5, N=23, num_query=32, d_model=256, num_features=5, pool_mode='mean_pool'):
        """
        Args:
            K: Number of future frames to predict
            N: Number of players
            num_query: Number of query tokens for Q-Former
            d_model: Dimension of internal model
            num_features: Number of features per player (e.g., x, y, confidence, etc.)
        """
        super().__init__()
        self.K = K
        self.N = N
        self.num_query = num_query
        
        # Tracking encoder: processes (B, T, N, F) -> (B, T, 768)
        self.tracking_encoder = TrackingEncoder(
            num_players=N,
            in_features=num_features,
            d_model=d_model,
            nhead=4,
            num_spatial_layers=2,
            num_temporal_layers=2,
            out_features=768,
            pool_mode=pool_mode,
        )
        
        # Q-Former initialization (same pattern as matchvoice_model_all_blocks.py)
        encoder_config = BertConfig.from_pretrained('bert-base-uncased')
        encoder_config.num_hidden_layers = 2
        encoder_config.encoder_width = 768
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query
        
        self.qformer = BertLMHeadModel(config=encoder_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query, encoder_config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        # Regression head: (768) -> (K, N, 2)
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, K * N * 2)
        )
    
    def forward(self, tracking, mask):
        """
        Args:
            tracking: (B, T, N, F) - tracking features
            mask: (B, T, N) bool - True indicates missing values
        
        Returns:
            pred: (B, K, N, 2) - predicted positions for next K frames
        """
        B = tracking.shape[0]
        
        # Get frame-level features from tracking encoder
        frame_feat = self.tracking_encoder(tracking, mask)  # (B, T, 768)
        
        # Q-Former forward pass
        frame_atts = torch.ones(frame_feat.shape[:2], dtype=torch.long, device=frame_feat.device)
        query = self.query_tokens.expand(B, -1, -1)
        
        qout = self.qformer.bert(
            query_embeds=query,
            encoder_hidden_states=frame_feat,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        ).last_hidden_state  # (B, num_query, 768)
        
        # Pool query tokens to get a fixed-size representation
        pooled = qout.mean(dim=1)  # (B, 768)
        
        # Regression head
        pred = self.regression_head(pooled).view(B, self.K, self.N, 2)  # (B, K, N, 2)
        
        return pred
    
    def compute_loss(self, pred, target_xy, target_mask):
        """
        Compute MSE loss only on valid (non-missing) targets.
        
        Args:
            pred: (B, K, N, 2) - predicted positions
            target_xy: (B, K, N, 2) - ground truth positions
            target_mask: (B, K, N) bool - True indicates missing values
        
        Returns:
            loss: scalar tensor
        """
        # Invert mask: True = valid, False = invalid
        valid = ~target_mask  # (B, K, N)
        valid_flat = valid.unsqueeze(-1).expand_as(pred)  # (B, K, N, 2)
        
        # Return zero loss if no valid samples to avoid NaN
        if valid_flat.sum() == 0:
            return torch.zeros(1, device=pred.device, requires_grad=True).squeeze()
        
        return F.mse_loss(pred[valid_flat], target_xy[valid_flat])
    
    def load_pretrained(self, path):
        """
        Load pretrained weights with shape matching and strict=False.
        
        Args:
            path: Path to checkpoint file
        """
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
        # Load only keys that match in shape
        model_state = self.state_dict()
        filtered = {
            k: v for k, v in state_dict.items() 
            if k in model_state and model_state[k].shape == v.shape
        }
        
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f'Loaded {len(filtered)} keys, missing {len(missing)}, unexpected {len(unexpected)}')
