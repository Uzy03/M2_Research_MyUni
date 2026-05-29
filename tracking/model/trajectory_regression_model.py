import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tracking.encoder import TrackingEncoder


class TrajectoryRegressionModel(nn.Module):
    def __init__(self, K=5, N=23, d_model=256, num_features=5, pool_mode='mean_pool', context_len=20):
        """
        Args:
            K: Number of future frames to predict
            N: Number of players
            d_model: Dimension of internal model
            num_features: Number of features per player (e.g., x, y, confidence, etc.)
            pool_mode: Pooling mode for encoder
            context_len: Context length for flattening
        """
        super().__init__()
        self.K = K
        self.N = N
        
        # Tracking encoder: processes (B, T, N, F) -> (B, T_or_N, 768)
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
        
        self.pool_mode = pool_mode
        if pool_mode == 'player_tokens':
            self.regression_head = nn.Linear(768, K * 2)
        else:
            input_dim = context_len * 768
            self.regression_head = nn.Sequential(
                nn.Linear(input_dim, 256),
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
        frame_feat = self.tracking_encoder(tracking, mask)  # (B, T_or_N, 768)
        if self.pool_mode == 'player_tokens':
            pred = self.regression_head(frame_feat).view(B, self.N, self.K, 2)
            pred = pred.permute(0, 2, 1, 3).contiguous()
        else:
            flat = frame_feat.reshape(B, -1)
            pred = self.regression_head(flat).view(B, self.K, self.N, 2)
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
