import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class TrajectoryRegressionDataset(Dataset):
    def __init__(self, json_path, context_len=20, K=5, step=1, max_games=0):
        """
        Dataset for trajectory regression.

        Args:
            json_path: Path to JSON file containing clip metadata
            context_len: Number of context frames to use as input
            K: Number of future frames to predict
            step: Stride for sliding window
            max_games: Max number of games to use (0 = all)
        """
        self.context_len = context_len
        self.K = K
        self.step = step
        self.base_dir = Path(json_path).parent

        # Load metadata
        with open(json_path) as f:
            self.data = json.load(f)

        if max_games > 0:
            seen, allowed = [], set()
            for e in self.data:
                if e['game_id'] not in allowed:
                    seen.append(e['game_id'])
                    if len(seen) > max_games:
                        break
                    allowed.add(e['game_id'])
            self.data = [e for e in self.data if e['game_id'] in allowed]
        
        # Build sliding windows
        self.windows = []
        for clip_idx, entry in enumerate(self.data):
            npy_path = self.base_dir / entry['npy_path']
            npy = np.load(npy_path)  # (T, N, F)
            total_frames = npy.shape[0]
            
            # Calculate max starting position
            max_start = total_frames - context_len - K + 1
            if max_start <= 0:
                continue
            
            # Create sliding windows
            for start in range(0, max_start, step):
                self.windows.append((clip_idx, start))
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Returns:
            context_feat: (context_len, N, F) - input tracking features
            context_mask: (context_len, N) bool - True indicates missing
            target_xy: (K, N, 2) - target x, y positions (missing set to 0)
            target_mask: (K, N) bool - True indicates missing
            seq_id: sequence identifier
        """
        clip_idx, start = self.windows[idx]
        entry = self.data[clip_idx]
        
        # Load numpy arrays
        npy = np.load(self.base_dir / entry['npy_path'])  # (T, N, F)
        mask = np.load(self.base_dir / entry['mask_path'])  # (T, N) bool
        
        # Extract context window
        context_feat = torch.FloatTensor(
            npy[start:start+self.context_len]
        )  # (context_len, N, F)
        context_mask = torch.BoolTensor(
            mask[start:start+self.context_len]
        )  # (context_len, N)
        
        # Extract target window
        target_npy = npy[
            start+self.context_len : start+self.context_len+self.K
        ]  # (K, N, F)
        target_mask_np = mask[
            start+self.context_len : start+self.context_len+self.K
        ]  # (K, N)
        
        # Extract x, y coordinates
        target_xy = torch.FloatTensor(target_npy[:, :, :2])  # (K, N, 2)
        
        # Set missing values to 0
        target_xy[torch.BoolTensor(target_mask_np)] = 0.0
        
        # Create target mask tensor
        target_mask = torch.BoolTensor(target_mask_np)  # (K, N)
        
        # Get sequence ID
        seq_id = entry.get('clip_id', entry.get('seq_id', str(clip_idx)))
        
        return context_feat, context_mask, target_xy, target_mask, seq_id
