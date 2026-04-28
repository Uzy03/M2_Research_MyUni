import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset


def format_trajectory(xy, K):
    """
    Convert trajectory coordinates to string format.
    
    Args:
        xy: numpy array of shape (K, N, 2) — x,y coordinates (features[:, :, :2] slice)
        K: number of frames in trajectory
    
    Returns:
        str: Formatted as "p0:[(0.82,0.38),(0.80,0.37),...],p1:[(x,y),...],p22:[(x,y),...]"
    """
    parts = []
    for n in range(xy.shape[1]):
        coords = ','.join(
            f'({xy[k,n,0]:.2f},{xy[k,n,1]:.2f})' 
            for k in range(K)
        )
        parts.append(f'p{n}:[{coords}]')
    return ','.join(parts)


def parse_trajectory(text, K, N=23):
    """
    Restore (K, N, 2) float32 ndarray from generated text.
    Failed to parse players are filled with NaN.
    
    Args:
        text: String containing trajectory data in format: "p0:[...],p1:[...],...,p22:[...]"
        K: Number of frames in trajectory
        N: Number of players (default 23)
    
    Returns:
        numpy.ndarray: Shape (K, N, 2) with float32 dtype, NaN for missing/invalid values
    """
    result = np.full((K, N, 2), np.nan, dtype=np.float32)
    
    player_pattern = re.compile(r'p(\d+):\[([^\]]*)\]')
    coord_pattern = re.compile(r'\(([0-9.nan]+),([0-9.nan]+)\)')
    
    for pm in player_pattern.finditer(text):
        n = int(pm.group(1))
        if n >= N:
            continue
        
        coords = coord_pattern.findall(pm.group(2))
        for k, (xv, yv) in enumerate(coords[:K]):
            try:
                result[k, n, 0] = float(xv)
                result[k, n, 1] = float(yv)
            except ValueError:
                pass
    
    return result


def compute_ade_fde(pred_xy, gt_xy):
    """
    Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).
    
    Args:
        pred_xy: predicted trajectory, shape (K, N, 2) float32 ndarray (NaN values ignored)
        gt_xy: ground truth trajectory, shape (K, N, 2) float32 ndarray (NaN values ignored)
    
    Returns:
        tuple: (ade: float, fde: float)
            - ADE: mean displacement error across all frames and players
            - FDE: mean displacement error at final frame (K-1)
    """
    diff = pred_xy - gt_xy  # (K, N, 2)
    dist = np.sqrt((diff**2).sum(-1))  # (K, N)
    
    # ADE: average across all valid frames and players
    valid = ~np.isnan(dist)
    ade = dist[valid].mean() if valid.any() else np.nan
    
    # FDE: average at final frame only
    fde_dist = dist[-1]
    fde_valid = ~np.isnan(fde_dist)
    fde = fde_dist[fde_valid].mean() if fde_valid.any() else np.nan
    
    return float(ade), float(fde)


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory prediction using sliding window over SoccerNet tracking data.
    
    Returns context features and masks for input, and target trajectory as formatted text.
    """
    
    def __init__(self, json_path, context_len=100, K=10, step=5):
        """
        Initialize TrajectoryDataset with sliding window configuration.
        
        Args:
            json_path (str): Path to JSON file with clip metadata (npy_path, mask_path, seq_id)
            context_len (int): Length of context window in frames (default 100)
            K (int): Number of frames in target trajectory (default 10)
            step (int): Stride for sliding window (default 5)
        """
        self.context_len = context_len
        self.K = K
        self.step = step
        
        with open(json_path) as f:
            self.data = json.load(f)
        
        # Create sliding windows: (clip_idx, start_frame)
        self.windows = []
        for clip_idx, entry in enumerate(self.data):
            npy = np.load(entry['npy_path'])
            total_frames = npy.shape[0]
            
            # Ensure we have enough frames for context + target
            max_start = total_frames - context_len - K + 1
            if max_start <= 0:
                continue
            
            # Generate sliding window positions
            for start in range(0, max_start, step):
                self.windows.append((clip_idx, start))
    
    def __len__(self):
        """Return total number of sliding windows."""
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a single sliding window sample.
        
        Args:
            idx (int): Index of the sliding window
        
        Returns:
            tuple: (context_feat, context_mask, target_text, seq_id, start_frame)
                - context_feat: FloatTensor of shape (context_len, N, F)
                - context_mask: BoolTensor of shape (context_len, N)
                - target_text: formatted trajectory string
                - seq_id: unique identifier for the sequence
                - start_frame: integer start frame index
        """
        clip_idx, start = self.windows[idx]
        entry = self.data[clip_idx]
        
        # Load features and masks
        npy = np.load(entry['npy_path'])     # (T, N, F)
        mask = np.load(entry['mask_path'])   # (T, N)
        
        # Extract context window
        context_feat = torch.FloatTensor(
            npy[start:start+self.context_len]
        )  # (context_len, N, F)
        context_mask = torch.BoolTensor(
            mask[start:start+self.context_len]
        )  # (context_len, N)
        
        # Extract target window and format as trajectory text
        target_npy = npy[start+self.context_len : start+self.context_len+self.K]
        target_xy = target_npy[:, :, :2]  # (K, N, 2)
        target_text = format_trajectory(target_xy, self.K)
        
        # Get sequence ID
        seq_id = entry.get('seq_id', entry.get('npy_path', str(clip_idx)))
        
        return context_feat, context_mask, target_text, seq_id, start
