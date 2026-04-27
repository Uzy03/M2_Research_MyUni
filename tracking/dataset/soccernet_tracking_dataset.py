import json
import numpy as np
import torch
from torch.utils.data import Dataset


class SoccerNetTrackingDataset(Dataset):
    """PyTorch Dataset for loading SoccerNet tracking clips with features, masks, and captions."""
    
    def __init__(self, json_path: str):
        """
        Initialize the dataset by loading JSON metadata.
        
        Args:
            json_path (str): Path to JSON file containing metadata.
        """
        with open(json_path) as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (features_tensor, mask_tensor, caption, label, seq_id, match_path)
        """
        entry = self.data[idx]
        
        features = torch.FloatTensor(np.load(entry["npy_path"]))    # (T, N, F)
        mask = torch.BoolTensor(np.load(entry["mask_path"]))        # (T, N)
        caption = entry.get("caption", "")
        label = entry.get("label", "")
        seq_id = entry.get("seq_id", "")
        match_path = entry.get("match_path", "")
        
        return features, mask, caption, label, seq_id, match_path
