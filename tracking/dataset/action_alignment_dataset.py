import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

ACTION_NAMES_EN = {
    '15': 'shot', '16': 'goalkeeper save', '17': 'direct free kick',
    '21': 'corner kick', '22': 'indirect free kick', '23': 'offside',
    '27': 'foul', '29': 'pass', '30': 'pass',
    '35': 'dribble', '36': 'through pass', '37': 'clearance',
    '38': 'foul received', '41': 'interception', '42': 'clearance',
    '43': 'block', '44': 'throw-in', '45': 'cross',
    '50': 'trap', '72': 'feed', '73': 'touch', '74': 'tackle',
    '75': 'flick-on',
}

class ActionAlignmentDataset(Dataset):
    def __init__(self, json_path, context_len=20):
        self.base_dir = Path(json_path).parent
        self.context_len = context_len
        with open(json_path) as f:
            data = json.load(f)
        self.entries = [e for e in data if str(e.get('action_label', '')) in ACTION_NAMES_EN]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        npy  = np.load(self.base_dir / entry['npy_path'])
        mask = np.load(self.base_dir / entry['mask_path'])
        T = npy.shape[0]
        if T >= self.context_len:
            feat_np = npy[-self.context_len:]
            mask_np = mask[-self.context_len:]
        else:
            pad = self.context_len - T
            feat_np = np.concatenate([np.zeros((pad, npy.shape[1], npy.shape[2]), dtype=np.float32), npy], axis=0)
            mask_np = np.concatenate([np.ones((pad, mask.shape[1]), dtype=bool), mask], axis=0)
        feat = torch.FloatTensor(feat_np)
        msk  = torch.BoolTensor(mask_np)
        action_text = ACTION_NAMES_EN[str(entry['action_label'])]
        seq_id = entry.get('clip_id', str(idx))
        return feat, msk, action_text, seq_id
