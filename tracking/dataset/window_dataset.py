import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

try:
    from .multitask_dataset import ACTION_NAMES_EN
except ImportError:
    from multitask_dataset import ACTION_NAMES_EN

class WindowDataset(Dataset):
    def __init__(self, json_path, window_size=2, n_frames=30, max_games=0):
        self.window_size = window_size
        self.n_frames = n_frames
        self.base_dir = Path(json_path).parent
        self.samples = []
        # Load clips.json
        with open(json_path, 'r') as f:
            clips = json.load(f)
        # Track unique game_ids if max_games > 0
        unique_games = set()
        for entry in clips:
            game_id = entry.get('game_id')
            if max_games > 0:
                if game_id not in unique_games:
                    if len(unique_games) >= max_games:
                        continue
                    unique_games.add(game_id)
                elif game_id not in unique_games:
                    continue
            # Check for action_sequence_frames
            if 'action_sequence_frames' not in entry:
                print(f"[WindowDataset] Warning: entry missing 'action_sequence_frames', skipping. id={entry.get('id')}")
                continue
            if not entry.get('action_sequence'):
                continue
            if len(entry['action_sequence']) != len(entry['action_sequence_frames']):
                print(f"[WindowDataset] Warning: action_sequence and action_sequence_frames length mismatch, skipping. id={entry.get('id')}")
                continue
            for i, (action_id, center_frame) in enumerate(zip(entry['action_sequence'], entry['action_sequence_frames'])):
                self.samples.append((entry, i, action_id, center_frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, action_idx, action_id, center_frame = self.samples[idx]
        W = 2 * self.window_size + 1
        npy_path = self.base_dir / entry['npy_path']
        mask_path = self.base_dir / entry['mask_path']
        tracking = np.load(str(npy_path))  # (30, 23, 5) float32
        mask = np.load(str(mask_path))    # (30, 23) bool
        start = max(0, center_frame - self.window_size)
        end = min(self.n_frames, center_frame + self.window_size + 1)
        win_track = np.zeros((W, 23, 5), dtype=np.float32)
        win_mask = np.ones((W, 23), dtype=bool)  # True = missing
        src = tracking[start:end]
        src_mask = mask[start:end]
        offset = (center_frame - self.window_size) - start
        if offset < 0:
            offset = 0
        win_track[offset:offset + len(src)] = src
        win_mask[offset:offset + len(src)] = src_mask
        action_text = ACTION_NAMES_EN.get(str(action_id), str(action_id))
        return (
            torch.tensor(win_track, dtype=torch.float32),
            torch.tensor(win_mask),
            action_text,
        )
