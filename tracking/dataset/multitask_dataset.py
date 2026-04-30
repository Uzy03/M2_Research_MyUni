import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

TASKS = [
    {
        "name": "action",
        "instruction": "List the soccer actions occurring in this tracking sequence in chronological order.",
        "label_field": "label_action",
    },
    {
        "name": "possession",
        "instruction": "Which team has ball possession in this sequence?",
        "label_field": "label_possession",
    },
    {
        "name": "zone",
        "instruction": "Where on the field is the play occurring in this sequence?",
        "label_field": "label_zone",
    },
    {
        "name": "pressure",
        "instruction": "Describe the pressing intensity in this sequence.",
        "label_field": "label_pressure",
    },
]

_FALLBACK_PRESSURE = "There is low pressure around the ball."


class MultiTaskDataset(Dataset):
    def __init__(self, json_path, context_len=20, max_games=0):
        self.base_dir = Path(json_path).parent
        self.context_len = context_len
        with open(json_path) as f:
            data = json.load(f)
        if max_games > 0:
            seen, allowed = [], set()
            for e in data:
                if e['game_id'] not in allowed:
                    seen.append(e['game_id'])
                    if len(seen) > max_games:
                        break
                    allowed.add(e['game_id'])
            data = [e for e in data if e['game_id'] in allowed]
        self.entries = data

    def __len__(self):
        return len(self.entries)

    def _load_arrays(self, entry):
        npy  = np.load(self.base_dir / entry['npy_path'])   # (T, N, F)
        mask = np.load(self.base_dir / entry['mask_path'])  # (T, N)
        return npy, mask

    def _make_feat(self, npy, mask):
        T = npy.shape[0]
        if T >= self.context_len:
            feat_np = npy[-self.context_len:]
            mask_np = mask[-self.context_len:]
        else:
            pad = self.context_len - T
            feat_np = np.concatenate(
                [np.zeros((pad, npy.shape[1], npy.shape[2]), dtype=np.float32), npy], axis=0
            )
            mask_np = np.concatenate(
                [np.ones((pad, mask.shape[1]), dtype=bool), mask], axis=0
            )
        return torch.FloatTensor(feat_np), torch.BoolTensor(mask_np)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        npy, mask = self._load_arrays(entry)
        feat, msk = self._make_feat(npy, mask)
        seq_id = entry.get('clip_id', str(idx))

        for task in random.sample(TASKS, len(TASKS)):
            answer = entry.get(task['label_field'])
            if answer:
                return feat, msk, task['instruction'], answer, task['name'], seq_id

        # Fallback: pressure label is always non-None after add_task_labels
        pressure_task = TASKS[3]
        answer = entry.get(pressure_task['label_field'], _FALLBACK_PRESSURE)
        return feat, msk, pressure_task['instruction'], answer, pressure_task['name'], seq_id
