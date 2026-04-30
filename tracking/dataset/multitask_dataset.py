import json
import random
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

TASKS = [
    {
        "name": "action",
        "instruction": "List the soccer actions occurring in this tracking sequence in chronological order.",
    },
    {
        "name": "possession",
        "instruction": "Which team has ball possession in this sequence?",
    },
    {
        "name": "zone",
        "instruction": "Where on the field is the play occurring in this sequence?",
    },
    {
        "name": "pressure",
        "instruction": "Describe the pressing intensity in this sequence.",
    },
]


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

    # --- label generators ---

    def _action_label(self, entry):
        names = [ACTION_NAMES_EN[a] for a in entry.get('action_sequence', []) if a in ACTION_NAMES_EN]
        return ', '.join(dict.fromkeys(names)) if names else None

    def _possession_label(self, npy, mask):
        # player 0 = ball, 1-11 = home, 12-22 = away
        ball_valid = ~mask[:, 0]
        valid_frames = np.where(ball_valid)[0]
        if len(valid_frames) == 0:
            return None
        t = valid_frames[-1]
        ball_pos = npy[t, 0, :2]
        min_dist, closest_n = float('inf'), -1
        for n in range(1, npy.shape[1]):
            if mask[t, n]:
                continue
            d = np.linalg.norm(npy[t, n, :2] - ball_pos)
            if d < min_dist:
                min_dist, closest_n = d, n
        if closest_n == -1:
            return None
        team = "home" if closest_n <= 11 else "away"
        return f"The {team} team has ball possession in this sequence."

    def _zone_label(self, npy, mask):
        ball_valid = ~mask[:, 0]
        valid_frames = np.where(ball_valid)[0]
        if len(valid_frames) == 0:
            return None
        t = valid_frames[-1]
        x, y = npy[t, 0, 0], npy[t, 0, 1]
        x_zone = "defensive third" if x < 0.33 else ("middle third" if x < 0.67 else "attacking third")
        y_zone = "left side" if y < 0.33 else ("center" if y < 0.67 else "right side")
        return f"The play is in the {y_zone} of the {x_zone}."

    def _pressure_label(self, npy, mask):
        ball_valid = ~mask[:, 0]
        valid_frames = np.where(ball_valid)[0]
        if len(valid_frames) == 0:
            return "low pressure"
        recent = valid_frames[-5:]
        speeds = []
        for t in recent:
            ball_pos = npy[t, 0, :2]
            for n in range(1, npy.shape[1]):
                if mask[t, n]:
                    continue
                if np.linalg.norm(npy[t, n, :2] - ball_pos) < 0.1:
                    speeds.append(npy[t, n, 2])
        if not speeds:
            return "There is low pressure around the ball."
        avg = float(np.mean(speeds))
        if avg > 0.5:
            return "The players are applying high pressure around the ball."
        elif avg > 0.2:
            return "The players are applying medium pressure around the ball."
        else:
            return "There is low pressure around the ball."

    def __getitem__(self, idx):
        entry = self.entries[idx]
        npy, mask = self._load_arrays(entry)
        feat, msk = self._make_feat(npy, mask)
        seq_id = entry.get('clip_id', str(idx))

        # Try tasks in random order until a valid label is generated
        for task in random.sample(TASKS, len(TASKS)):
            if task['name'] == 'action':
                answer = self._action_label(entry)
            elif task['name'] == 'possession':
                answer = self._possession_label(npy, mask)
            elif task['name'] == 'zone':
                answer = self._zone_label(npy, mask)
            else:  # pressure always returns a value
                answer = self._pressure_label(npy, mask)
            if answer is not None:
                return feat, msk, task['instruction'], answer, task['name'], seq_id

        # Fallback (should rarely trigger)
        return feat, msk, TASKS[3]['instruction'], "low pressure", "pressure", seq_id
