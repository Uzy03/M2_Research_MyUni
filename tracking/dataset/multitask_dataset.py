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
ACTION_VOCAB = sorted(set(ACTION_NAMES_EN.values()))
_ACTION_VOCAB_STR = ", ".join(ACTION_VOCAB)

TASKS = [
    {
        "name": "action",
        "instruction": (
            "List the soccer actions in this tracking sequence as comma-separated keywords. "
            f"Use only: {_ACTION_VOCAB_STR}."
        ),
        "short_instruction": f"Soccer actions (comma-separated): {_ACTION_VOCAB_STR}.",
        "label_field": "label_action",
        "max_new_tokens": 80,
    },
    {
        "name": "possession",
        "instruction": (
            "Which team has ball possession in this sequence? "
            "Answer with exactly one of: "
            "'The home team has ball possession in this sequence.' or "
            "'The away team has ball possession in this sequence.'"
        ),
        "short_instruction": "Ball possession? Home or away?",
        "label_field": "label_possession",
        "max_new_tokens": 40,
    },
    {
        "name": "zone",
        "instruction": (
            "Where on the field is the play occurring? "
            "Answer using the template: 'The play is in the {side} of the {third}.' "
            "Side: left side / center / right side. "
            "Third: defensive third / middle third / attacking third."
        ),
        "short_instruction": (
            "Field zone? Side: left side/center/right side. "
            "Third: defensive/middle/attacking third."
        ),
        "label_field": "label_zone",
        "max_new_tokens": 40,
    },
    {
        "name": "pressure",
        "instruction": (
            "Describe the pressing intensity in this sequence. "
            "Answer with exactly one of: "
            "'The players are applying high pressure around the ball.' / "
            "'The players are applying medium pressure around the ball.' / "
            "'There is low pressure around the ball.'"
        ),
        "short_instruction": "Pressure level? High/medium/low.",
        "label_field": "label_pressure",
        "max_new_tokens": 40,
    },
]

_FALLBACK_PRESSURE = "There is low pressure around the ball."


class MultiTaskDataset(Dataset):
    def __init__(self, json_path, context_len=20, max_games=0, use_short_instruction=False,
                 allowed_tasks=None):
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
        self.use_short_instruction = use_short_instruction
        if allowed_tasks is None:
            self._tasks = TASKS
        else:
            self._tasks = [t for t in TASKS if t['name'] in allowed_tasks]
            if not self._tasks:
                raise ValueError(f"allowed_tasks={allowed_tasks!r} に一致するタスクがありません")

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

        instr_key = 'short_instruction' if self.use_short_instruction else 'instruction'
        for task in random.sample(self._tasks, len(self._tasks)):
            answer = entry.get(task['label_field'])
            if answer:
                return feat, msk, task[instr_key], answer, task['name'], seq_id

        # Fallback: self._tasks[0] (allowed_tasks の先頭 = 通常 action)
        fallback_task = self._tasks[0]
        answer = entry.get(fallback_task['label_field'], '')
        return feat, msk, fallback_task[instr_key], answer, fallback_task['name'], seq_id
