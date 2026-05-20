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


def _action_to_sentence(label: str) -> str:
    actions = [a.strip() for a in label.split(',') if a.strip()]
    if not actions:
        return ''
    if len(actions) == 1:
        return f'In this soccer sequence, performing {actions[0]}.'
    return f'In this soccer sequence, performing {", ".join(actions[:-1])} and {actions[-1]}.'


_ANSWER_TEMPLATES = [
    # variant 0（既存フォーマット）
    lambda actions: (
        f'In this soccer sequence, performing {actions[0]}.'
        if len(actions) == 1
        else f'In this soccer sequence, performing {", ".join(actions[:-1])} and {actions[-1]}.'
    ),
    # variant 1
    lambda actions: (
        f'The action observed is: {actions[0]}.'
        if len(actions) == 1
        else f'The actions observed are: {", ".join(actions)}.'
    ),
    # variant 2
    lambda actions: (
        f'{actions[0].capitalize()} was performed in this sequence.'
        if len(actions) == 1
        else f'{", ".join(a.capitalize() for a in actions[:-1])} and {actions[-1].capitalize()} were performed.'
    ),
    # variant 3
    lambda actions: (
        f'This sequence shows {actions[0]}.'
        if len(actions) == 1
        else f'This sequence shows {" followed by ".join(actions)}.'
    ),
    # variant 4
    lambda actions: (
        f'The players executed {actions[0]}.'
        if len(actions) == 1
        else f'The players executed {" and ".join(actions)}.'
    ),
    # variant 5
    lambda actions: f'Soccer actions: {" ".join(actions)}.'
]


def _action_to_sentence_variant(label: str, variant: int = 0) -> str:
    actions = [a.strip() for a in label.split(',') if a.strip()]
    if not actions:
        return ''
    return _ANSWER_TEMPLATES[variant % len(_ANSWER_TEMPLATES)](actions)


TASKS = [
    {
        "name": "action",
        "instruction": (
            "List the soccer actions in this tracking sequence as comma-separated keywords. "
            f"Use only: {_ACTION_VOCAB_STR}."
        ),
        "short_instruction": f"Soccer actions (comma-separated): {_ACTION_VOCAB_STR}.",
        "sentence_instruction": "Describe the soccer actions in this tracking sequence.",
        "instruction_variants": [
            'Describe the soccer actions in this tracking sequence.',
            'What actions are taking place in this soccer sequence?',
            'What is happening on the pitch in this sequence?',
            'Identify the actions occurring in this soccer tracking data.',
            'Analyze the play sequence and describe the actions.',
            'What soccer actions can be observed in this sequence?',
        ],
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
    {
        "name": "formation",
        "instruction": "What formation is the defending team using in this sequence? Answer in a complete sentence.",
        "short_instruction": "Defending team formation? Answer in a sentence.",
        "label_field": "_spatial_formation_defend",
        "max_new_tokens": 20,
    },
    {
        "name": "def_line",
        "instruction": "How high is the defending team's defensive line in this sequence? Answer in a complete sentence.",
        "short_instruction": "Defensive line height? Answer in a sentence.",
        "label_field": "_spatial_def_line_label",
        "max_new_tokens": 20,
    },
]

_FALLBACK_PRESSURE = "There is low pressure around the ball."


class MultiTaskDataset(Dataset):
    def __init__(self, json_path, context_len=20, max_games=0, use_short_instruction=False,
                 allowed_tasks=None, use_sentence_format=False, use_instruction_diverse=False,
                 use_answer_diverse=False, use_llm_qa=False, spatial_labels_path=None):
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
        self.use_sentence_format = use_sentence_format
        self.use_instruction_diverse = use_instruction_diverse
        self.use_answer_diverse = use_answer_diverse
        if allowed_tasks is None:
            self._tasks = TASKS
        else:
            self._tasks = [t for t in TASKS if t['name'] in allowed_tasks]
            if not self._tasks:
                raise ValueError(f"allowed_tasks={allowed_tasks!r} に一致するタスクがありません")

        # LLM-QA サンプルの構築
        self._llm_samples = []
        if use_llm_qa:
            for entry in self.entries:
                for qa in entry.get("llm_qa", []):
                    qa_type = qa.get("type", "")
                    if "turns" in qa:
                        # multi-turn conversation: human/assistant ペアを個別サンプルに展開
                        turns = qa["turns"]
                        for i in range(len(turns) - 1):
                            if (turns[i].get("from") == "human" and
                                    turns[i + 1].get("from") == "assistant" and
                                    turns[i].get("value") and turns[i + 1].get("value")):
                                self._llm_samples.append((entry, {
                                    "type": f"llm_{qa_type}",
                                    "instruction": turns[i]["value"],
                                    "answer":      turns[i + 1]["value"],
                                }))
                    elif qa.get("instruction") and qa.get("answer"):
                        # 旧形式または description/reasoning
                        self._llm_samples.append((entry, qa))

        # spatial_labels の読み込み（formation / def_line タスク用）
        self.spatial_labels = {}
        if spatial_labels_path and Path(spatial_labels_path).exists():
            with open(spatial_labels_path) as f:
                self.spatial_labels = json.load(f)
            print(f"MultiTaskDataset: loaded spatial labels for {len(self.spatial_labels)} clips")

        self._n_base = len(self.entries)

    def __len__(self):
        return self._n_base + len(self._llm_samples)

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
        if idx >= self._n_base:
            # LLM-QA サンプル
            entry, qa = self._llm_samples[idx - self._n_base]
            npy, mask = self._load_arrays(entry)
            feat, msk = self._make_feat(npy, mask)
            seq_id = entry.get("clip_id", str(idx))
            slot_labels = ["", "", ""]
            return feat, msk, qa["instruction"], qa["answer"], f"llm_{qa.get('type', 'qa')}", seq_id, slot_labels

        entry = self.entries[idx]
        npy, mask = self._load_arrays(entry)
        feat, msk = self._make_feat(npy, mask)
        seq_id = entry.get('clip_id', str(idx))

        # spatial ラベルをエントリに注入（formation / def_line タスク用）
        if self.spatial_labels and seq_id in self.spatial_labels:
            s = self.spatial_labels[seq_id]
            entry = dict(entry)  # シャローコピーして元データを保護
            fd = s.get('formation_defend')
            dl = s.get('def_line_label')
            entry['_spatial_formation_defend'] = f"The defending team's formation is {fd}." if fd else None
            entry['_spatial_def_line_label'] = f"The defending team's defensive line height is {dl}." if dl else None

        instr_key = 'short_instruction' if self.use_short_instruction else 'instruction'
        for task in random.sample(self._tasks, len(self._tasks)):
            answer = entry.get(task['label_field'])
            if answer:
                if self.use_sentence_format and task['name'] == 'action':
                    if self.use_answer_diverse:
                        variant = random.randint(0, len(_ANSWER_TEMPLATES) - 1)
                        answer = _action_to_sentence_variant(answer, variant)
                    else:
                        answer = _action_to_sentence(answer)
                    if self.use_instruction_diverse and 'instruction_variants' in task:
                        instruction = random.choice(task['instruction_variants'])
                        raw_action = entry.get('label_action') or ''
                        if self.use_sentence_format and raw_action:
                            raw_action = _action_to_sentence(raw_action)
                        slot_labels = [
                            raw_action,
                            entry.get('label_zone') or '',
                            entry.get('label_pressure') or '',
                        ]
                        return feat, msk, instruction, answer, task['name'], seq_id, slot_labels
                    instr_key_used = 'sentence_instruction'
                else:
                    instr_key_used = instr_key
                raw_action = entry.get('label_action') or ''
                if self.use_sentence_format and raw_action:
                    raw_action = _action_to_sentence(raw_action)
                slot_labels = [
                    raw_action,
                    entry.get('label_zone') or '',
                    entry.get('label_pressure') or '',
                ]
                return feat, msk, task[instr_key_used], answer, task['name'], seq_id, slot_labels

        # Fallback: self._tasks[0] (allowed_tasks の先頭 = 通常 action)
        fallback_task = self._tasks[0]
        answer = entry.get(fallback_task['label_field']) or ''
        if self.use_sentence_format and fallback_task['name'] == 'action' and answer:
            if self.use_answer_diverse:
                variant = random.randint(0, len(_ANSWER_TEMPLATES) - 1)
                answer = _action_to_sentence_variant(answer, variant)
            else:
                answer = _action_to_sentence(answer)
            if self.use_instruction_diverse and 'instruction_variants' in fallback_task:
                instruction = random.choice(fallback_task['instruction_variants'])
                raw_action = entry.get('label_action') or ''
                if self.use_sentence_format and raw_action:
                    raw_action = _action_to_sentence(raw_action)
                slot_labels = [
                    raw_action,
                    entry.get('label_zone') or '',
                    entry.get('label_pressure') or '',
                ]
                return feat, msk, instruction, answer, fallback_task['name'], seq_id, slot_labels
            instr_key_used = 'sentence_instruction'
        else:
            instr_key_used = instr_key
        raw_action = entry.get('label_action') or ''
        if self.use_sentence_format and raw_action:
            raw_action = _action_to_sentence(raw_action)
        slot_labels = [
            raw_action,
            entry.get('label_zone') or '',
            entry.get('label_pressure') or '',
        ]
        return feat, msk, fallback_task[instr_key_used], answer, fallback_task['name'], seq_id, slot_labels
