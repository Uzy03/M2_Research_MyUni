#!/usr/bin/env python3
"""Add pre-computed task labels to clips.json in-place."""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

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


def action_label(entry):
    names = [ACTION_NAMES_EN[a] for a in entry.get('action_sequence', []) if a in ACTION_NAMES_EN]
    return ', '.join(dict.fromkeys(names)) if names else None


def possession_label(npy, mask):
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
        d = float(np.linalg.norm(npy[t, n, :2] - ball_pos))
        if d < min_dist:
            min_dist, closest_n = d, n
    if closest_n == -1:
        return None
    team = "home" if closest_n <= 11 else "away"
    return f"The {team} team has ball possession in this sequence."


def zone_label(npy, mask):
    ball_valid = ~mask[:, 0]
    valid_frames = np.where(ball_valid)[0]
    if len(valid_frames) == 0:
        return None
    t = valid_frames[-1]
    x, y = float(npy[t, 0, 0]), float(npy[t, 0, 1])
    x_zone = "defensive third" if x < 0.33 else ("middle third" if x < 0.67 else "attacking third")
    y_zone = "left side" if y < 0.33 else ("center" if y < 0.67 else "right side")
    return f"The play is in the {y_zone} of the {x_zone}."


def pressure_label(npy, mask):
    ball_valid = ~mask[:, 0]
    valid_frames = np.where(ball_valid)[0]
    if len(valid_frames) == 0:
        return "There is low pressure around the ball."
    recent = valid_frames[-5:]
    speeds = []
    for t in recent:
        ball_pos = npy[t, 0, :2]
        for n in range(1, npy.shape[1]):
            if mask[t, n]:
                continue
            if np.linalg.norm(npy[t, n, :2] - ball_pos) < 0.1:
                speeds.append(float(npy[t, n, 2]))
    if not speeds:
        return "There is low pressure around the ball."
    avg = float(np.mean(speeds))
    if avg > 0.5:
        return "The players are applying high pressure around the ball."
    elif avg > 0.2:
        return "The players are applying medium pressure around the ball."
    return "There is low pressure around the ball."


def main():
    parser = argparse.ArgumentParser(description='Add task labels to clips.json')
    parser.add_argument('--json_path', required=True, help='Path to clips.json')
    args = parser.parse_args()

    json_path = Path(args.json_path)
    base_dir = json_path.parent

    with open(json_path) as f:
        clips = json.load(f)

    for entry in tqdm(clips, desc='Computing labels'):
        npy  = np.load(base_dir / entry['npy_path'])
        mask = np.load(base_dir / entry['mask_path'])
        entry['label_action']    = action_label(entry)
        entry['label_possession'] = possession_label(npy, mask)
        entry['label_zone']      = zone_label(npy, mask)
        entry['label_pressure']  = pressure_label(npy, mask)

    with open(json_path, 'w') as f:
        json.dump(clips, f, ensure_ascii=False)

    print(f"Done. Updated {len(clips)} entries in {json_path}")


if __name__ == '__main__':
    main()
