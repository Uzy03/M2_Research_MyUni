#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from tracking.compute_spatial_labels import determine_possession


def _zone_counts(x_coords, y_coords):
    zones = {
        "def_left": 0, "def_center": 0, "def_right": 0,
        "mid_left": 0, "mid_center": 0, "mid_right": 0,
        "att_left": 0, "att_center": 0, "att_right": 0,
    }
    for x, y in zip(x_coords, y_coords):
        if x < 0.33:
            depth = "def"
        elif x < 0.67:
            depth = "mid"
        else:
            depth = "att"
        if y < 0.33:
            width = "left"
        elif y < 0.67:
            width = "center"
        else:
            width = "right"
        zones[f"{depth}_{width}"] += 1
    return zones


def _team_stats(data, mask, slots, team_flag):
    n_players = len(slots)
    all_x, all_y = [], []
    for slot in slots:
        valid = ~mask[:, slot]
        if valid.sum() > 0:
            all_x.append(float(data[valid, slot, 0].mean()))
            all_y.append(float(data[valid, slot, 1].mean()))

    if not all_x:
        return None

    x_arr = np.array(all_x)
    y_arr = np.array(all_y)

    if team_flag == 2:
        depth_arr = 1.0 - x_arr
    else:
        depth_arr = x_arr.copy()

    centroid = [float(x_arr.mean()), float(y_arr.mean())]
    spread_width = float(y_arr.std())
    spread_depth = float(depth_arr.std())
    zone_counts = _zone_counts(depth_arr, y_arr)

    return {
        "centroid": centroid,
        "spread_width": spread_width,
        "spread_depth": spread_depth,
        "zone_counts": zone_counts,
    }


def process_clip(clip_entry, base_dir):
    npy_path = Path(base_dir) / clip_entry['npy_path']
    mask_path = Path(base_dir) / clip_entry['mask_path']
    if not npy_path.exists() or not mask_path.exists():
        return None

    data = np.load(str(npy_path))
    mask = np.load(str(mask_path))

    possession_team = determine_possession(data, mask)
    attack_team = possession_team
    defend_team = 3 - possession_team

    attack_slots = list(range(1, 12)) if attack_team == 1 else list(range(12, 23))
    defend_slots = list(range(1, 12)) if defend_team == 1 else list(range(12, 23))

    team_attack = _team_stats(data, mask, attack_slots, attack_team)
    team_defend = _team_stats(data, mask, defend_slots, defend_team)

    return {
        "team_attack": team_attack,
        "team_defend": team_defend,
    }


def format_stats_text(stats_entry: dict) -> str:
    """stats_entry（clip1件分の辞書）をプロンプト注入用テキストに変換する。

    Returns:
        str: "[Tracking Statistics]\n..." 形式のテキスト
    """
    lines = ["[Tracking Statistics]"]
    for role, key in [("Attacking team", "team_attack"), ("Defending team", "team_defend")]:
        s = stats_entry.get(key)
        if s is None:
            lines.append(f"{role}: N/A")
            continue
        cx, cy = s["centroid"]
        sw = s["spread_width"]
        sd = s["spread_depth"]
        lines.append(f"{role}: centroid=({cx:.2f}, {cy:.2f}), spread W={sw:.2f} D={sd:.2f}")
        zc = s["zone_counts"]
        zone_str = " ".join(f"{k}={v}" for k, v in zc.items())
        lines.append(f"  Zones: {zone_str}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute tracking statistics from NPY clips")
    parser.add_argument("--clips_json", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--out_json", default="tracking_stats.json")
    parser.add_argument("--max_games", type=int, default=0)
    args = parser.parse_args()

    with open(args.clips_json) as f:
        clips = json.load(f)

    if isinstance(clips, dict):
        clips = list(clips.values())

    if args.max_games > 0:
        seen_games, filtered = [], []
        for c in clips:
            gid = c.get('game_id', '')
            if gid not in seen_games:
                seen_games.append(gid)
            if len(seen_games) <= args.max_games:
                filtered.append(c)
        clips = filtered

    results = {}
    for clip_entry in tqdm(clips, desc="Computing stats"):
        clip_id = clip_entry["clip_id"]
        result = process_clip(clip_entry, args.base_dir)
        results[clip_id] = result if result is not None else {"team_attack": None, "team_defend": None}

    out_path = Path(args.out_json)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved tracking stats to {out_path} ({len(results)} clips)")


if __name__ == "__main__":
    main()
