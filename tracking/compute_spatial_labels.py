"""
Compute spatial labels (formation and defensive line height) from tracking data.

This script reads NPY files containing tracking data and computes:
1. Formation for each team (attack and defense perspective)
2. Defensive line height for the defending team

Ground truth labels for LLM-as-a-Judge validation.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def load_tracking_data(npy_path, mask_path):
    """Load tracking data and mask.
    
    Args:
        npy_path: Path to NPY file with shape (T, N, 5)
                  Each slot: [x, y, speed, team_flag, is_ball]
                  Slot 0: ball (is_ball=1)
                  Slots 1-11: team 1 (team_flag=1, x direction attack: 0->1)
                  Slots 12-22: team 2 (team_flag=2, x direction attack: 1->0)
        mask_path: Path to mask file with shape (T, N)
                   True = missing, False = valid
    
    Returns:
        data: (T, N, 5) tracking data
        mask: (T, N) bool mask
    """
    data = np.load(npy_path)
    mask = np.load(mask_path)
    return data, mask


def compute_mean_positions(data, mask):
    """Compute mean positions for each player across valid frames.
    
    Args:
        data: (T, N, 5) tracking data
        mask: (T, N) bool mask
    
    Returns:
        mean_x: (N,) mean x coordinate for each slot
    """
    T, N, _ = data.shape
    mean_x = np.zeros(N)
    
    for n in range(N):
        valid_frames = ~mask[:, n]
        if valid_frames.sum() > 0:
            mean_x[n] = data[valid_frames, n, 0].mean()
    
    return mean_x


def determine_possession(data, mask):
    """Determine which team has possession.
    
    Args:
        data: (T, N, 5) tracking data
        mask: (T, N) bool mask
    
    Returns:
        possession_team: 1 (team_flag=1) or 2 (team_flag=2)
    """
    # Ball is slot 0
    valid_ball = ~mask[:, 0]
    if valid_ball.sum() == 0:
        return 1
    
    ball_x = data[valid_ball, 0, 0].mean()
    
    # Team 1: slots 1-11, team_flag=1
    team1_valid = ~mask[:, 1:12].any(axis=0)
    team1_x = []
    for slot in range(1, 12):
        valid = ~mask[:, slot]
        if valid.sum() > 0:
            team1_x.append(data[valid, slot, 0].mean())
    
    # Team 2: slots 12-22, team_flag=2
    team2_valid = ~mask[:, 12:23].any(axis=0)
    team2_x = []
    for slot in range(12, 23):
        valid = ~mask[:, slot]
        if valid.sum() > 0:
            team2_x.append(data[valid, slot, 0].mean())
    
    if not team1_x or not team2_x:
        return 1
    
    team1_mean = np.mean(team1_x)
    team2_mean = np.mean(team2_x)
    
    dist_team1 = np.abs(ball_x - team1_mean)
    dist_team2 = np.abs(ball_x - team2_mean)
    
    return 1 if dist_team1 < dist_team2 else 2


def get_gk_slot(team_flag, mean_x):
    """Get GK slot for a team.
    
    Args:
        team_flag: 1 or 2
        mean_x: (N,) mean x positions
    
    Returns:
        gk_slot: slot index
    """
    if team_flag == 1:
        # Team 1 attacks high x, GK at low x (self goal)
        return np.argmin(mean_x[1:12]) + 1
    else:
        # Team 2 attacks low x, GK at high x (self goal)
        return np.argmax(mean_x[12:23]) + 12


def compute_formation(team_slots, mean_x, team_flag):
    if len(team_slots) < 5:
        return None
    x_coords = mean_x[team_slots]
    if team_flag == 2:
        x_coords = 1.0 - x_coords

    best_k, best_score = 2, -1
    for k in range(2, min(5, len(team_slots))):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(x_coords.reshape(-1, 1))
            if len(set(labels)) < k:
                continue
            score = silhouette_score(x_coords.reshape(-1, 1), labels)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_coords.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)
    counts = [int(np.sum(labels == i)) for i in sorted_indices]
    return "-".join(str(c) for c in counts)


def get_defensive_line_label(def_line_m):
    """Convert defensive line distance to qualitative label.
    
    Args:
        def_line_m: distance from goal in meters
    
    Returns:
        label: "very high", "high", "medium", "low", or "very low"
    """
    if def_line_m >= 50:
        return "very high"
    elif def_line_m >= 40:
        return "high"
    elif def_line_m >= 30:
        return "medium"
    elif def_line_m >= 20:
        return "low"
    else:
        return "very low"


def compute_defensive_line_height(team_slots, mean_x, team_flag):
    """Compute defensive line height for a team.
    
    Args:
        team_slots: list of slot indices for defending team (10 without GK)
        mean_x: (N,) mean x positions
        team_flag: 1 or 2
    
    Returns:
        def_line_m: distance from goal in meters (float)
        def_line_label: qualitative label (str)
    """
    if len(team_slots) < 5:
        return None, None
    
    x_coords = mean_x[team_slots].reshape(-1, 1)
    
    # Use K-Means to find defensive line (most "own goal" side cluster)
    best_k = 3
    best_score = -1
    
    for k in [3, 4]:
        if len(team_slots) < k:
            continue
        
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(x_coords)
            
            if len(team_slots) < 2:
                continue
            
            score = silhouette_score(x_coords, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_coords)
    centers = kmeans.cluster_centers_.flatten()
    
    if team_flag == 1:
        # Team 1 defending: own goal at x=0, defensive line is cluster with min x
        df_cluster_idx = np.argmin(centers)
        x_median = np.median(x_coords[labels == df_cluster_idx])
        def_line_m = float(x_median * 105)
    else:
        # Team 2 defending: own goal at x=1, defensive line is cluster with max x
        df_cluster_idx = np.argmax(centers)
        x_median = np.median(x_coords[labels == df_cluster_idx])
        def_line_m = float((1 - x_median) * 105)
    
    def_line_label = get_defensive_line_label(def_line_m)
    
    return def_line_m, def_line_label


def process_clip(clip_entry, base_dir):
    """Process a single clip to extract spatial labels.
    
    Args:
        clip_entry: dict with 'npy_path', 'mask_path', etc.
        base_dir: base directory for NPY files
    
    Returns:
        dict with 'formation_attack', 'formation_defend', 'def_line_m', 'def_line_label'
        or None if processing fails
    """
    npy_path = Path(base_dir) / clip_entry['npy_path']
    mask_path = Path(base_dir) / clip_entry['mask_path']
    
    if not npy_path.exists() or not mask_path.exists():
        return None
    
    try:
        data, mask = load_tracking_data(str(npy_path), str(mask_path))
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None
    
    mean_x = compute_mean_positions(data, mask)
    possession_team = determine_possession(data, mask)
    
    # Get attacking and defending teams
    attack_team = possession_team
    defend_team = 3 - possession_team  # 1->2 or 2->1
    
    # Get slots for each team (excluding GK)
    attack_slots = list(range(1, 12)) if attack_team == 1 else list(range(12, 23))
    defend_slots = list(range(1, 12)) if defend_team == 1 else list(range(12, 23))
    
    # Remove GK
    gk_idx_attack = get_gk_slot(attack_team, mean_x)
    gk_idx_defend = get_gk_slot(defend_team, mean_x)
    
    attack_slots.remove(gk_idx_attack)
    defend_slots.remove(gk_idx_defend)
    
    # Compute formations
    formation_attack = compute_formation(attack_slots, mean_x, attack_team)
    formation_defend = compute_formation(defend_slots, mean_x, defend_team)
    
    # Compute defensive line height
    def_line_m, def_line_label = compute_defensive_line_height(
        defend_slots, mean_x, defend_team
    )
    
    result = {
        "formation_attack": formation_attack,
        "formation_defend": formation_defend,
        "def_line_m": def_line_m,
        "def_line_label": def_line_label,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute spatial labels (formation + defensive line) from tracking data"
    )
    parser.add_argument("--clips_json", required=True, help="Path to clips.json")
    parser.add_argument("--base_dir", required=True, help="Base directory for NPY files")
    parser.add_argument(
        "--out_json", default="spatial_labels.json", help="Output JSON path"
    )
    parser.add_argument("--max_clips", type=int, default=0, help="Max clips to process (0=all)")
    parser.add_argument("--max_games", type=int, default=0, help="Max games to process (0=all)")
    parser.add_argument("--save_interval", type=int, default=10, help="Save results every N clips (0=disable)")
    
    args = parser.parse_args()
    
    # Load clips.json
    with open(args.clips_json) as f:
        clips = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(clips, dict):
        clips_list = list(clips.values())
    else:
        clips_list = clips
    
    if args.max_games > 0:
        seen_games = []
        filtered = []
        for c in clips_list:
            gid = c.get('game_id', '')
            if gid not in seen_games:
                seen_games.append(gid)
            if len(seen_games) <= args.max_games:
                filtered.append(c)
        clips_list = filtered

    if args.max_clips > 0:
        clips_list = clips_list[:args.max_clips]

    # 既存結果を読み込んでスキップ対象を特定
    out_path = Path(args.out_json)
    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        clips_list = [c for c in clips_list if c["clip_id"] not in results]
        print(f"Skipping {len(results)} already processed clips, {len(clips_list)} remaining")

    # Process clips
    for i, clip_entry in enumerate(tqdm(clips_list, desc="Processing clips")):
        clip_id = clip_entry["clip_id"]
        label = process_clip(clip_entry, args.base_dir)

        results[clip_id] = label if label is not None else {
            "formation_attack": None,
            "formation_defend": None,
            "def_line_m": None,
            "def_line_label": None,
        }

        if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            tqdm.write(f"[checkpoint] saved {len(results)} clips so far...")

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved spatial labels to {out_path}")
    print(f"Processed {len(results)} clips")


if __name__ == "__main__":
    main()
