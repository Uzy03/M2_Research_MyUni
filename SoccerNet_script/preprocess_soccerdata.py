#!/usr/bin/env python3
"""
Preprocess SoccerData CSV files into sliding-window NPY clips.

Reads tracking.csv, play.csv, and players.csv from SoccerData directory,
generates sliding-window clips in NPY format with metadata.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


SRC_FPS = 25  # Source video FPS


def normalize_position(x_cm: float, y_cm: float) -> Tuple[float, float]:
    """
    Normalize position from cm units to [0, 1] range.
    
    Field center = origin (0, 0) in source coordinates.
    x_norm = (X/100 + 52.5) / 105.0
    y_norm = (Y/100 + 34.0) / 68.0
    """
    x_norm = (x_cm / 100.0 + 52.5) / 105.0
    y_norm = (y_cm / 100.0 + 34.0) / 68.0
    return x_norm, y_norm


def normalize_speed(speed_pixels: float, max_speed: float = 1.0) -> float:
    """Normalize speed and clip to max value."""
    return min(speed_pixels, max_speed)


def calculate_speed(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two normalized positions."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return np.sqrt(dx * dx + dy * dy)


def load_players(players_csv: Path) -> Tuple[List[int], List[int]]:
    """
    Load players.csv and extract home/away jersey numbers.
    
    Returns:
        (home_jerseys, away_jerseys) - both sorted ascending
    """
    try:
        df = pd.read_csv(players_csv, encoding='utf-8')
        
        # Look for Japanese column headers
        # "ホームアウェイF" for team, "背番号" for jersey number
        team_col = None
        jersey_col = None
        
        for col in df.columns:
            if 'ホーム' in col or 'アウェイ' in col:
                team_col = col
            if '背番号' in col or 'Jersey' in col or '番号' in col:
                jersey_col = col
        
        if team_col is None or jersey_col is None:
            # Fallback: use first 2 columns
            if len(df.columns) >= 2:
                team_col = df.columns[0]
                jersey_col = df.columns[1]
            else:
                return [], []
        
        # スタメン列があればスタメン(1)のみ使用、なければ全選手から先頭11人
        starter_col = None
        for col in df.columns:
            if 'スタメン' in col or 'starter' in col.lower():
                starter_col = col
        
        if starter_col is not None:
            home_df = df[(df[team_col] == 1) & (df[starter_col] == 1)]
            away_df = df[(df[team_col] == 2) & (df[starter_col] == 1)]
        else:
            home_df = df[df[team_col] == 1]
            away_df = df[df[team_col] == 2]
        
        home_jerseys = sorted(home_df[jersey_col].dropna().astype(int).unique().tolist())[:11]
        away_jerseys = sorted(away_df[jersey_col].dropna().astype(int).unique().tolist())[:11]
        
        return home_jerseys, away_jerseys
    except Exception as e:
        print(f"Warning: Failed to load players.csv: {e}")
        return [], []


def load_tracking_data(tracking_csv: Path) -> pd.DataFrame:
    """Load tracking.csv with error handling."""
    try:
        df = pd.read_csv(tracking_csv, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Warning: Failed to load tracking.csv: {e}")
        return pd.DataFrame()


def load_play_data(play_csv: Path) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Load play.csv and create frame -> action mapping.
    
    Returns:
        (dataframe, frame_to_action_dict)
    """
    try:
        df = pd.read_csv(play_csv, encoding='utf-8')
        
        # Find frame and action columns (Japanese headers)
        frame_col = None
        action_col = None
        
        for col in df.columns:
            if 'フレーム' in col or 'Frame' in col:
                frame_col = col
            if 'アクション' in col or 'Action' in col:
                action_col = col
        
        if frame_col is None or action_col is None:
            # Fallback: use first 2 columns
            if len(df.columns) >= 2:
                frame_col = df.columns[0]
                action_col = df.columns[1]
            else:
                return df, {}
        
        frame_to_action = {}
        for idx, row in df.iterrows():
            frame_num = int(row[frame_col])
            action = str(row[action_col]) if pd.notna(row[action_col]) else ""
            frame_to_action[frame_num] = (action, idx)
        
        return df, frame_to_action
    except Exception as e:
        print(f"Warning: Failed to load play.csv: {e}")
        return pd.DataFrame(), {}


SKIP_ACTION_IDS = {'1','2','3','4','19','25','26','40'}

def get_action_sequence(
    start_frame: int,
    end_frame: int,
    frame_to_action: Dict[int, Tuple[str, int]],
) -> List[str]:
    actions_in_window = [
        (frame, action_id)
        for frame, (action_id, _) in frame_to_action.items()
        if start_frame <= frame <= end_frame and action_id not in SKIP_ACTION_IDS
    ]
    actions_in_window.sort(key=lambda x: x[0])
    result = []
    prev = None
    for _, action_id in actions_in_window:
        if action_id != prev:
            result.append(action_id)
            prev = action_id
    return result


def get_nearest_action(
    frame_num: int,
    frame_to_action: Dict[int, Tuple[str, int]],
    search_range: int = 250
) -> Tuple[str, int]:
    """
    Get nearest action within search_range frames.
    
    Returns:
        (action_label, action_id) or ("", -1) if none found
    """
    if not frame_to_action:
        return "", -1
    
    min_dist = float('inf')
    best_action = ""
    best_action_id = -1
    
    for frame, (action, action_id) in frame_to_action.items():
        dist = abs(frame - frame_num)
        if dist <= search_range and dist < min_dist:
            min_dist = dist
            best_action = action
            best_action_id = action_id
    
    if min_dist == float('inf'):
        return "", -1
    
    return best_action, best_action_id


def create_clip(
    frame_lookup: Dict,
    sampled_frames: List[int],
    home_jerseys: List[int],
    away_jerseys: List[int],
    n_frames: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create NPY clip from sampled frames.
    
    Returns:
        (clip_array, mask_array, ball_coverage)
        clip_array: shape (n_frames, 23, 5), dtype float32
        mask_array: shape (n_frames, 23), dtype bool (True=missing, False=valid)
        ball_coverage: float (0.0-1.0)
    """
    clip = np.zeros((n_frames, 23, 5), dtype=np.float32)
    mask = np.ones((n_frames, 23), dtype=bool)  # True = missing
    
    ball_frames = 0
    
    for frame_idx, frame_num in enumerate(sampled_frames):
        frame_data = frame_lookup.get(frame_num, None)
        if frame_data is None:
            # Frame not in tracking data (e.g., halftime gap) - keep mask as True
            continue
        
        # Slot 0: Ball
        ball_data = frame_data[(frame_data['SysTarget'] == 0) | (frame_data['SysTarget'] == 7)]
        
        if not ball_data.empty:
            ball_frames += 1
            row = ball_data.iloc[0]
            x_norm, y_norm = normalize_position(row['X'], row['Y'])
            speed = row.get('Speed', 0.0)
            speed_norm = normalize_speed(speed)
            
            clip[frame_idx, 0] = [x_norm, y_norm, speed_norm, 0, 1]
            mask[frame_idx, 0] = False
        
        # Slots 1-11: Home players (sorted by jersey)
        for slot, jersey in enumerate(home_jerseys, start=1):
            player_data = frame_data[
                (frame_data['HA'] == 1) &
                (frame_data['No'] == jersey) &
                (~frame_data['SysTarget'].isin([0, 7]))
            ]
            
            if not player_data.empty:
                row = player_data.iloc[0]
                x_norm, y_norm = normalize_position(row['X'], row['Y'])
                speed = row.get('Speed', 0.0)
                speed_norm = normalize_speed(speed)
                
                clip[frame_idx, slot] = [x_norm, y_norm, speed_norm, 1, 0]
                mask[frame_idx, slot] = False
        
        # Slots 12-22: Away players (sorted by jersey)
        for slot, jersey in enumerate(away_jerseys, start=12):
            player_data = frame_data[
                (frame_data['HA'] == 2) &
                (frame_data['No'] == jersey) &
                (~frame_data['SysTarget'].isin([0, 7]))
            ]
            
            if not player_data.empty:
                row = player_data.iloc[0]
                x_norm, y_norm = normalize_position(row['X'], row['Y'])
                speed = row.get('Speed', 0.0)
                speed_norm = normalize_speed(speed)
                
                clip[frame_idx, slot] = [x_norm, y_norm, speed_norm, 2, 0]
                mask[frame_idx, slot] = False
    
    ball_coverage = ball_frames / n_frames if n_frames > 0 else 0.0
    
    # Calculate speeds between consecutive sampled frames
    for frame_idx in range(n_frames):
        for slot in range(23):
            if not mask[frame_idx, slot]:  # Valid data
                if frame_idx == 0:
                    # First frame speed = 0.0
                    clip[frame_idx, slot, 2] = 0.0
                else:
                    # Speed = distance from previous sampled frame
                    if not mask[frame_idx - 1, slot]:  # Previous frame also valid
                        speed = calculate_speed(
                            (clip[frame_idx - 1, slot, 0], clip[frame_idx - 1, slot, 1]),
                            (clip[frame_idx, slot, 0], clip[frame_idx, slot, 1])
                        )
                        clip[frame_idx, slot, 2] = normalize_speed(speed)
    
    return clip, mask, ball_coverage


def process_game(
    game_dir: Path,
    output_dir: Path,
    game_id: str,
    args
) -> List[Dict]:
    """
    Process a single game directory.
    
    Returns:
        List of clip metadata dicts
    """
    try:
        # Load CSV files
        players_csv = game_dir / "players.csv"
        tracking_csv = game_dir / "tracking.csv"
        play_csv = game_dir / "play.csv"
        
        if not all([f.exists() for f in [players_csv, tracking_csv, play_csv]]):
            return []
        
        home_jerseys, away_jerseys = load_players(players_csv)
        if not home_jerseys or not away_jerseys:
            return []
        
        tracking_df = load_tracking_data(tracking_csv)
        if tracking_df.empty:
            return []
        
        play_df, frame_to_action = load_play_data(play_csv)
        
        # Calculate processing constants
        step_raw = SRC_FPS // args.fps
        window_raw = args.sec * SRC_FPS
        slide_raw = args.step_sec * SRC_FPS
        n_frames = args.fps * args.sec
        
        # Create output directory for this game
        npy_game_dir = output_dir / "npy" / game_id
        npy_game_dir.mkdir(parents=True, exist_ok=True)
        
        clips_metadata = []
        clip_num = 0
        
        # Get valid frame range
        min_frame = tracking_df['Frame'].min()
        max_frame = tracking_df['Frame'].max()
        
        # Build frame lookup dict for O(1) access (critical for performance)
        frame_lookup = {
            frame: group.reset_index(drop=True)
            for frame, group in tracking_df.groupby('Frame')
        }
        
        # Generate sliding windows
        start_raw = min_frame
        while start_raw + window_raw <= max_frame:
            sampled_frames = [
                start_raw + i * step_raw
                for i in range(n_frames)
            ]
            
            clip_array, mask_array, ball_coverage = create_clip(
                frame_lookup,
                sampled_frames,
                home_jerseys,
                away_jerseys,
                n_frames
            )
            
            # Check ball coverage threshold
            if ball_coverage >= args.ball_cov:
                # Save NPY files
                clip_path = npy_game_dir / f"clip_{clip_num:04d}.npy"
                mask_path = npy_game_dir / f"clip_{clip_num:04d}_mask.npy"
                
                np.save(clip_path, clip_array)
                np.save(mask_path, mask_array)
                
                # Get action information
                action_label, action_id = get_nearest_action(start_raw, frame_to_action)
                end_raw = start_raw + window_raw
                action_sequence = get_action_sequence(start_raw, end_raw, frame_to_action)
                start_sec = start_raw / SRC_FPS
                
                # Create metadata entry
                npy_rel_path = str(Path("npy") / game_id / f"clip_{clip_num:04d}.npy")
                mask_rel_path = str(Path("npy") / game_id / f"clip_{clip_num:04d}_mask.npy")
                
                clip_metadata = {
                    "game_id": game_id,
                    "clip_id": f"{game_id}_{clip_num:04d}",
                    "start_frame_orig": int(start_raw),
                    "start_sec": float(start_sec),
                    "action_label": action_label,
                    "action_id": int(action_id),
                    "action_sequence": action_sequence,
                    "ball_coverage": float(ball_coverage),
                    "npy_path": npy_rel_path,
                    "mask_path": mask_rel_path,
                }
                
                clips_metadata.append(clip_metadata)
                clip_num += 1
            
            start_raw += slide_raw
        
        return clips_metadata
    
    except Exception as e:
        print(f"Error processing game {game_id}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SoccerData CSV files into sliding-window NPY clips"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/ujihura/m2_研究/SoccerData",
        help="Path to SoccerData directory containing 2023_data/ and 2024_data/"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="soccerdata_clips",
        help="Output directory for clips"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="fps1_sec30_onball_step5s",
        help="Configuration name for output subdirectory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Target FPS for sampled frames"
    )
    parser.add_argument(
        "--sec",
        type=int,
        default=30,
        help="Clip duration in seconds"
    )
    parser.add_argument(
        "--step_sec",
        type=int,
        default=5,
        help="Sliding window step in seconds"
    )
    parser.add_argument(
        "--ball_cov",
        type=float,
        default=0.7,
        help="Minimum ball coverage threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=0,
        help="Maximum number of games to process (0 = all games)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.out_dir) / args.config
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    
    # Collect all game directories from both seasons
    all_clips_metadata = []
    all_game_dirs = []
    
    for season_dir in ["2023_data", "2024_data"]:
        season_path = data_dir / season_dir
        if not season_path.exists():
            print(f"Warning: {season_path} not found, skipping")
            continue
        game_dirs = sorted([
            d for d in season_path.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 10
        ])
        all_game_dirs.extend(game_dirs)
    
    # Apply max_games limit
    if args.max_games > 0:
        all_game_dirs = all_game_dirs[:args.max_games]

    # Load existing clips.json to find already-processed games
    clips_json_path = output_dir / "clips.json"
    if clips_json_path.exists():
        with open(clips_json_path, encoding='utf-8') as f:
            all_clips_metadata = json.load(f)
        processed_games = {e['game_id'] for e in all_clips_metadata}
        print(f"Already processed: {len(processed_games)} games ({len(all_clips_metadata)} clips)")
    else:
        processed_games = set()

    pending = [d for d in all_game_dirs if d.name not in processed_games]
    print(f"Processing {len(pending)} new games (skipping {len(all_game_dirs) - len(pending)} already done)")

    for game_dir in tqdm(pending, desc="Processing games"):
        game_id = game_dir.name
        clips_metadata = process_game(game_dir, output_dir, game_id, args)
        all_clips_metadata.extend(clips_metadata)
        with open(clips_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_clips_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal clips: {len(all_clips_metadata)}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
