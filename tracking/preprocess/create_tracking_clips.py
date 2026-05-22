import pandas as pd
import numpy as np
import json
import argparse
import os
from tqdm import tqdm
from pathlib import Path

FPS = 25
TARGET_FPS = 5
WINDOW_SEC = 15
N_FRAMES = int(WINDOW_SEC * 2 * TARGET_FPS)
N_PLAYERS = 23
N_FEATURES = 5


def load_match_data(match_dir):
    """Load play.csv and tracking.csv from match directory."""
    play_path = match_dir / 'play.csv'
    tracking_path = match_dir / 'tracking.csv'
    
    play_df = pd.read_csv(play_path, encoding='utf-8-sig')
    tracking_df = pd.read_csv(tracking_path, encoding='utf-8-sig')
    
    return play_df, tracking_df


def build_frame_index(tracking_df):
    """Build frame index for fast lookup."""
    frame_index = {}
    for frame_no, group_df in tracking_df.groupby('Frame'):
        frame_index[frame_no] = group_df
    return frame_index


def extract_frame_features(frame_index, center_frame):
    """Extract features for a 30-second window centered at center_frame."""
    step = FPS // TARGET_FPS  # = 5
    target_frames = [center_frame + i * step for i in range(-WINDOW_SEC * FPS // step, WINDOW_SEC * FPS // step)]
    
    features = np.zeros((N_FRAMES, N_PLAYERS, N_FEATURES), dtype=np.float32)
    mask = np.ones((N_FRAMES, N_PLAYERS), dtype=bool)  # True = invalid (padding)
    
    for fi, frame_no in enumerate(target_frames):
        if frame_no not in frame_index:
            continue
        
        rows = frame_index[frame_no]
        entities = []
        
        # Extract ball
        ball_row = None
        if (rows['HA'] == 0).any():
            ball_row = rows[rows['HA'] == 0].iloc[0]
        else:
            sys_target_min = rows['SysTarget'].min()
            if not pd.isna(sys_target_min):
                ball_candidates = rows[rows['SysTarget'] == sys_target_min]
                if len(ball_candidates) > 0:
                    ball_row = ball_candidates.iloc[0]
        
        if ball_row is not None:
            x = float(ball_row['X']) / 5250.0
            y = float(ball_row['Y']) / 3400.0
            speed = min(float(ball_row['Speed']) / 34.0, 1.0)
            entities.append([x, y, speed, 0, 1])  # team_flag=0, is_ball=1
        
        # Extract home players (HA == 1)
        home_rows = rows[rows['HA'] == 1].sort_values('No')
        for _, player_row in home_rows.iterrows():
            x = float(player_row['X']) / 5250.0
            y = float(player_row['Y']) / 3400.0
            speed = min(float(player_row['Speed']) / 34.0, 1.0)
            entities.append([x, y, speed, 1, 0])  # team_flag=1, is_ball=0
        
        # Extract away players (HA == 2)
        away_rows = rows[rows['HA'] == 2].sort_values('No')
        for _, player_row in away_rows.iterrows():
            x = float(player_row['X']) / 5250.0
            y = float(player_row['Y']) / 3400.0
            speed = min(float(player_row['Speed']) / 34.0, 1.0)
            entities.append([x, y, speed, 2, 0])  # team_flag=2, is_ball=0
        
        # Store entities (up to N_PLAYERS)
        for eidx, entity_features in enumerate(entities[:N_PLAYERS]):
            features[fi, eidx, :] = entity_features
            mask[fi, eidx] = False  # Mark as valid
    
    return features, mask


def game_time_str(half: int, seconds: float) -> str:
    mm = int(seconds) // 60
    ss = int(seconds) % 60
    return f'{half} - {mm:02d}:{ss:02d}'


def main():
    parser = argparse.ArgumentParser(description='Create tracking clips from match data')
    parser.add_argument('--match_dir', type=str, required=True, help='Path to match directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--min_valid_frames', type=int, default=30, help='Minimum valid frames to save clip')
    
    args = parser.parse_args()
    
    match_dir = Path(args.match_dir)
    out_dir = Path(args.out_dir)
    match_id = match_dir.name
    
    # Load data
    play_df, tracking_df = load_match_data(match_dir)
    frame_index = build_frame_index(tracking_df)
    
    # Create output directory
    out_match_dir = out_dir / match_id
    out_match_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean play_df
    play_df = play_df.dropna(subset=['フレーム番号'])
    play_df['フレーム番号'] = play_df['フレーム番号'].astype(int)
    play_df = play_df.drop_duplicates(subset=['フレーム番号'], keep='first')
    
    # Process events
    results = []
    
    for _, row in tqdm(play_df.iterrows(), total=len(play_df)):
        center_frame = int(row['フレーム番号'])
        action = str(row.get('アクション名', ''))
        
        half_val = row.get('試合状態ID', 1)
        half = int(half_val) if half_val in [1, 2] else 1
        
        rel_time = float(row.get('ハーフ開始相対時間', 0) or 0)
        game_time = game_time_str(half, rel_time)
        
        try:
            features, mask = extract_frame_features(frame_index, center_frame)
        except Exception:
            continue
        
        valid = (~mask).sum()  # Count valid entity-frame pairs
        if valid < args.min_valid_frames:
            continue
        
        # Create safe filename
        safe = f'{match_id}_{game_time.replace(" - ", "_").replace(":", "_")}'
        
        # Save arrays
        npy_path = out_match_dir / f'{safe}.npy'
        mask_path = out_match_dir / f'{safe}_mask.npy'
        np.save(npy_path, features)
        np.save(mask_path, mask)
        
        results.append({
            'game_time': game_time,
            'action': action,
            'match_id': match_id,
            'npy_path': str(npy_path),
            'mask_path': str(mask_path),
            'center_frame': center_frame
        })
    
    # Save metadata
    json_path = Path(out_dir) / 'tracking_clips.json'
    existing_results = []
    if json_path.exists():
        with open(json_path, 'r') as f:
            existing_results = json.load(f)
    
    existing_results.extend(results)
    
    with open(json_path, 'w') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=2)
    
    print(f'{len(results)} clips saved to {out_dir}')


if __name__ == '__main__':
    main()
