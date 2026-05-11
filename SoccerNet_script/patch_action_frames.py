import argparse
import json
from pathlib import Path
import pandas as pd

SKIP_ACTION_IDS = {'1','2','3','4','19','25','26','40'}
STEP_RAW = 25      # 25fps → 1fps
WINDOW_RAW = 750   # 30sec * 25fps = 750 original frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='soccerdata_clips/fps1_sec30_onball_step5s/clips.json')
    parser.add_argument('--data_dir', default='/Users/ujihara/m2_研究/SoccerData')
    parser.add_argument('--max_games', type=int, default=0, help='処理するゲーム数上限（0=全件）')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    with open(args.json_path, encoding='utf-8') as f:
        clips = json.load(f)

    unique_games = set()
    for i, entry in enumerate(clips):
        game_id = entry.get('game_id')
        if args.max_games > 0:
            if game_id not in unique_games:
                if len(unique_games) >= args.max_games:
                    continue
                unique_games.add(game_id)
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(clips)}...")
        # Skip if action_sequence is empty
        if not entry.get('action_sequence'):
            entry['action_sequence_frames'] = []
            continue
        # Find game directory
        game_dir = None
        for p in Path(args.data_dir).rglob(entry['game_id']):
            if p.is_dir():
                game_dir = p
                break
        if game_dir is None:
            print(f"WARNING: game_id {entry['game_id']} not found, skipping")
            entry['action_sequence_frames'] = []
            continue
        play_csv_path = game_dir / 'play.csv'
        if not play_csv_path.exists():
            print(f"WARNING: play.csv not found for {entry['game_id']}, skipping")
            entry['action_sequence_frames'] = []
            continue
        try:
            df = pd.read_csv(play_csv_path)
        except Exception as e:
            print(f"WARNING: failed to read {play_csv_path}: {e}")
            entry['action_sequence_frames'] = []
            continue
        frame_col = None
        action_col = None
        for col in df.columns:
            if 'フレーム' in col or 'Frame' in col:
                frame_col = col
            if 'アクション' in col or 'Action' in col:
                action_col = col
        if frame_col is None:
            frame_col = df.columns[0]
        if action_col is None:
            action_col = df.columns[1]
        frame_to_action = {}
        for _, row in df.iterrows():
            if pd.notna(row[frame_col]):
                try:
                    frame_num = int(row[frame_col])
                    action = str(row[action_col]) if pd.notna(row[action_col]) else ""
                    frame_to_action[frame_num] = action
                except Exception:
                    continue
        start_raw = entry['start_frame_orig']
        end_raw   = start_raw + WINDOW_RAW
        actions_in_window = [
            (frame_num, action_id)
            for frame_num, action_id in frame_to_action.items()
            if start_raw <= frame_num <= end_raw and action_id not in SKIP_ACTION_IDS
        ]
        actions_in_window.sort(key=lambda x: x[0])
        sequence_frames = []
        prev = None
        for frame_num, action_id in actions_in_window:
            if action_id != prev:
                clip_frame = round((frame_num - start_raw) / STEP_RAW)
                clip_frame = max(0, min(29, clip_frame))
                sequence_frames.append(clip_frame)
                prev = action_id
        entry['action_sequence_frames'] = sequence_frames
        if len(sequence_frames) != len(entry.get('action_sequence', [])):
            print(f"WARNING: length mismatch for {entry['clip_id']}: "
                  f"action_sequence={len(entry['action_sequence'])}, "
                  f"action_sequence_frames={len(sequence_frames)}")
    if not args.dry_run:
        with open(args.json_path, 'w', encoding='utf-8') as f:
            json.dump(clips, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(clips)} entries to {args.json_path}")
    else:
        print("DRY RUN: no file written")

if __name__ == '__main__':
    main()
