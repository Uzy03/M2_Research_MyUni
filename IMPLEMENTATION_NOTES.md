# preprocess_soccerdata.py Implementation Notes

## File Location
`/Users/ujihara/m2_研究/UniSoccer/SoccerNet_script/preprocess_soccerdata.py`

## Overview
This script generates sliding-window on-ball clips from SoccerData tracking data, creating NPY format arrays with normalized positions, velocities, and metadata.

## Key Features
- Reads tracking.csv, play.csv, players.csv from SoccerData directories
- Generates (n_frames, 23, 5) NPY clips with position, speed, team, ball features
- Separate bool mask arrays for missing data
- JSON index with clip metadata and relative paths
- Configurable FPS, clip duration, step size, ball coverage threshold
- Automatic Japanese column header detection
- Progress tracking with tqdm
- Graceful error handling with per-game skipping

## Usage
```bash
python3 SoccerNet_script/preprocess_soccerdata.py \
  --fps 1 \
  --sec 30 \
  --step_sec 5 \
  --ball_cov 0.7
```

## Output
- `soccerdata_clips/fps1_sec30_onball_step5s/clips.json` — metadata index
- `soccerdata_clips/fps1_sec30_onball_step5s/npy/{game_id}/clip_XXXX.npy` — feature arrays
- `soccerdata_clips/fps1_sec30_onball_step5s/npy/{game_id}/clip_XXXX_mask.npy` — mask arrays

## NPY Format
- Shape: (n_frames, 23, 5) dtype=float32
- Features per slot: [x_norm, y_norm, speed_norm, team_flag, is_ball]
- Slots: 0=Ball, 1-11=Home, 12-22=Away (all sorted by jersey number)

## Verified ✓
- Syntax valid (Python 3.11)
- Imports available
- CLI argument parsing working
- 449 lines of robust code
