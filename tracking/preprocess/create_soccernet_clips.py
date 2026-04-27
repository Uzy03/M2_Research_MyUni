#!/usr/bin/env python3
"""
SoccerNet tracking clips preprocessing script.
Creates (tracking clip, commentary text) pairs from train.zip (MOT format) and Labels-caption.json.
Saves as .npy + JSON index.
"""

import argparse
import configparser
import glob
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm
from SoccerNet.utils import getListGames

# Constants
IMW = 1920
IMH = 1080
FPS_SRC = 25
FPS_TGT = 5
STEP = FPS_SRC // FPS_TGT  # = 5
N_FRAMES = 150
N_PLAYERS = 23
N_FEATURES = 5
DIAG = math.sqrt(IMW**2 + IMH**2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create SoccerNet tracking clips from MOT format data."
    )
    parser.add_argument(
        "--tracking_zip",
        required=True,
        help="Path to SoccerNet/tracking/train.zip",
    )
    parser.add_argument(
        "--caption_dir",
        required=True,
        help="Directory to search for Labels-caption.json (e.g., SoccerNet/caption-2023/)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for .npy and JSON files",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="SoccerNet split name (default: train)",
    )
    return parser.parse_args()


def load_caption_db(caption_dir):
    """
    Step 1: Recursively search for all Labels-caption.json files under caption_dir.
    Build a nested dict: caption_db[match_path][(half, position_ms)] = {annotation}
    """
    caption_dir_path = Path(caption_dir)
    caption_db = {}

    json_files = glob.glob(
        str(caption_dir_path / "**" / "Labels-caption.json"), recursive=True
    )

    for json_file in json_files:
        try:
            # Compute match_path (relative path without filename)
            rel_path = Path(json_file).relative_to(caption_dir_path)
            match_path = str(rel_path.parent)

            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            if "annotations" not in data:
                continue

            for entry in data["annotations"]:
                if "gameTime" not in entry or "position" not in entry:
                    continue

                game_time_str = entry["gameTime"].strip()
                try:
                    half = int(game_time_str.split(" - ")[0].strip())
                except (ValueError, IndexError):
                    continue

                try:
                    position_ms = int(entry["position"])
                except (ValueError, TypeError):
                    continue

                key = (half, position_ms)
                if match_path not in caption_db:
                    caption_db[match_path] = {}

                caption_db[match_path][key] = {
                    "anonymized": entry.get("anonymized", ""),
                    "label": entry.get("label", ""),
                }
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return caption_db


def process_sequence(zf, gi_path, game_list, caption_db, out_dir):
    """
    Steps 4-9: Process a single SNMOT sequence.
    Returns JSON entry if successful, None otherwise.
    """
    try:
        # Parse gameinfo.ini
        with zf.open(gi_path) as f:
            cfg_str = f.read().decode("utf-8")
        cfg = configparser.ConfigParser()
        cfg.read_string(cfg_str)

        if "Sequence" not in cfg:
            return None

        seq_cfg = cfg["Sequence"]
        gameID = int(seq_cfg.get("gameID", -1))
        clipStart = int(seq_cfg.get("clipStart", 0))
        clipStop = int(seq_cfg.get("clipStop", 0))
        gameTimeStart = seq_cfg.get("gameTimeStart", "1 - 00:00")
        seq_name = seq_cfg.get("name", "SNMOT-000")
        num_tracklets = int(seq_cfg.get("num_tracklets", 0))

        # Extract half
        try:
            half = int(gameTimeStart.split(" - ")[0].strip())
        except (ValueError, IndexError):
            return None

        # Get match_path from game_list
        if gameID < 0 or gameID >= len(game_list):
            return None
        match_path = game_list[gameID]

        # Check caption_db
        if match_path not in caption_db:
            return None

        # Find caption within time range
        matched_annotation = None
        for (cb_half, position_ms), annotation in caption_db[match_path].items():
            if cb_half == half and clipStart <= position_ms <= clipStop:
                matched_annotation = annotation
                break

        if matched_annotation is None:
            return None

        # Build tracklet ID map
        id_to_info = {}
        for i in range(1, num_tracklets + 1):
            key = f"trackletID_{i}"
            val = seq_cfg.get(key, "").strip()
            if not val:
                continue
            parts = val.split(";")
            class_str = parts[0].strip().lower()
            jersey = parts[1].strip() if len(parts) > 1 else ""

            if "ball" in class_str:
                cls = "ball"
            elif "left" in class_str:
                cls = "left"
            elif "right" in class_str:
                cls = "right"
            else:
                cls = "ref"

            id_to_info[i] = {"class": cls, "jersey": jersey}

        # Read gt.txt
        gt_path = f"train/{seq_name}/gt/gt.txt"
        try:
            with zf.open(gt_path) as f:
                lines = f.read().decode("utf-8").splitlines()
        except KeyError:
            return None

        frame_data = {}
        for line in lines:
            cols = line.strip().split(",")
            if len(cols) < 6:
                continue
            try:
                frame_no = int(cols[0])
                tid = int(cols[1])
                x = float(cols[2])
                y = float(cols[3])
                w = float(cols[4])
                h = float(cols[5])
            except ValueError:
                continue

            x_c = (x + w / 2) / IMW
            y_c = (y + h / 2) / IMH
            if frame_no not in frame_data:
                frame_data[frame_no] = {}
            frame_data[frame_no][tid] = (x_c, y_c)

        # Slot assignment
        ball_ids = [tid for tid, info in id_to_info.items() if info["class"] == "ball"]
        left_ids = sorted(
            [tid for tid, info in id_to_info.items() if info["class"] == "left"],
            key=lambda t: id_to_info[t]["jersey"],
        )
        right_ids = sorted(
            [tid for tid, info in id_to_info.items() if info["class"] == "right"],
            key=lambda t: id_to_info[t]["jersey"],
        )

        slot_to_tid = {}
        if ball_ids:
            slot_to_tid[0] = ball_ids[0]
        for i, tid in enumerate(left_ids[:11]):
            slot_to_tid[1 + i] = tid
        for i, tid in enumerate(right_ids[:11]):
            slot_to_tid[12 + i] = tid

        team_flag = {
            slot: (0 if slot == 0 else (1 if slot <= 11 else 2))
            for slot in slot_to_tid
        }
        is_ball = {slot: (1 if slot == 0 else 0) for slot in slot_to_tid}

        # Feature extraction
        sampled_frames = [1 + i * STEP for i in range(N_FRAMES)]
        features = np.zeros((N_FRAMES, N_PLAYERS, N_FEATURES), dtype=np.float32)
        mask = np.ones((N_FRAMES, N_PLAYERS), dtype=bool)

        prev_centers = {}
        for fi, frame_no in enumerate(sampled_frames):
            fdata = frame_data.get(frame_no, {})
            for slot, tid in slot_to_tid.items():
                if tid not in fdata:
                    continue
                x_c, y_c = fdata[tid]
                if slot in prev_centers:
                    px, py = prev_centers[slot]
                    speed = min(
                        math.sqrt((x_c - px) ** 2 + (y_c - py) ** 2) / DIAG, 1.0
                    )
                else:
                    speed = 0.0
                features[fi, slot, 0] = x_c
                features[fi, slot, 1] = y_c
                features[fi, slot, 2] = speed
                features[fi, slot, 3] = team_flag[slot]
                features[fi, slot, 4] = is_ball[slot]
                mask[fi, slot] = False
                prev_centers[slot] = (x_c, y_c)

        # Save .npy files
        out_dir_path = Path(out_dir)
        npy_path = out_dir_path / f"{seq_name}.npy"
        mask_path = out_dir_path / f"{seq_name}_mask.npy"

        np.save(npy_path, features)
        np.save(mask_path, mask)

        # Build JSON entry
        result = {
            "seq_id": seq_name,
            "match_path": match_path,
            "half": half,
            "clip_start_ms": clipStart,
            "clip_stop_ms": clipStop,
            "game_time_start": gameTimeStart,
            "caption": matched_annotation["anonymized"],
            "label": matched_annotation["label"],
            "npy_path": str(npy_path),
            "mask_path": str(mask_path),
        }

        return result

    except Exception as e:
        print(f"Warning: Failed to process {gi_path}: {e}")
        return None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading caption database...")
    caption_db = load_caption_db(args.caption_dir)
    print(f"  Loaded captions for {len(caption_db)} matches")

    print("Loading game list...")
    game_list = getListGames(args.split)
    print(f"  Loaded {len(game_list)} games")

    print("Processing sequences...")
    results = []
    with zipfile.ZipFile(args.tracking_zip, "r") as zf:
        gameinfo_paths = [
            p for p in zf.namelist() if p.endswith("/gameinfo.ini") and "/SNMOT-" in p
        ]
        for gi_path in tqdm(gameinfo_paths, desc="Sequences"):
            result = process_sequence(zf, gi_path, game_list, caption_db, out_dir)
            if result:
                results.append(result)

    # Save JSON index
    json_path = out_dir / "soccernet_clips.json"
    existing = []
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.extend(results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"{len(results)} clips processed -> {json_path}")


if __name__ == "__main__":
    main()
