import argparse
import configparser
import os
import zipfile

import SoccerNet.Downloader as SNDown
from SoccerNet.utils import getListGames


def is_valid_zip(path):
    try:
        return zipfile.ZipFile(path).testzip() is None
    except Exception:
        return False


def download_captions(downloader, local_dir, match_paths, split):
    """指定した試合のキャプションをダウンロード（既存はスキップ）"""
    for match_path in match_paths:
        cap_path = os.path.join(local_dir, "caption-2023", match_path, "Labels-caption.json")
        if os.path.exists(cap_path):
            print(f"  [skip caption] {match_path}")
            continue
        print(f"  Downloading caption: {match_path} ...")
        orig = SNDown.getListGames
        SNDown.getListGames = lambda s, task="caption-2023", mp=match_path: [mp]
        try:
            downloader.downloadDataTask(task="caption-2023", split=[split])
        except Exception as e:
            print(f"  ERROR caption: {e}")
        finally:
            SNDown.getListGames = orig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="SoccerNet")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD"))
    args = parser.parse_args()

    downloader = SNDown.SoccerNetDownloader(LocalDirectory=args.local_dir)
    if args.password:
        downloader.password = args.password

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")
        game_list = getListGames(split)

        # --- Part A: tracking（MOT zip）---
        print(f"\n[Part A] tracking MOT zip")
        zip_path = os.path.join(args.local_dir, "tracking", f"{split}.zip")
        if not os.path.exists(zip_path) or not is_valid_zip(zip_path):
            if os.path.exists(zip_path):
                print(f"  Corrupted zip, re-downloading: {zip_path}")
            else:
                print(f"  Downloading {zip_path} ...")
            downloader.downloadDataTask(task="tracking", split=[split])
        else:
            print(f"  [skip] {zip_path}")

        mot_games = []
        if os.path.exists(zip_path) and is_valid_zip(zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                gi_paths = [p for p in zf.namelist()
                            if p.endswith("/gameinfo.ini") and "/SNMOT-" in p]
                seen = set()
                for gi in gi_paths:
                    cfg = configparser.ConfigParser()
                    cfg.read_string(zf.read(gi).decode("utf-8"))
                    gid = int(cfg["Sequence"].get("gameid", -1))
                    if 0 <= gid < len(game_list) and gid not in seen:
                        seen.add(gid)
                        mot_games.append(game_list[gid])
            print(f"  Found {len(mot_games)} games in zip")
            download_captions(downloader, args.local_dir, mot_games, split)
        else:
            print(f"  WARNING: {zip_path} not available, skipping Part A")

        # --- Part B: tracking-2023（1試合ずつ）---
        print(f"\n[Part B] tracking-2023 per game")
        tracking_2023_games = []
        for match_path in game_list:
            game_dir = os.path.join(args.local_dir, match_path)
            if os.path.exists(os.path.join(game_dir, "tracking-2023")) or \
               os.path.exists(os.path.join(game_dir, "tracking_2023")):
                tracking_2023_games.append(match_path)
                print(f"  [skip] {match_path}")
                continue
            print(f"  Downloading tracking-2023: {match_path} ...")
            orig = SNDown.getListGames
            SNDown.getListGames = lambda s, task="tracking-2023", mp=match_path: \
                [mp] if s == split else []
            try:
                downloader.downloadDataTask(task="tracking-2023", split=[split])
                if os.path.exists(os.path.join(game_dir, "tracking-2023")) or \
                   os.path.exists(os.path.join(game_dir, "tracking_2023")):
                    tracking_2023_games.append(match_path)
            except Exception as e:
                print(f"  ERROR: {e}")
            finally:
                SNDown.getListGames = orig

        print(f"  Found {len(tracking_2023_games)} games with tracking-2023")
        download_captions(downloader, args.local_dir, tracking_2023_games, split)

    print("\nAll done.")


if __name__ == "__main__":
    main()
