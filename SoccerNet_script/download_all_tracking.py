import argparse
import configparser
import os
import zipfile

import SoccerNet.Downloader as SNDown
from SoccerNet.utils import getListGames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="SoccerNet")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD"))
    args = parser.parse_args()

    downloader = SNDown.SoccerNetDownloader(LocalDirectory=args.local_dir)
    if args.password:
        downloader.password = args.password

    for split in args.splits:
        zip_path = os.path.join(args.local_dir, "tracking", f"{split}.zip")

        # Step 1: zipが無ければダウンロード
        if not os.path.exists(zip_path):
            print(f"Downloading tracking {split}.zip ...")
            downloader.downloadDataTask(task="tracking", split=[split])
        else:
            print(f"[skip] {zip_path}")

        if not os.path.exists(zip_path):
            print(f"  WARNING: {zip_path} not found, skipping")
            continue

        # Step 2: zip内のゲームIDを収集
        game_list = getListGames(split)
        unique_games = {}
        with zipfile.ZipFile(zip_path) as zf:
            gi_paths = [p for p in zf.namelist()
                        if p.endswith("/gameinfo.ini") and "/SNMOT-" in p]
            for gi in gi_paths:
                cfg = configparser.ConfigParser()
                cfg.read_string(zf.read(gi).decode("utf-8"))
                gid = int(cfg["Sequence"].get("gameid", -1))
                if 0 <= gid < len(game_list) and gid not in unique_games:
                    unique_games[gid] = game_list[gid]
        print(f"  [{split}] Found {len(unique_games)} unique games")

        # Step 3: 各試合のキャプションをダウンロード（既存はスキップ）
        for gid, match_path in sorted(unique_games.items()):
            cap_path = os.path.join(args.local_dir, "caption-2023",
                                    match_path, "Labels-caption.json")
            if os.path.exists(cap_path):
                print(f"  [skip] {match_path}")
                continue
            print(f"  Downloading caption: {match_path} ...")
            orig = SNDown.getListGames
            SNDown.getListGames = lambda s, task="caption-2023", mp=match_path: [mp]
            try:
                downloader.downloadDataTask(task="caption-2023", split=[split])
            finally:
                SNDown.getListGames = orig

    print("All done.")


if __name__ == "__main__":
    main()
