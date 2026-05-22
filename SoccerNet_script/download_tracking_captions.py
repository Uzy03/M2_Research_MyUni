import argparse
import configparser
import os
import zipfile

import SoccerNet.Downloader as SNDown
from SoccerNet.utils import getListGames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_zip", default="SoccerNet/tracking/train.zip")
    parser.add_argument("--local_dir", default="./SoccerNet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD"))
    args = parser.parse_args()

    # Step 1: Extract unique games from tracking zip
    print(f"Reading gameinfo from {args.tracking_zip}...")
    game_list = getListGames(args.split)

    unique_games = {}  # gameID -> match_path (deduplicated)

    with zipfile.ZipFile(args.tracking_zip) as zf:
        gi_paths = [p for p in zf.namelist() if p.endswith("/gameinfo.ini") and "/SNMOT-" in p]
        for gi in gi_paths:
            cfg = configparser.ConfigParser()
            cfg.read_string(zf.read(gi).decode("utf-8"))
            gid = int(cfg["Sequence"].get("gameid", -1))
            if 0 <= gid < len(game_list) and gid not in unique_games:
                unique_games[gid] = game_list[gid]

    # Step 2: Display discovered games
    print(f"Found {len(unique_games)} unique games in tracking zip:")
    for gid, mp in sorted(unique_games.items()):
        print(f"  [{gid}] {mp}")

    # Step 3: Download captions for each game
    downloader = SNDown.SoccerNetDownloader(LocalDirectory=args.local_dir)
    if args.password:
        downloader.password = args.password

    print(f"\nDownloading captions to {args.local_dir}/caption-2023/...")
    for gid, match_path in sorted(unique_games.items()):
        cap_path = os.path.join(args.local_dir, "caption-2023", match_path, "Labels-caption.json")
        if os.path.exists(cap_path):
            print(f"  [skip] {match_path}")
            continue

        print(f"  Downloading: {match_path} ...")
        orig = SNDown.getListGames
        SNDown.getListGames = lambda split, task="caption-2023", mp=match_path: [mp]
        try:
            downloader.downloadDataTask(task="caption-2023", split=[args.split])
        finally:
            SNDown.getListGames = orig

    print("All done.")


if __name__ == "__main__":
    main()
