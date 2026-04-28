import argparse
import os

import SoccerNet.Downloader as SNDown
from SoccerNet.utils import getListGames


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
        print(f"\n=== Processing split: {split} ===")
        games = getListGames(split=split)
        print(f"Found {len(games)} games in {split}")

        for game_idx, match_path in enumerate(games, start=1):
            print(f"\n[{game_idx}/{len(games)}] {match_path}")

            game_dir = os.path.join(args.local_dir, match_path)
            if os.path.exists(os.path.join(game_dir, "tracking-2023")) or os.path.exists(
                os.path.join(game_dir, "tracking_2023")
            ):
                print("  [skip] tracking-2023 already exists")
                continue

            print(f"  Downloading tracking-2023 ...")
            orig = SNDown.getListGames
            SNDown.getListGames = lambda s, task="tracking-2023", mp=match_path: [mp] if s == split else []
            try:
                downloader.downloadDataTask(
                    task="tracking-2023",
                    split=[split],
                )
            finally:
                SNDown.getListGames = orig

    print("\n=== All done ===")


if __name__ == "__main__":
    main()
