import os
from pathlib import Path

print("=== /SoccerData mount check ===")
print("exists:", os.path.exists("/SoccerData"))
if os.path.exists("/SoccerData"):
    children = list(Path("/SoccerData").iterdir())[:5]
    print("contents:", children)
    found = False
    for p in Path("/SoccerData").rglob("2023102002"):
        if p.is_dir():
            print("game_dir found:", p)
            csv = p / "play.csv"
            print("play.csv exists:", csv.exists())
            found = True
            break
    if not found:
        print("game_id 2023102002 NOT found")
else:
    print("ERROR: /SoccerData not mounted")
