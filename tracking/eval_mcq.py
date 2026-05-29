#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

MCQ_MAPS = {
    "qa_formation_mcq": {
        "A": "4-4-2",
        "B": "4-3-3",
        "C": "3-5-2",
        "D": "4-2-3-1",
    },
    "qa_defensive_line_mcq": {
        "A": "very low",
        "B": "low",
        "C": "medium",
        "D": "high",
        "E": "very high",
    },
}

GT_FIELD = {
    "qa_formation_mcq": "formation_defend",
    "qa_defensive_line_mcq": "def_line_label",
}


def extract_choice(text: str) -> str | None:
    m = re.search(r'\b([A-E])\b', text.upper())
    return m.group(1) if m else None


def eval_config(phase4_dir: Path, config_name: str, spatial_labels: dict) -> dict:
    results_path = phase4_dir / config_name / "results.json"
    if not results_path.exists():
        print(f"  [SKIP] {results_path} not found")
        return None

    with open(results_path) as f:
        results = json.load(f)

    choice_map = MCQ_MAPS.get(config_name)
    gt_field = GT_FIELD.get(config_name)
    if choice_map is None or gt_field is None:
        print(f"  [SKIP] No MCQ mapping for config: {config_name}")
        return None

    correct = 0
    total = 0
    for entry in results:
        clip_id = entry.get("clip_id", "")
        generated = entry.get("generated", "")
        label_entry = spatial_labels.get(clip_id)
        if label_entry is None:
            continue
        gt = label_entry.get(gt_field)
        if gt is None:
            continue
        choice = extract_choice(generated)
        if choice is None:
            total += 1
            continue
        pred = choice_map.get(choice)
        if pred == gt:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"config": config_name, "n": total, "accuracy": accuracy, "correct": correct}


def main():
    parser = argparse.ArgumentParser(description="Evaluate MCQ accuracy from phase4 results")
    parser.add_argument("--phase4_dir", required=True)
    parser.add_argument("--spatial_labels", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()

    phase4_dir = Path(args.phase4_dir)
    with open(args.spatial_labels) as f:
        spatial_labels = json.load(f)

    summary_rows = []
    for config_name in args.configs:
        result = eval_config(phase4_dir, config_name, spatial_labels)
        if result is None:
            continue
        acc = result["accuracy"]
        n = result["n"]
        correct = result["correct"]
        print(f"{config_name}: n={n}, accuracy={acc:.4f} ({correct}/{n})")
        summary_rows.append({"config": config_name, "n": n, "accuracy": acc})

    if summary_rows:
        summary_path = phase4_dir / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["config", "n", "accuracy"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
