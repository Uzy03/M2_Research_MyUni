"""LLM baseline: メタデータのみで Phase 4 と同じ質問に回答"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ACTION_NAMES_EN = {
    '15': 'shot', '16': 'goalkeeper save', '17': 'direct free kick',
    '21': 'corner kick', '22': 'indirect free kick', '23': 'offside',
    '27': 'foul', '29': 'pass', '30': 'pass',
    '35': 'dribble', '36': 'through pass', '37': 'clearance',
    '38': 'foul received', '41': 'interception', '42': 'clearance',
    '43': 'block', '44': 'throw-in', '45': 'cross',
    '50': 'trap', '72': 'feed', '73': 'touch', '74': 'tackle',
    '75': 'flick-on',
}


def parse_args():
    parser = argparse.ArgumentParser(description="LLM baseline evaluation using metadata only")
    parser.add_argument("--clips_json", type=str, required=True)
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--configs", nargs="+", default=["qa_formation", "qa_commentary", "qa_first_action"])
    parser.add_argument("--out_dir", type=str, default="checkpoints/llm_baseline")
    parser.add_argument("--llm_ckpt", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_prompt(entry, instruction):
    possession = entry.get("label_possession") or ""
    zone = entry.get("label_zone") or ""
    pressure = entry.get("label_pressure") or ""
    action_names = ", ".join(
        ACTION_NAMES_EN.get(str(a), str(a)) for a in entry.get("action_sequence", [])
    )
    return (
        "You are a soccer analyst. Based on the following information about a soccer clip, "
        "answer the question concisely.\n\n"
        "## Clip Information\n"
        f"- Possession: {possession}\n"
        f"- Zone: {zone}\n"
        f"- Pressure: {pressure}\n"
        f"- Action sequence: {action_names}\n\n"
        f"## Question\n{instruction}\n\nAnswer:"
    )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    with open(args.clips_json, encoding="utf-8") as f:
        clips = json.load(f)

    if args.max_clips and args.max_clips < len(clips):
        rng = random.Random(args.seed)
        clips = rng.sample(clips, args.max_clips)

    print(f"Loading LLM: {args.llm_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_ckpt, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    device = args.device if args.device == "cpu" else f"{args.device}:{args.gpu}"

    for config_name in args.configs:
        cfg_path = Path(args.config_dir) / f"{config_name}.json"
        if not cfg_path.exists():
            print(f"Warning: {cfg_path} not found, skipping")
            continue

        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        instruction = cfg["instruction"]
        max_new_tokens = cfg.get("max_new_tokens", 100)

        print(f"\n=== {config_name}: {instruction[:60]}... ===")
        rows = []
        for i, entry in enumerate(clips):
            prompt = build_prompt(entry, instruction)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated = full_text.split("Answer:")[-1].strip()
            rows.append({
                "clip_id":     entry.get("clip_id", ""),
                "instruction": instruction,
                "generated":   generated,
                "action":      ", ".join(ACTION_NAMES_EN.get(str(a), str(a)) for a in entry.get("action_sequence", [])),
                "possession":  entry.get("label_possession") or "",
                "zone":        entry.get("label_zone") or "",
                "pressure":    entry.get("label_pressure") or "",
            })
            if (i + 1) % 10 == 0 or i == len(clips) - 1:
                print(f"  [{i+1}/{len(clips)}] {generated[:60]}")

        out_config_dir = out_dir / config_name
        out_config_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_config_dir / "results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path}  ({len(rows)} clips)")


if __name__ == "__main__":
    main()
