import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.matchvoice_model_tracking import matchvoice_model_tracking


def main():
    parser = argparse.ArgumentParser(
        description='Generate captions for tracking data using matchvoice model'
    )
    parser.add_argument('--json_path', type=str, default='checkpoints/tracking_test_split.json',
                        help='Path to test data JSON')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/tracking_finetuned.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--llm_ckpt', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Local path or HuggingFace ID for LLaMA-3-8B-Instruct')
    parser.add_argument('--out_csv', type=str, default='results/tracking_inference.csv',
                        help='Path to output CSV')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')

    args = parser.parse_args()

    # Step 1: Load model
    model = matchvoice_model_tracking(
        load_checkpoint=False,
        num_features=768,
        need_temporal="yes",
        llm_ckpt=args.llm_ckpt,
        tokenizer_ckpt=args.llm_ckpt,
        open_llm_decoder=True,
        num_players=23,
        in_features=5,
        d_model=256,
        max_frame_pos=200,
    )
    model.to(args.device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict
    model.eval()

    # Step 2: Load JSON entries
    with open(args.json_path) as f:
        data = json.load(f)

    # Step 3: Inference on each clip
    results = []
    for entry in tqdm(data, desc="Inference"):
        try:
            features = torch.FloatTensor(np.load(entry["npy_path"])).unsqueeze(0).to(args.device)  # (1, T, N, F)
            mask = torch.BoolTensor(np.load(entry["mask_path"])).unsqueeze(0).to(args.device)      # (1, T, N)

            samples = {
                "tracking": features,
                "mask": mask,
                "labels": torch.zeros(1, 1, dtype=torch.long).to(args.device),
                "attention_mask": torch.ones(1, 1, dtype=torch.long).to(args.device),
                "input_ids": torch.zeros(1, 1, dtype=torch.long).to(args.device),
                "caption_text": [entry.get("caption", "")],
                "video_path": [entry.get("seq_id", "")],
            }

            with torch.no_grad():
                generated_list, _, _ = model(samples, validating=True)

            generated = generated_list[0] if generated_list else ""
            print(f"{entry['seq_id']}: {generated[:80]}")

            results.append({
                "seq_id": entry.get("seq_id", ""),
                "caption_gt": entry.get("caption", ""),
                "generated": generated,
            })
        except Exception as e:
            print(f"Error {entry.get('seq_id', '')}: {e}")
            continue

    # Step 4: Save results to CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_id", "caption_gt", "generated"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved: {args.out_csv}")


if __name__ == "__main__":
    main()
