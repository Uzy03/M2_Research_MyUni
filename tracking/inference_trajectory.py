import argparse
import json
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
import pandas as pd

from tracking.dataset.trajectory_dataset import (
    format_trajectory, parse_trajectory, compute_ade_fde
)
from model.matchvoice_model_tracking import matchvoice_model_tracking

INSTRUCTION = (
    "Predict the (x,y) positions of all 23 players for the next 10 frames (5 FPS). "
    "Format: p0:[(x,y),...], p1:[(x,y),...], ..., p22:[(x,y),...]"
)


def main():
    parser = argparse.ArgumentParser(
        description='Inference trajectory prediction using matchvoice model'
    )
    parser.add_argument('--json_path', type=str, default='checkpoints/trajectory_test_split.json',
                        help='Path to test data JSON')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/trajectory.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--llm_ckpt', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Local path or HuggingFace ID for LLaMA-3-8B-Instruct')
    parser.add_argument('--out_csv', type=str, default='results/trajectory_inference.csv',
                        help='Path to output CSV')
    parser.add_argument('--context_len', type=int, default=100,
                        help='Length of context window in frames')
    parser.add_argument('--K', type=int, default=10,
                        help='Number of frames in target trajectory')
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
    
    # Load checkpoint and set instruction
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.instruction = ckpt.get('instruction', INSTRUCTION)
    model.use_logits_filter = False
    model._max_new_tokens = 1024
    
    del ckpt, state_dict
    model.eval()

    # Step 2: Load JSON entries
    with open(args.json_path) as f:
        test_data = json.load(f)

    # Step 3: Inference on each clip
    results = []
    for entry in tqdm(test_data, desc="Inference"):
        try:
            # Load features and masks
            npy = np.load(entry['npy_path'])   # (T, N, F)
            mask = np.load(entry['mask_path'])  # (T, N)

            start = entry['start_frame']
            
            # Extract context window
            context_feat = torch.FloatTensor(
                npy[start:start+args.context_len]
            ).unsqueeze(0).to(args.device)  # (1, context_len, N, F)
            context_mask = torch.BoolTensor(
                mask[start:start+args.context_len]
            ).unsqueeze(0).to(args.device)  # (1, context_len, N)

            # Dummy tokens
            dummy = torch.zeros(1, 1, dtype=torch.long).to(args.device)
            
            samples = {
                'tracking':       context_feat,
                'mask':           context_mask,
                'input_ids':      dummy,
                'attention_mask': torch.ones(1, 1, dtype=torch.long).to(args.device),
                'labels':         dummy,
                'caption_text':   [''],
                'video_path':     [entry.get('seq_id', '')],
            }

            with torch.no_grad():
                generated, _, _ = model(samples, validating=True)

            # Ground truth trajectory
            target_npy = npy[start+args.context_len : start+args.context_len+args.K]
            gt_xy = target_npy[:, :, :2]   # (K, N, 2)
            gt_text = format_trajectory(gt_xy, args.K)

            # Compute ADE/FDE
            pred_xy = parse_trajectory(generated, args.K)
            ade, fde = compute_ade_fde(pred_xy, gt_xy)

            results.append({
                'seq_id': entry.get('seq_id', ''),
                'start_frame': start,
                'generated': generated,
                'gt_text': gt_text,
                'ade': ade,
                'fde': fde,
            })
        except Exception as e:
            print(f"Error {entry.get('seq_id', '')}: {e}")
            continue

    # Step 4: Save results to CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    
    print(f'Saved: {args.out_csv}')
    print(f'ADE mean: {df.ade.mean():.4f}')
    print(f'FDE mean: {df.fde.mean():.4f}')


if __name__ == "__main__":
    main()
