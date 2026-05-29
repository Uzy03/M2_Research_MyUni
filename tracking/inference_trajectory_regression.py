import argparse
import csv
import json
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.model.trajectory_regression_model import TrajectoryRegressionModel
from tracking.dataset.trajectory_regression_dataset import TrajectoryRegressionDataset


def main():
    parser = argparse.ArgumentParser(
        description='Inference trajectory regression model'
    )
    parser.add_argument(
        '--json_path',
        type=str,
        default='checkpoints/trajectory_regression_test_split.json',
        help='Path to test data JSON'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='checkpoints/trajectory_regression.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--out_csv',
        type=str,
        default='results/trajectory_regression_inference.csv',
        help='Path to output CSV'
    )
    parser.add_argument(
        '--K',
        type=int,
        default=5,
        help='Number of frames in target trajectory'
    )
    parser.add_argument(
        '--context_len',
        type=int,
        default=20,
        help='Length of context window in frames'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='Stride for sliding window (used when json_path is clips.json format)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for inference'
    )

    args = parser.parse_args()

    # Step 1: Load model checkpoint and create model
    print(f"Loading checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # Extract model hyperparameters from checkpoint, fallback to args
    K = ckpt.get('K', args.K)
    context_len = ckpt.get('context_len', args.context_len)
    
    # Create model
    model = TrajectoryRegressionModel(
        K=K,
        N=23,
        d_model=256,
        num_features=5,
        context_len=context_len,
    )
    
    # Load state dict with shape filtering
    model_state = model.state_dict()
    filtered_state = {
        k: v for k, v in state_dict.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(filtered_state, strict=False)
    
    model.to(args.device)
    model.eval()
    print(f"Loaded {len(filtered_state)} parameters")

    # Step 2: Load test data
    print(f"Loading test data from {args.json_path}")
    
    # Determine JSON format and load accordingly
    with open(args.json_path) as f:
        test_data = json.load(f)
    
    # Check if it's test_split format (list of dicts with npy_path, mask_path, etc.)
    # or clips.json format (requires DataLoader)
    is_test_split_format = (
        isinstance(test_data, list) and
        len(test_data) > 0 and
        'npy_path' in test_data[0] and
        'mask_path' in test_data[0]
    )
    
    results = []
    
    if is_test_split_format:
        # Direct inference without DataLoader
        print(f"Using test_split format with {len(test_data)} entries")
        
        for entry in tqdm(test_data, desc="Inference (test_split)"):
            try:
                # Resolve paths relative to CWD
                npy_path = Path(entry['npy_path'])
                mask_path = Path(entry['mask_path'])
                start = entry['start_frame']
                seq_id = entry.get('seq_id', '')
                
                # Check if files exist
                if not npy_path.exists() or not mask_path.exists():
                    print(f"Skipping {seq_id}: files not found")
                    continue
                
                # Load features and masks
                npy = np.load(npy_path)      # (T, N, F)
                mask = np.load(mask_path)    # (T, N)
                
                # Extract context window
                context_npy = npy[start:start+context_len]    # (context_len, N, F)
                context_mask = mask[start:start+context_len]  # (context_len, N)
                target_npy = npy[start+context_len:start+context_len+K]      # (K, N, F)
                target_mask = mask[start+context_len:start+context_len+K]    # (K, N)
                
                # Skip if not enough frames
                if context_npy.shape[0] < context_len or target_npy.shape[0] < K:
                    continue
                
                # Prepare tensors
                tracking = torch.FloatTensor(context_npy).unsqueeze(0).to(args.device)   # (1, context_len, N, F)
                mask_t = torch.BoolTensor(context_mask).unsqueeze(0).to(args.device)     # (1, context_len, N)
                
                # Inference
                with torch.no_grad():
                    pred = model(tracking, mask_t)  # (1, K, N, 2)
                
                # Convert to numpy
                pred_np = pred.squeeze(0).cpu().numpy()   # (K, N, 2)
                target_xy = target_npy[:, :, :2]          # (K, N, 2)
                valid = ~target_mask                       # (K, N) True=valid
                
                # Compute ADE: mean displacement over valid (k, n) pairs
                diff = pred_np - target_xy  # (K, N, 2)
                dist = np.sqrt((diff ** 2).sum(axis=-1))  # (K, N)
                ade = float(dist[valid].mean()) if valid.any() else float('nan')
                
                # Compute FDE: displacement at last frame
                fde_valid = valid[-1]  # (N,)
                fde = float(dist[-1][fde_valid].mean()) if fde_valid.any() else float('nan')
                
                results.append({
                    'seq_id': seq_id,
                    'start_frame': start,
                    'ade': ade,
                    'fde': fde
                })
            
            except Exception as e:
                print(f"Error processing {entry.get('seq_id', '')}: {e}")
                continue
    
    else:
        # Use DataLoader for clips.json format
        print(f"Using clips.json format")
        dataset = TrajectoryRegressionDataset(
            args.json_path,
            context_len=context_len,
            K=K,
            step=args.step
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        for batch in tqdm(dataloader, desc="Inference (clips)"):
            try:
                context_feat, context_mask, target_xy, target_mask, seq_ids = batch
                
                # Move to device
                context_feat = context_feat.to(args.device)
                context_mask = context_mask.to(args.device)
                target_xy = target_xy.to(args.device)
                target_mask = target_mask.to(args.device)
                
                # Inference
                with torch.no_grad():
                    pred = model(context_feat, context_mask)  # (B, K, N, 2)
                
                # Compute metrics for each sample in batch
                pred_np = pred.cpu().numpy()           # (B, K, N, 2)
                target_xy_np = target_xy.cpu().numpy()  # (B, K, N, 2)
                target_mask_np = target_mask.cpu().numpy()  # (B, K, N)
                
                for b in range(pred_np.shape[0]):
                    valid = ~target_mask_np[b]  # (K, N)
                    
                    # ADE
                    diff = pred_np[b] - target_xy_np[b]  # (K, N, 2)
                    dist = np.sqrt((diff ** 2).sum(axis=-1))  # (K, N)
                    ade = float(dist[valid].mean()) if valid.any() else float('nan')
                    
                    # FDE
                    fde_valid = valid[-1]  # (N,)
                    fde = float(dist[-1][fde_valid].mean()) if fde_valid.any() else float('nan')
                    
                    results.append({
                        'seq_id': seq_ids[b] if isinstance(seq_ids, list) else str(seq_ids[b]),
                        'ade': ade,
                        'fde': fde
                    })
            
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    # Step 3: Save results to CSV
    print(f"Saving results to {args.out_csv}")
    output_dir = Path(args.out_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if results:
        fieldnames = list(results[0].keys())
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                # Convert NaN to string 'nan' for CSV output
                row_out = {}
                for k, v in row.items():
                    if isinstance(v, float) and math.isnan(v):
                        row_out[k] = 'nan'
                    else:
                        row_out[k] = v
                writer.writerow(row_out)
        print(f'Saved: {args.out_csv}')
    else:
        print("No results to save")

    # Step 4: Compute and save summary
    ade_values = [r['ade'] for r in results if isinstance(r['ade'], float) and not math.isnan(r['ade'])]
    fde_values = [r['fde'] for r in results if isinstance(r['fde'], float) and not math.isnan(r['fde'])]
    
    ade_mean = float(np.mean(ade_values)) if ade_values else float('nan')
    fde_mean = float(np.mean(fde_values)) if fde_values else float('nan')
    
    summary = {
        'ade_mean': ade_mean,
        'fde_mean': fde_mean,
        'n_results': len(results),
        'n_valid_ade': len(ade_values),
        'n_valid_fde': len(fde_values)
    }
    
    summary_path = Path(args.out_csv).with_name(
        Path(args.out_csv).stem + '_summary.json'
    )
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {summary_path}')
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total results: {summary['n_results']}")
    print(f"Valid ADE samples: {summary['n_valid_ade']}")
    print(f"Valid FDE samples: {summary['n_valid_fde']}")
    if not math.isnan(ade_mean):
        print(f"ADE mean: {ade_mean:.6f}")
    else:
        print("ADE mean: nan")
    if not math.isnan(fde_mean):
        print(f"FDE mean: {fde_mean:.6f}")
    else:
        print("FDE mean: nan")


if __name__ == '__main__':
    main()
