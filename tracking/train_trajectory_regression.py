"""
Trajectory Regression Model Training Script.
Uses TrajectoryRegressionDataset to train a model that predicts future player positions
as continuous (x, y) coordinates without using tokenization.
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.dataset.trajectory_regression_dataset import TrajectoryRegressionDataset
from tracking.model.trajectory_regression_model import TrajectoryRegressionModel


def collate_fn(batch):
    """
    Collate function for regression batches.
    batch: list of (context_feat, context_mask, target_xy, target_mask, seq_id)
    """
    feats, masks, target_xys, target_masks, seq_ids = zip(*batch)
    return {
        'tracking': torch.stack(feats),           # (B, context_len, N, F)
        'mask': torch.stack(masks),               # (B, context_len, N)
        'target_xy': torch.stack(target_xys),     # (B, K, N, 2)
        'target_mask': torch.stack(target_masks), # (B, K, N)
        'seq_id': list(seq_ids),
    }


def compute_ade(pred, target_xy, target_mask):
    """
    Compute Average Displacement Error.
    pred: (B, K, N, 2)
    target_xy: (B, K, N, 2)
    target_mask: (B, K, N) - True where masked, False where valid
    """
    valid = ~target_mask  # (B, K, N) - True where valid
    if valid.sum() == 0:
        return float('nan')
    diff = pred - target_xy  # (B, K, N, 2)
    dist = diff.pow(2).sum(-1).sqrt()  # (B, K, N)
    return dist[valid].mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory regression training with continuous (x, y) predictions"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json",
        help="Input clips JSON path",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Pre-trained checkpoint for weight initialization",
    )
    parser.add_argument(
        "--out_ckpt",
        type=str,
        default="checkpoints/trajectory_regression.pth",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=20,
        help="Context window length (frames)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Number of future frames to predict",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Sliding window stride",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Val/test data ratio each (train = 1 - 2*test_ratio)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=0,
        help="Max number of games to use (0 = all)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Cap total samples (0=all). Useful for smoke tests.",
    )

    args = parser.parse_args()

    # Step 1: Load dataset and split into train/val/test
    print("=" * 60)
    print("Step 1: Loading dataset and train/val/test split")
    print("=" * 60)
    random.seed(args.seed)
    full_dataset = TrajectoryRegressionDataset(
        args.json_path, args.context_len, args.K, args.step, max_games=args.max_games
    )
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    if args.max_samples > 0:
        indices = indices[:args.max_samples]
    n_test = max(1, int(len(indices) * args.test_ratio))
    n_val  = max(1, int(len(indices) * args.test_ratio))
    test_indices  = indices[:n_test]
    val_indices   = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)
    test_dataset  = Subset(full_dataset, test_indices)
    print(
        f"Full dataset: {len(full_dataset)} samples, "
        f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # Save test split
    split_json = Path(args.out_ckpt).parent / "trajectory_regression_splits.json"
    split_json.parent.mkdir(parents=True, exist_ok=True)
    with open(split_json, "w") as f:
        json.dump({"train": train_indices, "val": val_indices, "test": test_indices}, f)
    print(f"Splits saved: {split_json}")

    # Step 2: Initialize model
    print("\n" + "=" * 60)
    print("Step 2: Model initialization")
    print("=" * 60)
    model = TrajectoryRegressionModel(
        K=args.K,
        N=23,
        num_query=32,
        d_model=256,
        num_features=5
    )
    model.to(args.device)
    print(f"Model initialized and moved to {args.device}")

    # Step 3: Load pre-trained weights
    print("\n" + "=" * 60)
    print("Step 3: Loading pre-trained weights")
    print("=" * 60)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        model.load_pretrained(args.ckpt_path)
        print(f"Pre-trained weights loaded from {args.ckpt_path}")
    elif not args.ckpt_path:
        print("No ckpt_path specified, training from scratch")
    else:
        print(f"WARNING: ckpt_path not found ({args.ckpt_path}), training from scratch")

    # Step 4: Create DataLoaders
    print("\n" + "=" * 60)
    print("Step 4: Creating DataLoaders")
    print("=" * 60)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Train DataLoader: {len(train_loader)} batches")
    print(f"Val DataLoader: {len(val_loader)} batches")

    # Step 5: Training loop
    print("\n" + "=" * 60)
    print("Step 5: Training loop")
    print("=" * 60)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_csv = Path(args.out_ckpt).with_suffix('.train_log.csv')
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_ade'])

    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            pred = model(batch['tracking'], batch['mask'])
            loss = model.compute_loss(pred, batch['target_xy'].to(args.device), batch['target_mask'].to(args.device))
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_ade = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                pred = model(batch['tracking'], batch['mask'])
                ade = compute_ade(pred, batch['target_xy'], batch['target_mask'])
                if not (ade != ade):  # Check if not NaN
                    val_ade += ade
                    val_count += 1

        val_ade = val_ade / max(1, val_count) if val_count > 0 else float('nan')

        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_train_loss:.4f}  val_ade={val_ade:.4f}")
        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{val_ade:.6f}"])

    # Step 6: Test evaluation
    print("\n" + "=" * 60)
    print("Step 6: Test evaluation")
    print("=" * 60)
    model.eval()
    test_ade, test_count = 0.0, 0
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = model(batch['tracking'], batch['mask'])
            ade = compute_ade(pred, batch['target_xy'], batch['target_mask'])
            if not (ade != ade):
                test_ade += ade
                test_count += 1
    test_ade = test_ade / max(1, test_count) if test_count > 0 else float('nan')
    print(f"Final test_ade={test_ade:.4f}")
    with open(log_csv, "a", newline="") as f:
        csv.writer(f).writerow(["test", "", f"{test_ade:.6f}"])

    # Step 7: Save checkpoint
    print("\n" + "=" * 60)
    print("Step 7: Saving checkpoint")
    print("=" * 60)
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'state_dict': model.state_dict(),
            'K': args.K,
            'context_len': args.context_len,
        },
        args.out_ckpt,
    )
    print(f"Checkpoint saved: {args.out_ckpt}")
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
