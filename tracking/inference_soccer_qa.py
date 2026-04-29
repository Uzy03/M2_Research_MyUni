#!/usr/bin/env python3
"""
Zero-shot QA inference script for soccer video understanding.

This script loads pre-trained tracking and LLM models to generate
natural language answers to questions about soccer video clips based on
player tracking data and temporal features.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.matchvoice_model_tracking import matchvoice_model_tracking


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-shot QA inference on soccer video clips"
    )
    parser.add_argument(
        "--json_path",
        default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json",
        help="Path to clips.json file",
    )
    parser.add_argument(
        "--ckpt_path",
        default="checkpoints/action_alignment.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--llm_ckpt",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="LLM model checkpoint or huggingface model name",
    )
    parser.add_argument(
        "--config",
        default="configs/qa_action.json",
        help="Path to config JSON with instruction and max_new_tokens",
    )
    parser.add_argument(
        "--out_csv",
        default="results/soccer_qa_results.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=20,
        help="Context length for tracking features (frames)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Maximum number of samples to process (0 = all)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_config(config_path):
    """
    Load config JSON file.
    
    Args:
        config_path: Path to config JSON file
    
    Returns:
        Dictionary with 'instruction' and 'max_new_tokens' keys
    """
    with open(config_path) as f:
        config = json.load(f)
    instruction = config.get("instruction", "")
    max_new_tokens = config.get("max_new_tokens", 64)
    return instruction, max_new_tokens


def load_clips(json_path, max_samples, seed):
    """
    Load clips from JSON file and sample if needed.
    
    Args:
        json_path: Path to clips.json
        max_samples: Maximum number of samples (0 = all)
        seed: Random seed
    
    Returns:
        (clips_list, base_dir_path)
    """
    with open(json_path) as f:
        all_clips = json.load(f)
    
    random.seed(seed)
    if max_samples > 0 and len(all_clips) > max_samples:
        clips = random.sample(all_clips, max_samples)
    else:
        clips = all_clips
    
    base_dir = Path(json_path).parent
    return clips, base_dir


def initialize_model(ckpt_path, llm_ckpt, device, instruction, max_new_tokens):
    """
    Initialize model with checkpoint.
    
    Args:
        ckpt_path: Path to model checkpoint
        llm_ckpt: LLM model checkpoint/huggingface name
        device: Device to run on
        instruction: QA instruction prompt
        max_new_tokens: Max tokens for generation
    
    Returns:
        Initialized model on device in eval mode
    """
    model = matchvoice_model_tracking(
        load_checkpoint=False,
        num_features=768,
        need_temporal="yes",
        llm_ckpt=llm_ckpt,
        tokenizer_ckpt=llm_ckpt,
        open_llm_decoder=False,
        num_players=23,
        in_features=5,
        d_model=256,
        max_frame_pos=200,
    )
    
    # Load checkpoint - exclude LLM keys (already loaded from llm_ckpt, ~16GB)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    # Exclude frozen LLM weights (llama_model.*) to avoid CPU OOM
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("llama_model.")}
    
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in state_dict.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    
    # Set QA-specific attributes
    model.instruction = instruction
    model._max_new_tokens = max_new_tokens
    model.use_logits_filter = False
    
    return model


def process_clip(model, base_dir, entry, context_len, device):
    """
    Process a single clip and generate QA output.
    
    Args:
        model: Initialized model
        base_dir: Base directory for relative paths
        entry: Dictionary with 'npy_path', 'mask_path', etc.
        context_len: Context length in frames
        device: Device to run on
    
    Returns:
        Dictionary with results or None if file not found
    """
    npy_path = base_dir / entry["npy_path"]
    mask_path = base_dir / entry["mask_path"]
    
    if not npy_path.exists():
        return None
    
    # Load tracking features and mask
    npy = np.load(npy_path)  # (T, N, F) - time, players, features
    mask = np.load(mask_path)  # (T, N) - time, players
    T = npy.shape[0]
    
    # Use last context_len frames, zero-pad if shorter
    if T >= context_len:
        feat_np = npy[-context_len:]
        mask_np = mask[-context_len:]
    else:
        pad = context_len - T
        feat_np = np.concatenate(
            [
                np.zeros((pad, npy.shape[1], npy.shape[2]), dtype=np.float32),
                npy,
            ]
        )
        mask_np = np.concatenate(
            [np.ones((pad, mask.shape[1]), dtype=bool), mask]
        )
    
    # Convert to tensors and move to device
    tracking = torch.FloatTensor(feat_np).unsqueeze(0).to(device)  # (1, T, N, F)
    mask_t = torch.BoolTensor(mask_np).unsqueeze(0).to(device)  # (1, T, N)
    
    # Prepare sample dictionary for model
    samples = {
        "tracking": tracking,
        "mask": mask_t,
        "labels": torch.zeros(1, 1, dtype=torch.long).to(device),
        "attention_mask": torch.ones(1, 1, dtype=torch.long).to(device),
        "input_ids": torch.zeros(1, 1, dtype=torch.long).to(device),
        "caption_text": [entry.get("action_label", "")],
        "video_path": [entry.get("clip_id", "")],
    }
    
    # Run inference
    with torch.no_grad():
        generated_list, _, _ = model(samples, validating=True)
    
    generated = generated_list[0] if generated_list else ""
    
    return {
        "clip_id": entry.get("clip_id", ""),
        "action_label": entry.get("action_label", ""),
        "generated": generated,
    }


def main():
    """Main inference loop."""
    args = parse_args()
    
    # Step 1: Load config
    instruction, max_new_tokens = load_config(args.config)
    print(f"[Config] instruction: {instruction[:50]}...")
    print(f"[Config] max_new_tokens: {max_new_tokens}")
    
    # Step 2: Load and sample clips
    clips, base_dir = load_clips(args.json_path, args.max_samples, args.seed)
    print(f"[Clips] Loaded {len(clips)} clips from {args.json_path}")
    
    # Step 3: Initialize model
    print(f"[Model] Loading from {args.ckpt_path}...")
    model = initialize_model(
        args.ckpt_path, args.llm_ckpt, args.device, instruction, max_new_tokens
    )
    print(f"[Model] Ready on {args.device}")
    
    # Step 4: Inference loop
    results = []
    for i, entry in enumerate(clips):
        print(f"\n[{i+1}/{len(clips)}] Processing {entry.get('clip_id', 'unknown')}...")
        
        result = process_clip(model, base_dir, entry, args.context_len, args.device)
        if result is None:
            print(f"  → File not found, skipping")
            continue
        
        results.append({
            "clip_id": result["clip_id"],
            "action_label": result["action_label"],
            "instruction": instruction,
            "generated": result["generated"],
        })
        print(f"  → action={result['action_label']}")
        print(f"  → generated: {result['generated'][:80]}...")
    
    # Step 5: Save results to CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_id", "action_label", "instruction", "generated"]
        )
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[Results] Saved {len(results)} results to {args.out_csv}")


if __name__ == "__main__":
    main()
