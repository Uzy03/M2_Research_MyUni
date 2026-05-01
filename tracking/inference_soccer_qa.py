#!/usr/bin/env python3
"""Phase 3: Multi-task QA inference for all 4 tasks."""
import argparse
import csv
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer as rs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.matchvoice_model_tracking import matchvoice_model_tracking
from tracking.dataset.multitask_dataset import TASKS, ACTION_VOCAB

_rouge = rs.RougeScorer(['rougeL'], use_stemmer=True)
_ACTION_VOCAB_SET = set(ACTION_VOCAB)


def compute_f1_action(pred, gt):
    if not gt.strip():
        return None
    gt_labels   = {w.strip() for w in gt.split(',')   if w.strip() in _ACTION_VOCAB_SET}
    pred_labels = {w.strip() for w in pred.split(',') if w.strip() in _ACTION_VOCAB_SET}
    if not gt_labels:
        return None
    if not pred_labels:
        return 0.0
    tp = len(pred_labels & gt_labels)
    p  = tp / len(pred_labels)
    r  = tp / len(gt_labels)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_rouge_l(pred, gt):
    if not gt.strip():
        return None
    return _rouge.score(gt, pred)['rougeL'].fmeasure


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Multi-task QA inference")
    parser.add_argument("--json_path",   default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json")
    parser.add_argument("--ckpt_path",   default="checkpoints/action_alignment.pth")
    parser.add_argument("--llm_ckpt",    default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--out_csv",     default="results/soccer_qa_results.csv")
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--max_games",   type=int, default=0)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--max_new_tokens',     type=int,   default=128)
    return parser.parse_args()


def load_clips(json_path, max_samples, seed, max_games=0):
    with open(json_path) as f:
        clips = json.load(f)
    if max_games > 0:
        seen, allowed = [], set()
        for e in clips:
            if e['game_id'] not in allowed:
                seen.append(e['game_id'])
                if len(seen) > max_games:
                    break
                allowed.add(e['game_id'])
        clips = [e for e in clips if e['game_id'] in allowed]
    random.seed(seed)
    if max_samples > 0 and len(clips) > max_samples:
        clips = random.sample(clips, max_samples)
    return clips, Path(json_path).parent


def load_model(ckpt_path, llm_ckpt, device):
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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()
                  if not k.startswith("llama_model.")}
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    model.use_logits_filter = False
    return model


def make_feat(entry, base_dir, context_len, device):
    npy  = np.load(base_dir / entry['npy_path'])
    mask = np.load(base_dir / entry['mask_path'])
    T = npy.shape[0]
    if T >= context_len:
        feat_np, mask_np = npy[-context_len:], mask[-context_len:]
    else:
        pad = context_len - T
        feat_np = np.concatenate([np.zeros((pad, npy.shape[1], npy.shape[2]), dtype=np.float32), npy])
        mask_np = np.concatenate([np.ones((pad, mask.shape[1]), dtype=bool), mask])
    tracking = torch.FloatTensor(feat_np).unsqueeze(0).to(device)
    mask_t   = torch.BoolTensor(mask_np).unsqueeze(0).to(device)
    return tracking, mask_t


def main():
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*attention mask.*")

    args = parse_args()
    clips, base_dir = load_clips(args.json_path, args.max_samples, args.seed, args.max_games)
    print(f"Loaded {len(clips)} clips, running {len(TASKS)} tasks each")

    model = load_model(args.ckpt_path, args.llm_ckpt, args.device)
    model._repetition_penalty = args.repetition_penalty
    model._max_new_tokens     = args.max_new_tokens
    print(f"Model ready on {args.device}")

    task_scores = {t['name']: [] for t in TASKS}
    rows = []

    for i, entry in enumerate(clips):
        npy_path = base_dir / entry['npy_path']
        if not npy_path.exists():
            continue
        tracking, mask_t = make_feat(entry, base_dir, args.context_len, args.device)

        for task in TASKS:
            gt = entry.get(task['label_field'], '')
            if not gt:
                continue
            model.instruction = task['instruction']
            samples = {
                "tracking":       tracking,
                "mask":           mask_t,
                "labels":         torch.zeros(1, 1, dtype=torch.long).to(args.device),
                "attention_mask": torch.ones(1, 1, dtype=torch.long).to(args.device),
                "input_ids":      torch.zeros(1, 1, dtype=torch.long).to(args.device),
                "caption_text":   [gt],
                "video_path":     [entry.get('clip_id', '')],
            }
            with torch.no_grad():
                generated_list, _, _ = model(samples, validating=True)
            gen = generated_list[0] if generated_list else ""

            if task['name'] == 'action':
                score = compute_f1_action(gen, gt)
                metric = 'f1'
            else:
                score = compute_rouge_l(gen, gt)
                metric = 'rouge_l'

            if score is not None:
                task_scores[task['name']].append(score)

            rows.append({
                'clip_id':     entry.get('clip_id', ''),
                'task':        task['name'],
                'metric':      metric,
                'gt':          gt,
                'generated':   gen,
                'score':       f"{score:.4f}" if score is not None else '',
            })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(clips)}] processed")

    # Summary
    print("\n=== Results ===")
    for task in TASKS:
        name = task['name']
        scores = task_scores[name]
        metric = 'f1_action' if name == 'action' else f'rouge_l_{name}'
        avg = sum(scores) / len(scores) if scores else float('nan')
        print(f"  {metric}: {avg:.4f}  (n={len(scores)})")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['clip_id', 'task', 'metric', 'gt', 'generated', 'score'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {args.out_csv}")


if __name__ == '__main__':
    main()
