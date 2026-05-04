#!/usr/bin/env python3
"""Phase 2 (対照学習): Q-Former を action ラベルの LLM 埋め込みに整合させる。"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.matchvoice_model_tracking import matchvoice_model_tracking
from tracking.dataset.multitask_dataset import MultiTaskDataset


def contrastive_collate_fn(batch):
    feats, masks, _, answers, _, _ = zip(*batch)
    return {
        'tracking': torch.stack(feats),
        'mask':     torch.stack(masks),
        'caption_text': list(answers),
    }


def get_text_repr(model, text_list, device):
    """action ラベル文字列を LLM embed_fn で埋め込んで平均プールする。"""
    embed_fn = model.llama_model.model.embed_tokens
    reprs = []
    with torch.no_grad():
        for text in text_list:
            ids = model.tokenizer(
                text, add_special_tokens=False, return_tensors='pt'
            ).input_ids.to(device)
            if ids.numel() == 0:
                ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            emb = embed_fn(ids).float().mean(dim=1)  # (1, llm_hidden)
            reprs.append(emb)
    return torch.cat(reprs, dim=0)  # (B, llm_hidden)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_path",   default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json")
    p.add_argument("--ckpt_path",   default="checkpoints/phase1/trajectory_regression.pth")
    p.add_argument("--llm_ckpt",    default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--out_ckpt",    default="checkpoints/phase2_contrastive/contrastive.pth")
    p.add_argument("--context_len", type=int,   default=20)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--max_games",   type=int,   default=0)
    p.add_argument("--max_samples", type=int,   default=0)
    p.add_argument("--test_ratio",  type=float, default=0.1)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset = MultiTaskDataset(args.json_path, args.context_len,
                               max_games=args.max_games, allowed_tasks=['action'])
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    if args.max_samples > 0:
        indices = indices[:args.max_samples]
    n_val = max(1, int(len(indices) * args.test_ratio))
    val_indices   = indices[:n_val]
    train_indices = indices[n_val:]

    train_loader = DataLoader(Subset(dataset, train_indices),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=contrastive_collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(dataset, val_indices),
                              batch_size=args.batch_size, shuffle=False,
                              collate_fn=contrastive_collate_fn,
                              num_workers=4, pin_memory=True)

    model = matchvoice_model_tracking(
        load_checkpoint=False,
        num_features=768, need_temporal="yes",
        llm_ckpt=args.llm_ckpt, tokenizer_ckpt=args.llm_ckpt,
        open_llm_decoder=False,
        num_players=23, in_features=5, d_model=256, max_frame_pos=200,
    )
    model.to(args.device)

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        remap = {"tracking_encoder.": "visual_encoder.", "qformer.": "video_Qformer."}
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            for old, new in remap.items():
                if k.startswith(old):
                    new_k = new + k[len(old):]
                    break
            remapped[new_k] = v
        model_state = model.state_dict()
        filtered = {k: v for k, v in remapped.items()
                    if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} keys from Phase 1 checkpoint")

    for param in model.visual_encoder.parameters():
        param.requires_grad = False
    for param in model.llama_model.parameters():
        param.requires_grad = False

    trainable = (
        list(model.video_Qformer.parameters()) +
        [model.video_query_tokens] +
        list(model.llama_proj.parameters()) +
        list(model.ln_vision.parameters())
    )
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.out_ckpt).with_suffix('.log.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            inputs_llama = model.forward_contrastive(batch)        # (B, 32, llm_hidden)
            track_repr   = inputs_llama.mean(dim=1).float()        # (B, llm_hidden)
            text_repr    = get_text_repr(model, batch['caption_text'], args.device)
            loss = (1 - F.cosine_similarity(track_repr, text_repr, dim=-1)).mean()
            loss.backward()
            optimizer.step()
            total += loss.item(); n += 1
        avg_train = total / max(1, n)
        scheduler.step()

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                inputs_llama = model.forward_contrastive(batch)
                track_repr   = inputs_llama.mean(dim=1).float()
                text_repr    = get_text_repr(model, batch['caption_text'], args.device)
                loss = (1 - F.cosine_similarity(track_repr, text_repr, dim=-1)).mean()
                total += loss.item(); n += 1
        avg_val = total / max(1, n)

        mark = ""
        if avg_val < best_val:
            best_val = avg_val
            save_state = {k: v for k, v in model.state_dict().items()
                          if not k.startswith('llama_model.')}
            torch.save({"state_dict": save_state}, args.out_ckpt)
            mark = " <- best"

        print(f"Epoch {epoch}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}{mark}")
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{avg_train:.6f}', f'{avg_val:.6f}'])

    print(f"Best val cosine loss: {best_val:.4f}  Saved: {args.out_ckpt}")


if __name__ == '__main__':
    main()
