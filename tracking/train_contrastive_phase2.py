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
from tracking.dataset.window_dataset import WindowDataset

def window_collate_fn(batch):
    # batch: list of (tracking_window: Tensor(W,23,5), mask_window: Tensor(W,23), action_text: str)
    trackings, masks, texts = zip(*batch)
    return {
        'tracking':    torch.stack(trackings),   # (B, W, 23, 5)
        'mask':        torch.stack(masks),       # (B, W, 23)
        'action_text': list(texts),              # list[str]
    }

def info_nce_loss(track_emb, text_emb, temperature):
    """対称型 InfoNCE。両入力は L2 正規化済み (B, D)。"""
    B = track_emb.shape[0]
    logits   = track_emb @ text_emb.T / temperature   # (B, B)
    labels   = torch.arange(B, device=track_emb.device)
    loss_t2s = F.cross_entropy(logits,   labels)
    loss_s2t = F.cross_entropy(logits.T, labels)
    return (loss_t2s + loss_s2t) / 2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_path",   default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json")
    p.add_argument("--ckpt_path",   default="checkpoints/phase1/trajectory_regression.pth")
    p.add_argument("--llm_ckpt",    default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--out_ckpt",    default="checkpoints/phase1_5/encoder_contrastive.pth")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--max_games",   type=int,   default=0)
    p.add_argument("--max_samples", type=int,   default=0)
    p.add_argument("--test_ratio",  type=float, default=0.1)
    p.add_argument("--window_size", type=int,   default=2)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    dataset = WindowDataset(args.json_path, window_size=args.window_size,
                            max_games=args.max_games)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    if args.max_samples > 0:
        indices = indices[:args.max_samples]
    n_val = max(1, int(len(indices) * args.test_ratio))
    val_indices   = indices[:n_val]
    train_indices = indices[n_val:]
    train_loader = DataLoader(Subset(dataset, train_indices),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=window_collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(dataset, val_indices),
                              batch_size=args.batch_size, shuffle=False,
                              collate_fn=window_collate_fn, num_workers=4, pin_memory=True)
    print(f"Dataset: {len(dataset)} window samples  Train: {len(train_indices)}  Val: {len(val_indices)}")

    model = matchvoice_model_tracking(
        load_checkpoint=False, num_features=768, need_temporal="yes",
        llm_ckpt=args.llm_ckpt, tokenizer_ckpt=args.llm_ckpt,
        open_llm_decoder=False, num_players=23, in_features=5,
        d_model=256, max_frame_pos=200,
    )
    model.to(args.device)

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        remap_prefix = {'tracking_encoder.': 'visual_encoder.', 'qformer.': 'video_Qformer.'}
        remap_exact  = {'query_tokens': 'video_query_tokens'}
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            for old_p, new_p in remap_prefix.items():
                if k.startswith(old_p):
                    new_k = new_p + k[len(old_p):]
                    break
            if new_k == k and k in remap_exact:
                new_k = remap_exact[k]
            remapped[new_k] = v
        model_state = model.state_dict()
        filtered = {k: v for k, v in remapped.items()
                    if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} keys from Phase 1 checkpoint")
    else:
        print(f"WARNING: ckpt_path not found ({args.ckpt_path}), training from scratch")

    # 全パラメータを凍結
    for param in model.parameters():
        param.requires_grad = False
    # visual_encoder + enc_proj のみ解凍
    for param in model.visual_encoder.parameters():
        param.requires_grad = True
    for param in model.enc_proj.parameters():
        param.requires_grad = True

    trainable = list(model.visual_encoder.parameters()) + list(model.enc_proj.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.out_ckpt).with_suffix('.log.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])
    embed_fn = model.llama_model.model.embed_tokens

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()

            # Encoder → enc_proj → L2 normalize
            enc_out    = model.visual_encoder(batch['tracking'], batch['mask'])  # (B, W, 768)
            track_mean = enc_out.mean(dim=1)                                      # (B, 768)
            track_proj = model.enc_proj(track_mean.to(model.enc_proj.weight.dtype))  # (B, 4096)

            # テキスト埋め込み（LLM frozen）
            text_embs = []
            for text in batch['action_text']:
                ids = model.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to(args.device)
                with torch.no_grad():
                    emb = embed_fn(ids).float().mean(dim=1)   # (1, 4096)
                text_embs.append(emb)
            text_emb = torch.cat(text_embs, dim=0)            # (B, 4096)

            track_norm = F.normalize(track_proj.float(), dim=-1)
            text_norm  = F.normalize(text_emb,           dim=-1)
            loss = info_nce_loss(track_norm, text_norm, args.temperature)

            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total += loss.item(); n += 1
        avg_train = total / max(1, n)
        scheduler.step()

        # Validation
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                enc_out    = model.visual_encoder(batch['tracking'], batch['mask'])
                track_mean = enc_out.mean(dim=1)
                track_proj = model.enc_proj(track_mean.to(model.enc_proj.weight.dtype))
                text_embs  = []
                for text in batch['action_text']:
                    ids = model.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to(args.device)
                    emb = embed_fn(ids).float().mean(dim=1)
                    text_embs.append(emb)
                text_emb   = torch.cat(text_embs, dim=0)
                track_norm = F.normalize(track_proj.float(), dim=-1)
                text_norm  = F.normalize(text_emb,           dim=-1)
                loss = info_nce_loss(track_norm, text_norm, args.temperature)
                if not torch.isnan(loss):
                    total += loss.item(); n += 1
        avg_val = total / max(1, n)

        mark = ""
        if avg_val < best_val:
            best_val = avg_val
            save_state = {k: v for k, v in model.state_dict().items()
                          if not k.startswith('llama_model.')}
            torch.save({'state_dict': save_state}, args.out_ckpt)
            mark = " <- best"

        print(f"Epoch {epoch}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}{mark}")
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{avg_train:.6f}', f'{avg_val:.6f}'])

    print(f"Best val InfoNCE loss: {best_val:.4f}  Saved: {args.out_ckpt}")

if __name__ == '__main__':
    main()
