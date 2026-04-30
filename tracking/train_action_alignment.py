import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.dataset.action_alignment_dataset import ActionAlignmentDataset
from model.matchvoice_model_tracking import matchvoice_model_tracking


INSTRUCTION = 'List the soccer actions occurring in this tracking sequence in chronological order.'


def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        feats, masks, target_texts, seq_ids = zip(*batch)
        tracking = torch.stack(feats)
        mask_tensor = torch.stack(masks)

        input_ids_list, labels_list, attn_list = [], [], []
        for target_text in target_texts:
            bos_id = tokenizer.bos_token_id
            inst_ids = tokenizer(INSTRUCTION, add_special_tokens=False).input_ids
            ans_ids = tokenizer(
                target_text + tokenizer.eos_token, add_special_tokens=False
            ).input_ids

            full_ids = ([bos_id] + inst_ids + ans_ids)[: max_length]
            lbl = ([-100] + [-100] * len(inst_ids) + ans_ids)[: max_length]

            pad_len = max_length - len(full_ids)
            attn = [1] * len(full_ids) + [0] * pad_len
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            lbl = lbl + [-100] * pad_len

            input_ids_list.append(full_ids)
            labels_list.append(lbl)
            attn_list.append(attn)

        return {
            "tracking": tracking,
            "mask": mask_tensor,
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "caption_text": list(target_texts),
            "video_path": list(seq_ids),
        }

    return collate_fn


def main():
    parser = argparse.ArgumentParser(
        description="Action alignment fine-tuning with TrackingEncoder + Q-Former + LLM"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json",
        help="入力クリップデータ JSON パス",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/trajectory_regression.pth",
        help="Q-Former+LLM 初期化用チェックポイント",
    )
    parser.add_argument(
        "--llm_ckpt",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="LLM チェックポイント",
    )
    parser.add_argument(
        "--out_ckpt",
        type=str,
        default="checkpoints/action_alignment.pth",
        help="学習済みモデルの保存先",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=20,
        help="コンテキストウィンドウの長さ（フレーム）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="エポック数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学習率",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="バッチサイズ",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="トークナイザーの最大長",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="計算デバイス",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=0,
        help="使用する試合数上限（0=全試合）",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: データセット読み込み")
    print("=" * 60)
    random.seed(args.seed)
    dataset = ActionAlignmentDataset(args.json_path, args.context_len, max_games=args.max_games)
    print(f"Dataset: {len(dataset)} samples")

    print("\n" + "=" * 60)
    print("Step 2: モデル初期化")
    print("=" * 60)
    model = matchvoice_model_tracking(
        load_checkpoint=False,
        num_features=768,
        need_temporal="yes",
        llm_ckpt=args.llm_ckpt,
        tokenizer_ckpt=args.llm_ckpt,
        open_llm_decoder=False,
        num_players=23,
        in_features=5,
        d_model=256,
        max_frame_pos=200,
    )
    model.to(args.device)
    print(f"Model initialized and moved to {args.device}")

    print("\n" + "=" * 60)
    print("Step 3: 既存チェックポイントからロード")
    print("=" * 60)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        # Remap Phase 1 key names to Phase 2 naming convention
        remap_prefix = {
            "tracking_encoder.": "visual_encoder.",
            "qformer.": "video_Qformer.",
        }
        remap_exact = {"query_tokens": "video_query_tokens"}
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
        state_dict = remapped
        model_state = model.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = [
            k
            for k, v in state_dict.items()
            if k in model_state and model_state[k].shape != v.shape
        ]
        if skipped:
            print(f"Skipping size-mismatched keys: {skipped}")
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} keys from Phase 1 checkpoint")
        print(f"Missing keys (LLM + unmatched): {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: ckpt_path not found ({args.ckpt_path}), training from scratch")

    print("\n" + "=" * 60)
    print("Step 4: collate_fn 定義")
    print("=" * 60)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    collate_fn = make_collate_fn(model.tokenizer, args.max_length)
    print("collate_fn created")

    print("\n" + "=" * 60)
    print("Step 5: DataLoader 作成")
    print("=" * 60)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    print(f"DataLoader created: {len(train_loader)} batches")

    print("\n" + "=" * 60)
    print("Step 6: 訓練ループ")
    print("=" * 60)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    log_csv = Path(args.out_ckpt).with_suffix('.train_log.csv')
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            loss = model(batch, validating=False)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_loss:.4f}")
        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_loss:.6f}"])

    print("\n" + "=" * 60)
    print("Step 7: チェックポイント保存")
    print("=" * 60)
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    save_state = {k: v for k, v in model.state_dict().items() if not k.startswith('llama_model.')}
    torch.save(
        {"state_dict": save_state, "instruction": INSTRUCTION},
        args.out_ckpt,
    )
    print(f"Checkpoint saved: {args.out_ckpt}")
    print("=" * 60)
    print("学習完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
