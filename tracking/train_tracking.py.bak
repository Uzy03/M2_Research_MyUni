"""
TrackingEncoder → Q-Former → LLaMA パイプラインの学習スクリプト
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.matchvoice_model_tracking import matchvoice_model_tracking
from tracking.dataset.soccernet_tracking_dataset import SoccerNetTrackingDataset


def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        features, masks, captions, labels, seq_ids, _ = zip(*batch)
        tracking = torch.stack(features)
        mask_tensor = torch.stack(masks)
        texts = [cap + tokenizer.eos_token for cap in captions]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        lbl = enc.input_ids.clone()
        lbl[enc.attention_mask == 0] = -100
        return {
            "tracking": tracking,
            "mask": mask_tensor,
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": lbl,
            "caption_text": list(captions),
            "video_path": list(seq_ids),
        }
    return collate_fn


def main():
    parser = argparse.ArgumentParser(
        description="TrackingEncoder → Q-Former → LLaMA パイプラインの学習"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="tracking_clips_sn/soccernet_clips.json",
        help="入力クリップデータ JSON パス",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/downstream_commentary_all_open.pth",
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
        default="checkpoints/tracking_finetuned.pth",
        help="学習済みモデルの保存先",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="テストデータの割合",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
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
        default=64,
        help="トークナイザーの最大長",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="計算デバイス",
    )

    args = parser.parse_args()

    # Step 1: データ読み込みと train/test 分割
    print("=" * 60)
    print("Step 1: データ読み込みと train/test 分割")
    print("=" * 60)
    random.seed(args.seed)
    with open(args.json_path) as f:
        data = json.load(f)
    random.shuffle(data)
    n_test = max(1, int(len(data) * args.test_ratio))
    test_data = data[:n_test]
    train_data = data[n_test:]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # test split を保存（inference_tracking.py から参照するため）
    test_json = Path(args.out_ckpt).parent / "tracking_test_split.json"
    test_json.parent.mkdir(parents=True, exist_ok=True)
    with open(test_json, "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"Test split saved: {test_json}")

    # Step 2: Dataset と一時 JSON
    print("\n" + "=" * 60)
    print("Step 2: Dataset と一時 JSON")
    print("=" * 60)
    tmp_json = "/tmp/tracking_train_tmp.json"
    with open(tmp_json, "w") as f:
        json.dump(train_data, f)
    train_dataset = SoccerNetTrackingDataset(tmp_json)
    print(f"Dataset loaded: {len(train_dataset)} samples")

    # Step 3: モデル初期化
    print("\n" + "=" * 60)
    print("Step 3: モデル初期化")
    print("=" * 60)
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
    )
    model.to(args.device)
    print(f"Model initialized and moved to {args.device}")

    # Step 4: 既存チェックポイントから Q-Former + LLM 重みをロード
    print("\n" + "=" * 60)
    print("Step 4: 既存チェックポイントからロード")
    print("=" * 60)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys (expected for TrackingEncoder): {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: ckpt_path not found ({args.ckpt_path}), training from scratch")

    # Step 5: collate_fn の定義 (already defined above)
    print("\n" + "=" * 60)
    print("Step 5: collate_fn 定義")
    print("=" * 60)
    collate_fn = make_collate_fn(model.tokenizer, args.max_length)
    print("collate_fn created")

    # Step 6: DataLoader 作成
    print("\n" + "=" * 60)
    print("Step 6: DataLoader 作成")
    print("=" * 60)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"DataLoader created: {len(train_loader)} batches")

    # Step 7: 訓練ループ
    print("\n" + "=" * 60)
    print("Step 7: 訓練ループ")
    print("=" * 60)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            loss = model(batch, validating=False)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}")

    # Step 8: チェックポイント保存
    print("\n" + "=" * 60)
    print("Step 8: チェックポイント保存")
    print("=" * 60)
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out_ckpt)
    print(f"Checkpoint saved: {args.out_ckpt}")
    print("=" * 60)
    print("学習完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
