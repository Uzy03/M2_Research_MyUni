"""
TrajectoryDataset を使い、TrackingEncoder + Q-Former → LLM(凍結) で
次 K フレームの座標テキストを生成する学習スクリプト。
InstructBLIP 方式: 指示文を -100 マスクして答えトークンのみで loss 計算。
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.dataset.trajectory_dataset import TrajectoryDataset
from model.matchvoice_model_tracking import matchvoice_model_tracking


INSTRUCTION = (
    "Predict the (x,y) positions of all 23 players for the next 10 frames (5 FPS). "
    "Format: p0:[(x,y),...], p1:[(x,y),...], ..., p22:[(x,y),...]"
)


def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        # batch: list of (context_feat, context_mask, target_text, seq_id, start_frame)
        feats, masks, target_texts, seq_ids, start_frames = zip(*batch)
        tracking = torch.stack(feats)  # (B, context_len, N, F)
        mask_tensor = torch.stack(masks)  # (B, context_len, N)

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
        description="Trajectory prediction fine-tuning with TrackingEncoder + Q-Former + LLM"
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
        default="checkpoints/trajectory.pth",
        help="学習済みモデルの保存先",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=100,
        help="コンテキストウィンドウの長さ（フレーム）",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="ターゲット軌跡のフレーム数",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="スライディングウィンドウのストライド",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
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
        default=768,
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
        "--test_ratio",
        type=float,
        default=0.2,
        help="テストデータの割合",
    )

    args = parser.parse_args()

    # Step 1: データ読み込みと train/test 分割
    print("=" * 60)
    print("Step 1: データセット読み込みと train/test 分割")
    print("=" * 60)
    random.seed(args.seed)
    full_dataset = TrajectoryDataset(
        args.json_path, args.context_len, args.K, args.step
    )
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    n_test = max(1, int(len(indices) * args.test_ratio))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    train_dataset = Subset(full_dataset, train_indices)
    print(
        f"Full dataset: {len(full_dataset)} samples, "
        f"Train: {len(train_indices)}, Test: {len(test_indices)}"
    )

    # test split を保存
    test_json = Path(args.out_ckpt).parent / "trajectory_test_split.json"
    test_json.parent.mkdir(parents=True, exist_ok=True)
    test_windows = [
        {
            "clip_idx": full_dataset.windows[i][0],
            "start_frame": full_dataset.windows[i][1],
            "npy_path": full_dataset.data[full_dataset.windows[i][0]]["npy_path"],
            "mask_path": full_dataset.data[full_dataset.windows[i][0]]["mask_path"],
            "seq_id": full_dataset.data[full_dataset.windows[i][0]].get("seq_id", ""),
        }
        for i in test_indices
    ]
    with open(test_json, "w") as f:
        json.dump(test_windows, f, ensure_ascii=False, indent=2)
    print(f"Test split saved: {test_json} ({len(test_windows)} samples)")

    # Step 2: モデル初期化
    print("\n" + "=" * 60)
    print("Step 2: モデル初期化")
    print("=" * 60)
    model = matchvoice_model_tracking(
        load_checkpoint=False,
        num_features=768,
        need_temporal="yes",
        llm_ckpt=args.llm_ckpt,
        tokenizer_ckpt=args.llm_ckpt,
        open_llm_decoder=False,  # LLM 凍結
        num_players=23,
        in_features=5,
        d_model=256,
        max_frame_pos=200,
    )
    model.to(args.device)
    print(f"Model initialized and moved to {args.device}")

    # Step 3: 既存チェックポイントから Q-Former + LLM 重みをロード
    print("\n" + "=" * 60)
    print("Step 3: 既存チェックポイントからロード")
    print("=" * 60)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
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
        print(f"Missing keys (expected for TrackingEncoder): {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: ckpt_path not found ({args.ckpt_path}), training from scratch")

    # Step 4: collate_fn の定義
    print("\n" + "=" * 60)
    print("Step 4: collate_fn 定義")
    print("=" * 60)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    collate_fn = make_collate_fn(model.tokenizer, args.max_length)
    print("collate_fn created")

    # Step 5: DataLoader 作成
    print("\n" + "=" * 60)
    print("Step 5: DataLoader 作成")
    print("=" * 60)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"DataLoader created: {len(train_loader)} batches")

    # Step 6: 訓練ループ
    print("\n" + "=" * 60)
    print("Step 6: 訓練ループ")
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

    # Step 7: チェックポイント保存
    print("\n" + "=" * 60)
    print("Step 7: チェックポイント保存")
    print("=" * 60)
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    model.instruction = INSTRUCTION
    torch.save(
        {"state_dict": model.state_dict(), "instruction": INSTRUCTION},
        args.out_ckpt,
    )
    print(f"Checkpoint saved: {args.out_ckpt}")
    print("=" * 60)
    print("学習完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
