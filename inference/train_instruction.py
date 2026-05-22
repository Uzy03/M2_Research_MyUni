"""
SoccerReplay-1988 Instruction-following Fine-tuning スクリプト
動画クリップ + テキスト指示 → アクションラベル生成
InstructBLIP 方式：指示文トークンを labels で -100 マスクして、答えトークンのみで loss 計算
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
from dataset.video_dataset import VideoCaptionDataset
from model.matchvoice_model_all_blocks import matchvoice_model_all_blocks


INSTRUCTIONS = [
    'What action is occurring in this soccer clip?',
    'Identify the soccer action shown in this video clip.',
    'What event is taking place in this soccer scene?',
]


def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        # batch: list of (frames, caption_idx, video_path, caption_text, text_key_val)
        frames_list, _, video_paths, caption_texts, _ = zip(*batch)
        frames_batch = torch.stack(frames_list)   # [B, C, T, H, W]

        input_ids_list = []
        labels_list = []
        for caption_text in caption_texts:
            inst = random.choice(INSTRUCTIONS)
            answer = caption_text

            bos_id  = tokenizer.bos_token_id
            inst_ids = tokenizer(inst, add_special_tokens=False).input_ids
            ans_ids  = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False).input_ids

            full_ids = [bos_id] + inst_ids + ans_ids
            lbl      = [-100]   + [-100]*len(inst_ids) + ans_ids

            # max_length でクリップ
            full_ids = full_ids[:max_length]
            lbl      = lbl[:max_length]
            input_ids_list.append(full_ids)
            labels_list.append(lbl)

        # パディング（右パディング）
        max_len = max(len(x) for x in input_ids_list)
        pad_id  = tokenizer.pad_token_id
        attention_masks = []
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            attention_masks.append([1]*len(input_ids_list[i]) + [0]*pad_len)
            input_ids_list[i] = input_ids_list[i] + [pad_id]*pad_len
            labels_list[i]    = labels_list[i]    + [-100]*pad_len

        return {
            'frames':          frames_batch,
            'input_ids':       torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask':  torch.tensor(attention_masks, dtype=torch.long),
            'labels':          torch.tensor(labels_list, dtype=torch.long),
            'caption_text':    list(caption_texts),
            'video_path':      list(video_paths),
        }
    return collate_fn


def main():
    parser = argparse.ArgumentParser(
        description="SoccerReplay-1988 Instruction-following Fine-tuning"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="train_data/json/SoccerReplay-1988/classification_train.json",
        help="SoccerReplay-1988 classification JSON パス",
    )
    parser.add_argument(
        "--video_base",
        type=str,
        default="/path/to/soccerreplay/videos",
        help="SoccerReplay-1988 動画フォルダ",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/downstream_commentary_all_open.pth",
        help="初期化用チェックポイント",
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
        default="checkpoints/instruction_action.pth",
        help="学習済みモデルの保存先",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="テストデータの割合",
    )

    args = parser.parse_args()

    # Step 1: データ読み込みと train/test 分割
    print("=" * 60)
    print("Step 1: データ読み込みと train/test 分割")
    print("=" * 60)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = VideoCaptionDataset(
        json_file=[args.json_path],
        video_base_dir=[args.video_base],
        require_text=True,
        text_key='comments_text_anonymized',
        keywords=[
            'var', 'end of half game', 'clearance', 'second yellow card', 'injury',
            'ball possession', 'throw in', 'show added time', 'shot off target',
            'start of half game', 'substitution', 'saved by goal-keeper', 'red card',
            'lead to corner', 'ball out of play', 'off side', 'goal', 'penalty',
            'yellow card', 'foul lead to penalty', 'corner', 'free kick', 'foul with no card',
        ],
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # train/test 分割
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_test = max(1, int(len(indices) * args.test_ratio))
    train_indices = indices[n_test:]
    test_indices = indices[:n_test]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Step 2: モデル初期化
    print("\n" + "=" * 60)
    print("Step 2: モデル初期化")
    print("=" * 60)
    model = matchvoice_model_all_blocks(
        load_checkpoint=False,
        num_features=768,
        need_temporal='yes',
        llm_ckpt=args.llm_ckpt,
        tokenizer_ckpt=args.llm_ckpt,
        open_llm_decoder=True,
        open_visual_encoder=False,
    )
    model.to(args.device)
    print(f"Model initialized and moved to {args.device}")

    # Step 3: 既存チェックポイントからロード
    print("\n" + "=" * 60)
    print("Step 3: 既存チェックポイントからロード")
    print("=" * 60)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f'Loaded: missing={len(missing)}, unexpected={len(unexpected)}')
    else:
        print(f"WARNING: ckpt_path not found ({args.ckpt_path}), training from scratch")

    # Step 4: collate_fn 定義と pad_token 設定
    print("\n" + "=" * 60)
    print("Step 4: collate_fn 定義と pad_token 設定")
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
        num_workers=0,
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
    model.instruction = INSTRUCTIONS[0]
    torch.save({'state_dict': model.state_dict(), 'instruction': INSTRUCTIONS[0]}, args.out_ckpt)
    print(f"Checkpoint saved: {args.out_ckpt}")
    print("=" * 60)
    print("学習完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
