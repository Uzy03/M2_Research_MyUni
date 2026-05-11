import argparse
import csv
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

import torch
from rouge_score import rouge_scorer as rs
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.dataset.multitask_dataset import MultiTaskDataset, TASKS, ACTION_VOCAB
from model.matchvoice_model_tracking import matchvoice_model_tracking

STAGE_TASKS = [
    ['action'],
    ['action', 'possession'],
    ['action', 'possession', 'zone'],
    ['action', 'possession', 'zone', 'pressure'],
]


def make_collate_fn(tokenizer, max_length, use_ans_token=False, ans_token_id=None,
                    use_chat_template=False, asst_header_ids=None, no_instruction=False):
    def collate_fn(batch):
        feats, masks, instructions, target_texts, task_names, seq_ids, slot_labels_list = zip(*batch)
        tracking = torch.stack(feats)
        mask_tensor = torch.stack(masks)

        input_ids_list, labels_list, attn_list = [], [], []
        bos_id = tokenizer.bos_token_id
        for instruction, target_text in zip(instructions, target_texts):
            if no_instruction:
                instruction = ''
            inst_ids = tokenizer(instruction, add_special_tokens=False).input_ids
            ans_ids  = tokenizer(target_text + tokenizer.eos_token, add_special_tokens=False).input_ids
            if use_chat_template and asst_header_ids:
                full_ids = ([bos_id] + inst_ids + asst_header_ids + ans_ids)[:max_length]
                lbl      = ([-100]   + [-100] * len(inst_ids) + [-100] * len(asst_header_ids) + ans_ids)[:max_length]
            elif use_ans_token and ans_token_id is not None:
                full_ids = ([bos_id] + inst_ids + [ans_token_id] + ans_ids)[:max_length]
                lbl      = ([-100]   + [-100] * len(inst_ids) + [-100] + ans_ids)[:max_length]
            else:
                full_ids = ([bos_id] + inst_ids + ans_ids)[:max_length]
                lbl      = ([-100]   + [-100] * len(inst_ids) + ans_ids)[:max_length]
            pad_len  = max_length - len(full_ids)
            attn     = [1] * len(full_ids) + [0] * pad_len
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            lbl      = lbl + [-100] * pad_len
            input_ids_list.append(full_ids)
            labels_list.append(lbl)
            attn_list.append(attn)

        return {
            "tracking":       tracking,
            "mask":           mask_tensor,
            "input_ids":      torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_list, dtype=torch.long),
            "labels":         torch.tensor(labels_list, dtype=torch.long),
            "caption_text":   list(target_texts),
            "video_path":     list(seq_ids),
            "instruction":    list(instructions),
            "task_name":      list(task_names),
            "slot_labels":    list(slot_labels_list),
        }
    return collate_fn


_rouge_scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)
_ACTION_VOCAB_SET = set(ACTION_VOCAB)


def compute_rouge_l(pred: str, gt: str):
    if not gt.strip():
        return None
    return _rouge_scorer.score(gt, pred)['rougeL'].fmeasure


def compute_f1_action(pred: str, gt: str):
    if not gt.strip():
        return None
    gt_labels   = {w for w in _ACTION_VOCAB_SET if w in gt}
    pred_labels = {w for w in _ACTION_VOCAB_SET if w in pred}
    if not gt_labels:
        return None
    if not pred_labels:
        return 0.0
    tp = len(pred_labels & gt_labels)
    p  = tp / len(pred_labels)
    r  = tp / len(gt_labels)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_metrics(model, dataset, device, max_eval=200, seed=0, allowed_tasks=None, no_instruction=False):
    """Generate predictions and compute per-task metrics (F1 for action, ROUGE-L for others)."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*attention mask.*")
    model.eval()
    active_tasks = [t for t in TASKS if allowed_tasks is None or t['name'] in allowed_tasks]
    task_scores = {t['name']: [] for t in active_tasks}

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    if len(indices) > max_eval:
        indices = rng.sample(indices, max_eval)

    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            feat, msk, instruction, answer, task_name, seq_id, _ = item
            if task_name not in task_scores:
                continue
            model.instruction = '' if no_instruction else instruction
            samples = {
                "tracking":       feat.unsqueeze(0).to(device),
                "mask":           msk.unsqueeze(0).to(device),
                "caption_text":   [answer],
                "video_path":     [seq_id],
                "labels":         torch.zeros(1, 1, dtype=torch.long).to(device),
                "attention_mask": torch.ones(1, 1, dtype=torch.long).to(device),
                "input_ids":      torch.zeros(1, 1, dtype=torch.long).to(device),
            }
            generated_list, _, _ = model(samples, validating=True)
            gen = generated_list[0] if generated_list else ""
            if task_name == 'action':
                score = compute_f1_action(gen, answer)
            else:
                score = compute_rouge_l(gen, answer)
            if score is not None:
                task_scores[task_name].append(score)

    model.train()
    return {
        t: (sum(s) / len(s) if s else float('nan'))
        for t, s in task_scores.items()
    }


def run_curriculum_training(args, model, full_dataset_all,
                            train_indices, val_indices, test_indices,
                            stage_epochs, collate_fn):
    log_csv = Path(args.out_ckpt).with_suffix('.curriculum_log.csv')
    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["stage", "epoch", "train_loss", "val_loss"])

    for stage_idx, task_names in enumerate(STAGE_TASKS):
        stage_num = stage_idx + 1
        n_epochs  = stage_epochs[stage_idx]
        print("\n" + "=" * 60)
        print(f"Curriculum Stage {stage_num}/4: tasks={task_names}  epochs={n_epochs}")
        print("=" * 60)

        stage_dataset = MultiTaskDataset(
            args.json_path, context_len=args.context_len, max_games=args.max_games,
            use_short_instruction=args.short_instruction, allowed_tasks=task_names,
            use_instruction_diverse=args.instruction_diverse,
            use_answer_diverse=args.answer_diverse,
        )
        stage_train = Subset(stage_dataset, train_indices)
        stage_val   = Subset(stage_dataset, val_indices)
        train_loader = DataLoader(stage_train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(stage_val,   batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_fn, num_workers=4, pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        if stage_num < 4:
            stage_ckpt = Path(args.out_ckpt).parent / f"action_alignment_stage{stage_num}.pth"
        else:
            stage_ckpt = Path(args.out_ckpt)

        best_val = float('inf')

        for epoch in range(1, n_epochs + 1):
            model.train()
            total_train, n_train = 0.0, 0
            for batch in train_loader:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                optimizer.zero_grad()
                loss = model(batch, validating=False, lambda_align=args.lambda_align, lambda_slot=args.lambda_slot)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train += loss.item()
                n_train += 1
            avg_train = total_train / max(1, n_train)
            scheduler.step()

            model.eval()
            total_val, n_val = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    loss = model(batch, validating=False, lambda_align=args.lambda_align, lambda_slot=args.lambda_slot)
                    if not torch.isnan(loss):
                        total_val += loss.item()
                        n_val += 1
            avg_val = total_val / max(1, n_val)

            mark = ""
            if avg_val < best_val:
                best_val = avg_val
                save_state = {k: v for k, v in model.state_dict().items()
                              if not k.startswith('llama_model.')}
                torch.save({"state_dict": save_state}, stage_ckpt)
                mark = " ← best"

            print(f"  [Stage {stage_num}] Epoch {epoch}/{n_epochs}"
                  f"  train={avg_train:.4f}  val={avg_val:.4f}{mark}")
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([stage_num, epoch, f"{avg_train:.6f}", f"{avg_val:.6f}"])

        print(f"Stage {stage_num} best val: {best_val:.4f}  Saved: {stage_ckpt}")

        loaded = torch.load(stage_ckpt, map_location="cpu")
        model.load_state_dict(loaded["state_dict"], strict=False)
        model.to(args.device)
        test_subset = Subset(full_dataset_all, test_indices)
        test_r = evaluate_metrics(model, test_subset, args.device)
        print(
            f"[Stage {stage_num}] Test"
            f"  f1_action={test_r['action']:.4f}"
            f"  rouge_possession={test_r['possession']:.4f}"
            f"  rouge_zone={test_r['zone']:.4f}"
            f"  rouge_pressure={test_r['pressure']:.4f}"
        )

    print("\n" + "=" * 60)
    print(f"カリキュラム学習完了！最終ckpt: {args.out_ckpt}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Multi-task action alignment with optional LoRA"
    )
    parser.add_argument("--json_path",     type=str,   default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json")
    parser.add_argument("--ckpt_path",     type=str,   default="checkpoints/trajectory_regression.pth")
    parser.add_argument("--llm_ckpt",      type=str,   default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--out_ckpt",      type=str,   default="checkpoints/action_alignment.pth")
    parser.add_argument("--context_len",   type=int,   default=20)
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--max_length",    type=int,   default=128)
    parser.add_argument("--test_ratio",    type=float, default=0.1,
                        help="Val/test data ratio each (train = 1 - 2*test_ratio)")
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--max_games",     type=int,   default=0)
    parser.add_argument("--max_samples",   type=int,   default=0,
                        help="Cap total samples (0=all). Useful for smoke tests.")
    parser.add_argument("--open_lora",     action="store_true",
                        help="Enable LoRA to partially unfreeze LLM")
    parser.add_argument("--lora_rank",     type=int,   default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--use_ans_token", action="store_true",
                        help="Insert <ANS> token between instruction and answer")
    parser.add_argument("--qformer_heads", type=int, default=1,
                        help="Multi-head Q-Former heads (1=baseline, 4 or 8 for ablation)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use LLaMA-3 assistant header as answer boundary signal")
    parser.add_argument("--short_instruction", action="store_true",
                        help="Use shortened instruction texts to reduce token count")
    parser.add_argument("--curriculum_stages", type=str, default=None,
                        help="カリキュラム学習のエポック数（例: '5,5,5,5'）。省略時は joint training。")
    parser.add_argument("--allowed_tasks", type=str, default=None,
                        help="学習・評価するタスク（カンマ区切り、例: action）。省略時は全タスク")
    parser.add_argument('--no_instruction', action='store_true',
                        help='指示文なしでトラッキング埋め込みのみをLLMに渡す')
    parser.add_argument('--sentence_format', action='store_true',
                        help='アクションラベルを自然文に変換して学習・評価する')
    parser.add_argument('--instruction_diverse', action='store_true',
                        help='アクション指示文を複数バリエーションからランダムサンプリングする')
    parser.add_argument('--answer_diverse', action='store_true',
                        help='アクション回答を複数フォーマットからランダムサンプリングする')
    parser.add_argument('--lambda_align', type=float, default=0.0,
                        help='Token-Word Alignment loss weight (0.0 = disabled)')
    parser.add_argument('--lambda_slot', type=float, default=0.0,
                        help='per-slot alignment loss の係数（B-2）')
    parser.add_argument('--test_only', action='store_true',
                        help='学習をスキップして --out_ckpt からロードしてテスト評価のみ実行')
    parser.add_argument('--open_visual_encoder', action='store_true',
                        help='TrackingEncoder を解凍して Phase 2 でも学習する（Approach B\'）')
    parser.add_argument('--lr_encoder', type=float, default=1e-5,
                        help='open_visual_encoder 時の Encoder 専用学習率（デフォルト: 1e-5）')
    args = parser.parse_args()

    allowed_tasks = [t.strip() for t in args.allowed_tasks.split(',')] if args.allowed_tasks else None

    qformer_heads = args.qformer_heads
    if args.lambda_slot > 0.0 and qformer_heads < 4:
        qformer_heads = 4

    print("=" * 60)
    print("Step 1: データセット読み込み・train/val/test 分割")
    print("=" * 60)
    random.seed(args.seed)
    full_dataset = MultiTaskDataset(args.json_path, args.context_len, max_games=args.max_games,
                                    use_short_instruction=args.short_instruction,
                                    allowed_tasks=allowed_tasks, use_sentence_format=args.sentence_format,
                                    use_instruction_diverse=args.instruction_diverse,
                                    use_answer_diverse=args.answer_diverse)
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
    print(f"Total: {len(full_dataset)}  Train: {len(train_indices)}  Val: {len(val_indices)}  Test: {len(test_indices)}")

    split_json = Path(args.out_ckpt).parent / "action_alignment_splits.json"
    split_json.parent.mkdir(parents=True, exist_ok=True)
    with open(split_json, 'w') as f:
        json.dump({"train": train_indices, "val": val_indices, "test": test_indices}, f)
    print(f"Splits saved: {split_json}")

    print("\n" + "=" * 60)
    print("Step 2: モデル初期化")
    print("=" * 60)
    model = matchvoice_model_tracking(
        load_checkpoint=False,
        num_features=768,
        need_temporal="yes",
        llm_ckpt=args.llm_ckpt,
        tokenizer_ckpt=args.llm_ckpt,
        open_llm_decoder=args.open_lora,
        llm_lora_rank=args.lora_rank,
        use_ans_token=args.use_ans_token,
        qformer_heads=qformer_heads,
        use_chat_template=args.use_chat_template,
        open_visual_encoder=args.open_visual_encoder,
        num_players=23,
        in_features=5,
        d_model=256,
        max_frame_pos=200,
    )
    model.to(args.device)
    print(f"LoRA: {'enabled' if args.open_lora else 'disabled'}")
    enc_status = "unfrozen (Approach B')" if args.open_visual_encoder else "frozen"
    print(f"Encoder: {enc_status}")

    print("\n" + "=" * 60)
    print("Step 3: Phase 1 チェックポイントロード")
    print("=" * 60)
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        remap_prefix = {"tracking_encoder.": "visual_encoder.", "qformer.": "video_Qformer."}
        remap_exact  = {"query_tokens": "video_query_tokens"}
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
        filtered = {k: v for k, v in remapped.items() if k in model_state and model_state[k].shape == v.shape}
        skipped = [k for k, v in remapped.items() if k in model_state and model_state[k].shape != v.shape]
        if skipped:
            print(f"Skipping size-mismatched keys: {skipped}")
        missing, _ = model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} keys from Phase 1 checkpoint, missing: {len(missing)}")
    else:
        print(f"WARNING: ckpt_path not found ({args.ckpt_path}), training from scratch")

    print("\n" + "=" * 60)
    print("Step 4: DataLoader 作成")
    print("=" * 60)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    collate_fn = make_collate_fn(
        model.tokenizer, args.max_length,
        use_ans_token=args.use_ans_token,
        ans_token_id=model.ans_token_id,
        use_chat_template=args.use_chat_template,
        asst_header_ids=model._asst_header_ids,
        no_instruction=args.no_instruction,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_loader)} batches  Val: {len(val_loader)} batches")

    print("\n" + "=" * 60)
    if args.test_only:
        print("Step 5: スキップ（--test_only）")
        print("=" * 60)
    elif args.curriculum_stages is not None:
        print("Step 5: カリキュラム訓練")
        print("=" * 60)
        stage_epochs = [int(x) for x in args.curriculum_stages.split(",")]
        run_curriculum_training(
            args, model,
            full_dataset,
            train_indices, val_indices, test_indices,
            stage_epochs, collate_fn,
        )
    else:
        print("Step 5: 訓練ループ")
        print("=" * 60)
        if args.open_visual_encoder:
            other_params = [p for n, p in model.named_parameters()
                            if not n.startswith('visual_encoder.') and p.requires_grad]
            optimizer = torch.optim.Adam([
                {'params': model.visual_encoder.parameters(), 'lr': args.lr_encoder},
                {'params': other_params, 'lr': args.lr},
            ])
        else:
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        log_csv = Path(args.out_ckpt).with_suffix('.train_log.csv')
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

        best_val = float('inf')
        Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            # --- train ---
            model.train()
            total_train = 0.0
            for batch in train_loader:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad()
                loss = model(batch, validating=False, lambda_align=args.lambda_align, lambda_slot=args.lambda_slot)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train += loss.item()
            avg_train = total_train / len(train_loader)
            scheduler.step()

            # --- val loss ---
            model.eval()
            total_val = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    loss = model(batch, validating=False, lambda_align=args.lambda_align, lambda_slot=args.lambda_slot)
                    if not torch.isnan(loss):
                        total_val += loss.item()
                        n_val_batches += 1
            avg_val = total_val / max(1, n_val_batches)

            mark = ""
            if avg_val < best_val:
                best_val = avg_val
                save_state = {k: v for k, v in model.state_dict().items() if not k.startswith('llama_model.')}
                torch.save({"state_dict": save_state}, args.out_ckpt)
                mark = " ← best"

            print(f"Epoch {epoch}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}{mark}")
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{avg_train:.6f}", f"{avg_val:.6f}"])

        print(f"Best val loss: {best_val:.4f}  Checkpoint: {args.out_ckpt}")

    if args.curriculum_stages is None:
        print("\n" + "=" * 60)
        print("Step 6: テスト評価")
        print("=" * 60)
        # best checkpoint をロードしてテスト評価
        ckpt = torch.load(args.out_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.to(args.device)
        test_r = evaluate_metrics(model, test_dataset, args.device, allowed_tasks=allowed_tasks,
                                  no_instruction=args.no_instruction)
        for name, score in test_r.items():
            metric = 'f1' if name == 'action' else 'rouge_l'
            print(f"  Test {name} {metric}={score:.4f}")

        print("\n" + "=" * 60)
        print("Step 7: チェックポイント保存済み（best val epoch）")
        print("=" * 60)
        print(f"Checkpoint saved: {args.out_ckpt}")
        print("=" * 60)
        print("学習完了！")
        print("=" * 60)


if __name__ == "__main__":
    main()
