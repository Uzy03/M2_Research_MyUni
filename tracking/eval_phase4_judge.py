#!/usr/bin/env python3
"""Phase 4 Free QA の LLM-as-a-Judge 評価スクリプト"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 Free QA: LLM-as-a-Judge evaluation")
    parser.add_argument("--phase4_dir", type=str, required=True,
                        help="Phase 4 出力ディレクトリ（例: checkpoints/RUN_TS/phase4_TAG）")
    parser.add_argument("--llm_ckpt", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Judge LLM のチェックポイント")
    parser.add_argument("--configs", nargs='+', default=None,
                        help="評価する config 名のリスト（例: qa_formation qa_commentary qa_first_action）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda / cpu")
    parser.add_argument("--gpu", type=int, default=0,
                        help="使用 GPU 番号")
    return parser.parse_args()


def load_judge_model(llm_ckpt, device):
    """Judge LLM をロード"""
    print(f"Loading Judge LLM: {llm_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(llm_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        llm_ckpt,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


def build_judge_prompt(entry):
    """エントリから Judge プロンプトを構築"""
    action = entry.get('action', '')
    possession = entry.get('possession', '')
    zone = entry.get('zone', '')
    pressure = entry.get('pressure', '')
    instruction = entry.get('instruction', '')
    generated = entry.get('generated', '')
    
    prompt = f"""You are evaluating a soccer video QA model's output.

## Clip Context (ground truth labels from tracking data)
- Actions: {action}
- Possession: {possession} team
- Zone: {zone}
- Pressure: {pressure}

## Question given to model
{instruction}

## Model's response
{generated}

## Scoring criteria
- 3: Accurate and specific (consistent with clip metadata, appropriate format)
- 2: Mostly accurate but minor format issues or partial
- 1: Format followed but generic/templated content
- 0: Empty, garbage, completely off-topic, or repetition loop

Output JSON only (no other text):
{{"score": <int 0-3>, "reason": "<brief reason>"}}"""
    return prompt


def extract_json_from_text(text):
    """テキストから JSON を抽出"""
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def judge_entry(entry, tokenizer, model, device):
    """エントリを評価"""
    prompt = build_judge_prompt(entry)
    
    # プロンプトをトークン化
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # LLM で生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0,  # greedy
            do_sample=False,
        )
    
    # 生成テキストをデコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 最後の部分（プロンプト以降）を抽出
    response_text = generated_text[len(prompt):].strip() if len(generated_text) > len(prompt) else ""
    
    # JSON を抽出
    json_obj = extract_json_from_text(response_text)
    
    if json_obj and 'score' in json_obj and 'reason' in json_obj:
        score = int(json_obj['score'])
        reason = str(json_obj['reason'])
        # スコアを0-3の範囲に制限
        score = max(0, min(3, score))
    else:
        score = 0
        reason = "parse error"
    
    return score, reason


def main():
    args = parse_args()
    phase4_dir = Path(args.phase4_dir)
    
    if not phase4_dir.exists():
        print(f"Error: phase4_dir does not exist: {phase4_dir}")
        return
    
    # configs が指定されていない場合、phase4_dir 直下のディレクトリを探す
    if args.configs is None:
        configs = [d.name for d in phase4_dir.iterdir() if d.is_dir()]
        if not configs:
            print(f"Error: No config directories found in {phase4_dir}")
            return
    else:
        configs = args.configs
    
    print(f"Found {len(configs)} configs: {configs}")
    
    # Judge LLM をロード
    tokenizer, model = load_judge_model(args.llm_ckpt, args.device)
    device = args.device if args.device == "cpu" else f"{args.device}:{args.gpu}"
    
    summary_data = []
    
    for config_name in configs:
        config_dir = phase4_dir / config_name
        json_path = config_dir / 'results.json'
        
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping {config_name}")
            continue
        
        print(f"\n=== Evaluating {config_name} ===")
        
        # results.json を読み込む
        with open(json_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        print(f"Loaded {len(entries)} entries from {json_path}")
        
        # 各エントリを評価
        judge_results = []
        scores = []
        
        for i, entry in enumerate(entries):
            score, reason = judge_entry(entry, tokenizer, model, device)
            
            result_entry = entry.copy()
            result_entry['score'] = score
            result_entry['reason'] = reason
            judge_results.append(result_entry)
            scores.append(score)
            
            if (i + 1) % 10 == 0 or i == len(entries) - 1:
                print(f"  [{i+1}/{len(entries)}] processed")
        
        # judge_results.json を保存
        judge_json_path = config_dir / 'judge_results.json'
        with open(judge_json_path, 'w', encoding='utf-8') as f:
            json.dump(judge_results, f, ensure_ascii=False, indent=2)
        print(f"Saved judge_results.json: {judge_json_path}")
        
        # サマリー統計
        mean_score = sum(scores) / len(scores) if scores else 0.0
        summary_data.append({
            'config': config_name,
            'n': len(scores),
            'mean_score': mean_score
        })
        
        print(f"Mean score: {mean_score:.4f} (n={len(scores)})")
    
    # judge_summary.csv を保存
    summary_csv_path = phase4_dir / 'judge_summary.csv'
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'n', 'mean_score'])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"\nSaved judge_summary.csv: {summary_csv_path}")


if __name__ == '__main__':
    main()
