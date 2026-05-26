#!/usr/bin/env python3
"""Phase 4 Free QA の LLM-as-a-Judge 評価スクリプト"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import urllib.request
import urllib.error


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 Free QA: LLM-as-a-Judge evaluation")
    parser.add_argument("--phase4_dir", type=str, required=True,
                        help="Phase 4 出力ディレクトリ（例: checkpoints/RUN_TS/phase4_TAG）")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Judge LLM モデル名（例: gpt-4o, gpt-4o-mini）")
    parser.add_argument("--base_url", type=str, default="https://models.inference.ai.azure.com",
                        help="API base URL。GitHub Models: https://models.inference.ai.azure.com / OpenAI: https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API キー。省略時は環境変数 GITHUB_TOKEN または OPENAI_API_KEY を使用")
    parser.add_argument("--configs", nargs='+', default=None,
                        help="評価する config 名のリスト（例: qa_formation qa_commentary qa_first_action）")
    parser.add_argument("--spatial_labels", type=str,
                        default="soccerdata_clips/fps1_sec30_onball_step5s/spatial_labels.json",
                        help="spatial_labels.json のパス")
    return parser.parse_args()


def setup_client(base_url: str, api_key: str = None) -> dict:
    """API クライアント設定を返す（標準 urllib のみ使用、外部依存なし）"""
    if api_key is None:
        import os
        api_key = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key not found. Set GITHUB_TOKEN or OPENAI_API_KEY env var, or pass --api_key")
    return {"base_url": base_url.rstrip("/"), "api_key": api_key}


def build_judge_prompt(entry, spatial=None, config_name=""):
    """エントリから Judge プロンプトを構築"""
    action = entry.get('action', '')
    possession = entry.get('possession', '')
    zone = entry.get('zone', '')
    pressure = entry.get('pressure', '')
    instruction = entry.get('instruction', '')
    generated = entry.get('generated', '')
    
    # Ground Truth Facts ブロックの構築
    gt_lines = []
    if spatial:
        if spatial.get("formation_attack"):
            gt_lines.append(f"- Attacking team formation: {spatial['formation_attack']}")
        if spatial.get("formation_defend"):
            gt_lines.append(f"- Defending team formation: {spatial['formation_defend']}")
        if spatial.get("def_line_label"):
            gt_lines.append(f"- Defensive line height: {spatial['def_line_label']} ({spatial['def_line_m']:.1f}m from goal)")

    gt_block = ""
    if gt_lines:
        gt_block = "\n## Ground Truth Facts (from rule-based tracking analysis)\n" + "\n".join(gt_lines) + "\n"
    
    # タスク固有ルールを構築
    task_rule = ""
    if "formation" in config_name and spatial and spatial.get("formation_attack"):
        task_rule = f"\nRULE 5 (formation): Use ## Ground Truth Facts as reference. Score 100 if the format is 'X-Y-Z' and matches the ground truth exactly. Score 75 if the total player count is right but one line differs by 1. Score 50 if the structure is similar but off. Score 25 if the format is correct but content is wrong. Score 0 if format is missing or unrecognizable."
    elif "defensive_line" in config_name and spatial and spatial.get("def_line_label"):
        def_line_order = ["very low", "low", "medium", "high", "very high"]
        task_rule = f"\nRULE 5 (defensive_line): The ground truth is '{spatial['def_line_label']}'. Score 100 if exact match. Score 75 if off by one level (e.g. medium vs high). Score 50 if off by two levels. Score 0 if completely wrong or missing."
    
    prompt = f"""You are a strict evaluator of a soccer video QA model's output.

## Clip Context (from tracking data)
- Possession: {possession}
- Zone: {zone}
- Pressure: {pressure}
- Action sequence: {action}
{gt_block}
## Question given to model
{instruction}

## Model's response
"{generated}"

## Scoring rules (0-100)

RULE 1 — HIGHEST PRIORITY: If the model's response (shown above between double quotes) is
empty or contains only whitespace, you MUST output score=0. No exceptions.

RULE 2: If the response is a repetition loop (e.g. "and learn, and learn, and learn..."),
output score=0.

RULE 3: If the response is off-topic (e.g. answers a different question than asked),
output score=0.

RULE 4: Score 0-100 based on quality:
- 0: empty, garbage, or completely off-topic (covered by RULE 1-3)
- 25: response attempts to answer but is mostly wrong or incoherent
- 50: partially correct or correct but vague/generic
- 75: mostly correct with minor issues (slightly wrong detail, minor format issue)
- 100: accurate, specific, and appropriate format for the question{task_rule}

Output JSON only (no other text):
{{"score": <integer 0-100>, "reason": "<one sentence>"}}"""
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


def judge_entry(entry, client, model_name, spatial=None, config_name=""):
    """エントリを評価"""
    prompt = build_judge_prompt(entry, spatial=spatial, config_name=config_name)
    
    # API 呼び出し（リトライ対応）
    response_text = None
    score = 0.0
    reason = "api error"
    
    url = f"{client['base_url']}/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a strict evaluator of soccer video QA model outputs. Always output valid JSON only, no other text."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 128,
        "temperature": 0
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client['api_key']}"
    }

    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
            response_text = resp_json["choices"][0]["message"]["content"].strip()
            break
        except urllib.error.HTTPError as e:
            wait = 60 if e.code == 429 else 5
            print(f"    HTTP {e.code} (attempt {attempt+1}/5), waiting {wait}s...")
            time.sleep(wait)
            if attempt == 4:
                reason = f"api error: HTTP {e.code}"
        except Exception as e:
            if attempt < 4:
                time.sleep(5)
                continue
            reason = f"api error: {e}"
    
    # JSON を抽出
    if response_text:
        json_obj = extract_json_from_text(response_text)
        
        if json_obj and 'score' in json_obj and 'reason' in json_obj:
            score = int(json_obj['score'])
            reason = str(json_obj['reason'])
            # 0-100 を 0-1 に正規化
            score = max(0.0, min(1.0, score / 100.0))
        else:
            score = 0.0
            reason = "parse error"
    
    return score, reason


def main():
    args = parse_args()
    phase4_dir = Path(args.phase4_dir)
    
    if not phase4_dir.exists():
        print(f"Error: phase4_dir does not exist: {phase4_dir}")
        return
    
    # spatial_labels.json を読み込む（存在しない場合は空 dict）
    spatial_labels = {}
    spatial_labels_path = Path(args.spatial_labels)
    if spatial_labels_path.exists():
        with open(spatial_labels_path, 'r', encoding='utf-8') as f:
            spatial_labels = json.load(f)
        print(f"Loaded spatial labels for {len(spatial_labels)} clips from {spatial_labels_path}")
    else:
        print(f"Warning: spatial_labels not found at {spatial_labels_path}, proceeding without it")
    
    # configs が指定されていない場合、phase4_dir 直下のディレクトリを探す
    if args.configs is None:
        configs = [d.name for d in phase4_dir.iterdir() if d.is_dir()]
        if not configs:
            print(f"Error: No config directories found in {phase4_dir}")
            return
    else:
        configs = args.configs
    
    print(f"Found {len(configs)} configs: {configs}")
    
    # OpenAI 互換 API クライアントをセットアップ
    client = setup_client(args.base_url, args.api_key)
    print(f"Judge model: {args.model} @ {args.base_url}")
    
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
        
        # 既存 judge_results.json があればロードして resume
        judge_json_path = config_dir / 'judge_results.json'
        existing = {}
        if judge_json_path.exists():
            with open(judge_json_path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            # api error でないエントリのみ有効とみなす
            existing = {r['clip_id']: r for r in prev if not str(r.get('reason', '')).startswith('api error')}
            print(f"  Resume: {len(existing)} valid entries already scored")

        # 各エントリを評価
        judge_results = []
        scores = []

        for i, entry in enumerate(entries):
            clip_id = entry.get('clip_id', '')
            # 有効なスコアが既にあればスキップ
            if clip_id in existing:
                result_entry = existing[clip_id]
                judge_results.append(result_entry)
                scores.append(result_entry['score'])
                continue

            spatial = spatial_labels.get(clip_id)
            score, reason = judge_entry(entry, client, args.model, spatial=spatial, config_name=config_name)
            time.sleep(5)  # GitHub Models: 15rpm → 4s/req + margin

            result_entry = entry.copy()
            result_entry['score'] = score
            result_entry['reason'] = reason
            judge_results.append(result_entry)
            scores.append(score)

            # チェックポイント保存（5件ごと）
            if (i + 1) % 5 == 0:
                with open(judge_json_path, 'w', encoding='utf-8') as f:
                    json.dump(judge_results, f, ensure_ascii=False, indent=2)

            if (i + 1) % 10 == 0 or i == len(entries) - 1:
                print(f"  [{i+1}/{len(entries)}] processed")
        
        # 最終保存
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
