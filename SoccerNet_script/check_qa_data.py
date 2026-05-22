"""
check_qa_data.py - generated llm_qa の品質確認スクリプト
Usage: python SoccerNet_script/check_qa_data.py --json_path <path> [--n 3]
"""
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json")
    p.add_argument("--n", type=int, default=3, help="表示件数")
    return p.parse_args()


def check_entry(c):
    """1エントリの品質チェック。問題があれば警告文字列のリストを返す"""
    issues = []
    qa_list = c.get("llm_qa", [])
    if len(qa_list) != 3:
        issues.append(f"llm_qa length={len(qa_list)} (expected 3)")
        return issues

    conv, desc, reas = qa_list

    # conversation: turns形式・2往復
    turns = conv.get("turns", [])
    if len(turns) != 4:
        issues.append(f"conversation turns={len(turns)} (expected 4)")
    for i, t in enumerate(turns):
        if not t.get("value"):
            issues.append(f"turns[{i}].value is empty")

    # description/reasoning: instruction & answer 存在
    for label, qa in [("description", desc), ("reasoning", reas)]:
        if not qa.get("instruction"):
            issues.append(f"{label}.instruction is empty")
        if not qa.get("answer"):
            issues.append(f"{label}.answer is empty")
        if len(qa.get("answer", "")) < 30:
            issues.append(f"{label}.answer too short ({len(qa.get('answer',''))} chars)")

    return issues


def main():
    args = parse_args()
    with open(args.json_path, encoding="utf-8") as f:
        clips = json.load(f)

    total = len(clips)
    has_qa = [c for c in clips if c.get("llm_qa")]
    valid  = [c for c in has_qa if len(c.get("llm_qa", [])) == 3]
    issues_count = sum(1 for c in valid if check_entry(c))

    print(f"=== 統計 ===")
    print(f"総クリップ数  : {total}")
    print(f"llm_qa あり   : {len(has_qa)}")
    print(f"フォーマット正常: {len(valid)}")
    print(f"内容に問題あり : {issues_count}")
    print()

    # 正常サンプルを n 件表示
    shown = 0
    for c in valid:
        if check_entry(c):
            continue
        print(f"{'='*60}")
        print(f"clip_id : {c.get('clip_id')}")
        print(f"labels  : action={c.get('label_action')} | possession={c.get('label_possession')}")
        print(f"          zone={c.get('label_zone')} | pressure={c.get('label_pressure')}")
        print()

        conv, desc, reas = c["llm_qa"]
        print("[conversation]")
        for t in conv.get("turns", []):
            role = "Human" if t["from"] == "human" else "  GPT"
            print(f"  {role}: {t['value']}")
        print()
        print("[description]")
        print(f"  Q: {desc['instruction']}")
        print(f"  A: {desc['answer']}")
        print()
        print("[reasoning]")
        print(f"  Q: {reas['instruction']}")
        print(f"  A: {reas['answer']}")
        print()

        shown += 1
        if shown >= args.n:
            break

    # 問題サンプルを最大3件表示
    bad_shown = 0
    for c in valid:
        issues = check_entry(c)
        if not issues:
            continue
        if bad_shown == 0:
            print(f"\n{'='*60}")
            print("=== 問題サンプル ===")
        print(f"clip_id: {c.get('clip_id')}  issues: {issues}")
        bad_shown += 1
        if bad_shown >= 3:
            break


if __name__ == "__main__":
    main()
