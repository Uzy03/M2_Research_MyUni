import json
import argparse
import sys, os
from pathlib import Path

# Add parent directory to sys.path for ACTION_NAMES_EN import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.dataset.multitask_dataset import ACTION_NAMES_EN

SYSTEM_PROMPT = (
    "You are an AI assistant specialized in soccer match analysis. "
    "You will receive structured labels from a 30-second soccer tracking clip. "
    "Design your responses as if you can actually watch this soccer sequence. "
    "Cover diverse aspects: actions performed, timing, ball possession, field position, "
    "pressing intensity, and tactical implications. "
    "Only include questions and answers that are clearly supported by the provided data. "
    "For complex reasoning questions, provide step-by-step logical explanations "
    "with concrete supporting evidence from the clip data."
)

# --- LLaVA-style few-shot example ---
FEW_SHOT_EXAMPLE = """
=== EXAMPLE INPUT ===
Clip data (30-second soccer sequence):
- Actions: pass, through pass (timing: pass at ~3s, through pass at ~11s)
- Ball possession: The home team has ball possession in this sequence.
- Field position: The play is in the center of the middle third.
- Pressing intensity: The players are applying medium pressure around the ball.

=== EXAMPLE OUTPUT ===
{
  "conversation": [
    {"from": "human", "value": "What actions are being performed in this soccer clip?"},
    {"from": "gpt", "value": "The home team performs a pass at around 3 seconds, followed by a through pass at around 11 seconds into the sequence."},
    {"from": "human", "value": "How intense is the defensive pressure in this clip?"},
    {"from": "gpt", "value": "The players are applying medium pressure around the ball, meaning the defending team is actively contesting possession but has not fully committed to a high press."}
  ],
  "description": {
    "instruction": "Describe what is happening in this soccer sequence in detail.",
    "answer": "In this 30-second clip, the home team maintains ball possession in the central middle third. Starting at around 3 seconds, a pass is played to a teammate, who then plays a through pass at approximately 11 seconds, attempting to break through the midfield line. Throughout the sequence, the away team applies medium pressure, creating a competitive midfield contest while the home team works to advance play."
  },
  "reasoning": {
    "instruction": "What tactical objective is the home team pursuing, and what risks do they face?",
    "answer": "The home team is attempting a line-breaking combination: a short pass followed by a penetrating through pass. This suggests they are trying to exploit space behind the midfield block. However, under medium defensive pressure, executing this combination requires precise timing — a mistimed through pass in the central third could result in a dangerous turnover and a counter-attack opportunity for the away team."
  }
}
"""

def format_action_timing(action_sequence, action_sequence_frames):
    """action IDs + clip frame indices → readable timing string"""
    if not action_sequence or not action_sequence_frames:
        return None
    parts = []
    for action_id, frame in zip(action_sequence, action_sequence_frames):
        name = ACTION_NAMES_EN.get(str(action_id), f"action_{action_id}")
        parts.append(f"{name} at ~{frame}s")
    return ", ".join(parts)

def build_prompt(label_action, label_possession, label_zone, label_pressure, timing=None) -> str:
    action_detail = label_action
    if timing:
        action_detail = f"{label_action} (timing: {timing})"
    return f"""{FEW_SHOT_EXAMPLE}
=== NOW GENERATE FOR THE FOLLOWING CLIP ===
Clip data (30-second soccer sequence):
- Actions: {action_detail}
- Ball possession: {label_possession}
- Field position: {label_zone}
- Pressing intensity: {label_pressure}

Generate the same 3 types as the example above. Output ONLY the JSON object, no other text.
Rules:
1. conversation: exactly 2 human/gpt turn pairs (4 items total)
2. description: one instruction and one detailed answer (3-4 sentences)
3. reasoning: one question requiring tactical analysis and one logical step-by-step answer
4. Only mention facts clearly supported by the provided clip data
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="soccerdata_clips/fps1_sec30_onball_step5s/clips.json")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_games", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()

def is_valid_llm_qa(llm_qa):
    if not isinstance(llm_qa, list) or len(llm_qa) != 3:
        return False
    conv, desc, reas = llm_qa
    if conv.get("type") != "conversation" or not isinstance(conv.get("turns"), list):
        return False
    if desc.get("type") != "description" or not desc.get("instruction") or not desc.get("answer"):
        return False
    if reas.get("type") != "reasoning" or not reas.get("instruction") or not reas.get("answer"):
        return False
    return True

def main():
    args = parse_args()
    try:
        with open(args.json_path, encoding="utf-8") as f:
            clips = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load {args.json_path}: {e}")
        return

    if not args.dry_run:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        llm = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto"
        )
        llm.eval()
        print("Model loaded.")

    unique_games = set()
    generated = 0

    for i, entry in enumerate(clips):
        game_id = entry.get("game_id")
        if args.max_games > 0:
            if game_id not in unique_games:
                if len(unique_games) >= args.max_games:
                    continue
                unique_games.add(game_id)

        # Skip if already has valid llm_qa
        if is_valid_llm_qa(entry.get("llm_qa")):
            continue

        label_action = entry.get("label_action") or "no notable actions"
        label_possession = entry.get("label_possession", "")
        label_zone = entry.get("label_zone", "")
        label_pressure = entry.get("label_pressure", "")
        timing = format_action_timing(
            entry.get("action_sequence", []),
            entry.get("action_sequence_frames", [])
        )

        if args.dry_run:
            entry["llm_qa"] = [
                {"type": "conversation", "turns": [
                    {"from": "human", "value": "What is happening in this soccer sequence?"},
                    {"from": "gpt",   "value": f"[dry_run] Actions: {label_action}."},
                    {"from": "human", "value": "Which team has the ball?"},
                    {"from": "gpt",   "value": f"[dry_run] {label_possession}"},
                ]},
                {"type": "description",
                 "instruction": "Describe this soccer sequence in detail.",
                 "answer": f"[dry_run] {label_possession} {label_zone} {label_pressure}"},
                {"type": "reasoning",
                 "instruction": "Analyze the tactical situation in this sequence.",
                 "answer": "[dry_run] Tactical analysis placeholder."},
            ]
        else:
            prompt = build_prompt(label_action, label_possession, label_zone, label_pressure, timing)
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ]
                text_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text_input, return_tensors="pt").to(llm.device)
                with torch.no_grad():
                    output_ids = llm.generate(
                        **inputs,
                        max_new_tokens=600,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                qa_data = json.loads(raw)
                # conversation: turns list
                conv = qa_data["conversation"]
                conv_entry = {"type": "conversation", "turns": conv if isinstance(conv, list) else []}
                desc_entry = {"type": "description",
                              "instruction": qa_data["description"]["instruction"],
                              "answer":      qa_data["description"]["answer"]}
                reas_entry = {"type": "reasoning",
                              "instruction": qa_data["reasoning"]["instruction"],
                              "answer":      qa_data["reasoning"]["answer"]}
                entry["llm_qa"] = [conv_entry, desc_entry, reas_entry]
            except Exception as e:
                print(f"WARNING: clip {entry.get('clip_id')} failed: {e}")
                entry["llm_qa"] = []
                continue

        generated += 1
        if i % 50 == 0:
            print(f"Progress: {i+1}/{len(clips)}, generated={generated}")

        if args.save_interval > 0 and generated % args.save_interval == 0:
            with open(args.json_path, "w", encoding="utf-8") as f:
                json.dump(clips, f, indent=2, ensure_ascii=False)
            print(f"[checkpoint] saved {generated} generated so far...")

    with open(args.json_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2, ensure_ascii=False)
    print(f"Done. Total generated: {generated}")

if __name__ == "__main__":
    main()
