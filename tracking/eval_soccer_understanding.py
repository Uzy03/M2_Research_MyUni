import argparse
import json
import os
import re
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BENCHMARK = [
    # --- Dimension 1: Rules / Logic ---
    {"id": "D1Q1", "dim": 1,
     "question": "A defender intentionally back-passes to the goalkeeper with his foot. The goalkeeper picks it up by hand inside the penalty area. What is the correct restart?",
     "ground_truths": ["indirect free kick"]},
    {"id": "D1Q2", "dim": 1,
     "question": "A player receives the ball directly from a throw-in while positioned behind the second-to-last defender of the opposing team. What is the correct referee decision?",
     "ground_truths": ["play continues", "no offside", "legal play", "not offside"]},
    {"id": "D1Q3", "dim": 1,
     "question": "A defending player commits a foul exactly on the line of their own penalty area. What is the correct restart?",
     "ground_truths": ["penalty kick", "penalty"]},
    {"id": "D1Q4", "dim": 1,
     "question": "A goalkeeper takes a goal kick, but strong wind blows the ball directly into their own goal without touching any other player. What is the correct restart?",
     "ground_truths": ["corner kick", "corner"]},
    # --- Dimension 2: Spatial / Formation ---
    {"id": "D2Q1", "dim": 2,
     "question": "A team has 4 defenders in a flat line near their own box, 1 holding midfielder in front of them, 2 central midfielders further up, 2 wide midfielders, and 1 striker. What formation is this? Answer only with the formation code (e.g. 4-4-2).",
     "regex": r"4-1-4-1|4-3-3|4-1-2-2-1"},
    {"id": "D2Q2", "dim": 2,
     "question": "A team is lined up with 3 central defenders, 2 wing-backs pushing high on the flanks, 2 central midfielders, 2 attacking midfielders playing just behind 1 central striker. What formation is this? Answer only with the formation code.",
     "regex": r"3-4-2-1|5-2-2-1|3-4-3"},
    {"id": "D2Q3", "dim": 2,
     "question": "The defending team has 4 defenders, a midfield diamond consisting of 1 defensive midfielder, 2 central midfielders, and 1 attacking midfielder, supporting 2 strikers up front. What formation is this? Answer only with the formation code.",
     "regex": r"4-4-2|4-1-2-1-2"},
    {"id": "D2Q4", "dim": 2,
     "question": "A team plays with 4 defenders, 2 holding defensive midfielders side-by-side, 3 attacking midfielders (1 central, 2 wide), and 1 main striker. What formation is this? Answer only with the formation code.",
     "regex": r"4-2-3-1"},
    {"id": "D2Q5", "dim": 2,
     "question": "The lineup features 3 central defenders, 1 holding midfielder, 2 wide midfielders, 2 central midfielders, and 2 strikers. What formation is this? Answer only with the formation code.",
     "regex": r"3-1-4-2|3-5-2"},
    # --- Dimension 3: Tactical Reasoning ---
    {"id": "D3Q1", "dim": 3,
     "question": "The attacking team continuously passes on the left flank, drawing the opponent's defensive block heavily to that side. Suddenly, they play a long diagonal pass to their right winger who is left unguarded. What is the tactical intent of this sequence?",
     "keyword_groups": [
         ["overload", "draw defense", "draw the opponent", "draw defenders"],
         ["isolate", "1v1", "one on one", "unguarded", "one-on-one"],
         ["switch of play", "opposite flank", "weak side", "switch play", "change the point"],
     ]},
    {"id": "D3Q2", "dim": 3,
     "question": "A central striker deliberately drops deep into the midfield during build-up play, pulling an opposing center-back out of the defensive line. What is this specific player role called, and what is its spatial purpose?",
     "keyword_groups": [
         ["false 9", "false nine"],
         ["creating space", "opening gaps", "space in behind", "gap in the defensive line", "open space"],
         ["midfield overload", "numerical superiority", "overload in midfield", "extra man in midfield"],
     ]},
    {"id": "D3Q3", "dim": 3,
     "question": "A defending team allows the opposing center-backs to pass the ball freely but aggressively closes down the fullbacks the moment they receive a pass near the touchline. Why is the touchline used as a tactical tool in this specific pressing strategy?",
     "keyword_groups": [
         ["pressing trap", "press trap", "trap"],
         ["touchline", "extra defender", "limits passing", "restricts options", "narrows options", "180 degree", "180-degree"],
         ["forcing turnovers", "winning the ball", "regain possession", "high press", "high up the pitch"],
     ]},
    {"id": "D3Q4", "dim": 3,
     "question": "A team is playing a very high defensive line against an opponent with exceptionally fast wingers. What is the primary spatial vulnerability, and what adjustment should the goalkeeper make?",
     "keyword_groups": [
         ["space behind", "space in behind", "behind the defense", "behind the defensive line"],
         ["sweeper keeper", "sweeper-keeper", "sweeper"],
         ["pushing higher", "come out", "off the line", "higher up", "clearing through balls", "intercept"],
     ]},
]


def eval_dim1(answer: str, ground_truths: List[str]) -> float:
    """Keyword exact match (case-insensitive)"""
    ans_lower = answer.lower()
    return 1.0 if any(gt.lower() in ans_lower for gt in ground_truths) else 0.0


def eval_dim2(answer: str, regex: str) -> float:
    """Regex match. Also accepts dashless form (e.g. '4231' → '4-2-3-1')."""
    if re.search(regex, answer):
        return 1.0
    # Extract digit sequence and insert dash between every digit
    digits = re.findall(r'\d', answer)
    normalized = '-'.join(digits)
    return 1.0 if re.search(regex, normalized) else 0.0


def eval_dim3(answer: str, keyword_groups: List[List[str]]) -> float:
    """Keyword group rubric, normalized to 0.0-1.0"""
    ans_lower = answer.lower()
    score = sum(1 for group in keyword_groups if any(kw.lower() in ans_lower for kw in group))
    return score / len(keyword_groups) if keyword_groups else 0.0


def ask(tokenizer, model, question: str, max_new_tokens: int) -> str:
    """Generate answer using chat template (Gemini-recommended pattern for Qwen2.5)."""
    messages = [
        {"role": "system", "content": "You are a knowledgeable soccer expert. Answer concisely in plain text only."},
        {"role": "user", "content": question},
    ]
    # tokenize=False → str → re-tokenize avoids tiktoken BPE decode artifacts
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    # Trim input tokens, then batch_decode with clean_up_tokenization_spaces
    trimmed = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Soccer Understanding Benchmark")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens for generation")
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        model_slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
        output_path = f"results/soccer_understanding_{model_slug}.json"
    else:
        output_path = args.output
    
    # Create results directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print("Model loaded successfully")
    
    # Process benchmark
    details = []
    dim1_scores = []
    dim2_scores = []
    dim3_scores = []
    dim2_incorrect = []
    
    for item in BENCHMARK:
        print(f"Processing {item['id']}...", end=" ", flush=True)
        
        # Get model answer
        answer = ask(tokenizer, model, item["question"], args.max_new_tokens)
        
        # Score based on dimension
        dim = item["dim"]
        if dim == 1:
            score = eval_dim1(answer, item["ground_truths"])
            dim1_scores.append(score)
        elif dim == 2:
            score = eval_dim2(answer, item["regex"])
            dim2_scores.append(score)
            if score == 0.0:
                dim2_incorrect.append((item["id"], item["question"], answer))
        elif dim == 3:
            score = eval_dim3(answer, item["keyword_groups"])
            dim3_scores.append(score)
        
        details.append({
            "id": item["id"],
            "dim": dim,
            "question": item["question"],
            "answer": answer,
            "score": score
        })
        
        print(f"Score: {score:.2f}")
    
    # Calculate dimension averages
    avg_dim1 = sum(dim1_scores) / len(dim1_scores) if dim1_scores else 0.0
    avg_dim2 = sum(dim2_scores) / len(dim2_scores) if dim2_scores else 0.0
    avg_dim3 = sum(dim3_scores) / len(dim3_scores) if dim3_scores else 0.0
    
    # Calculate SUS score
    sus = avg_dim1 * 0.20 + avg_dim2 * 0.40 + avg_dim3 * 0.40
    
    # Prepare output
    output_data = {
        "model": args.model,
        "scores": {
            "dim1_accuracy": avg_dim1,
            "dim2_accuracy": avg_dim2,
            "dim3_normalized": avg_dim3,
            "sus": sus
        },
        "details": details
    }
    
    # Save JSON output
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"=== Soccer Understanding Benchmark: {args.model} ===")
    print("=" * 70)
    print(f"Dimension 1 (Rules)    Accuracy: {avg_dim1*100:5.1f}%  ({int(sum(dim1_scores))}/{len(dim1_scores)})")
    print(f"Dimension 2 (Spatial)  Accuracy: {avg_dim2*100:5.1f}%  ({int(sum(dim2_scores))}/{len(dim2_scores)})")
    print(f"Dimension 3 (Tactical) Score:    {avg_dim3*100:5.1f}%  (normalized)")
    print("-" * 70)
    print(f"Overall SUS Score:               {sus*100:5.1f}%")
    print(f"Results saved to: {output_path}")
    
    # Print incorrect Dimension 2 questions
    if dim2_incorrect:
        print("\n--- Incorrect Dimension 2 Answers (Spatial Hallucination Verification) ---")
        for q_id, question, answer in dim2_incorrect:
            print(f"\n[{q_id}] {question}")
            print(f"Model answer: {answer}")


if __name__ == "__main__":
    main()
