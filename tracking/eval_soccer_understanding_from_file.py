#!/usr/bin/env python3
"""
Evaluate manually annotated answers from a JSON file using the same scoring logic
as eval_soccer_understanding.py. This is a standalone script that does not require
model loading or GPU resources.
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Any

# Add tracking directory to sys.path to import eval functions
sys.path.insert(0, os.path.dirname(__file__))
from eval_soccer_understanding import BENCHMARK, eval_dim1, eval_dim2, eval_dim3, clean_answer


def load_answers(answers_file: str) -> tuple[str, Dict[str, str]]:
    """
    Load answers from JSON file.
    
    Args:
        answers_file: Path to JSON file containing answers
        
    Returns:
        Tuple of (model_name, answers_dict) where answers_dict maps id -> answer
    """
    with open(answers_file, "r") as f:
        data = json.load(f)
    
    model = data.get("model", "unknown")
    answers_dict = {}
    
    for item in data.get("answers", []):
        question_id = item.get("id", "")
        answer = item.get("answer", "")
        if question_id:
            answers_dict[question_id] = answer
    
    return model, answers_dict


def get_output_path(model_name: str, output_arg: str = None) -> str:
    """
    Determine output path.
    
    Args:
        model_name: Model name from JSON
        output_arg: Optional output path from CLI argument
        
    Returns:
        Output path string
    """
    if output_arg:
        return output_arg
    
    # Create model slug: replace /, -, . with _
    model_slug = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"results/soccer_understanding_{model_slug}_manual.json"


def evaluate_answers(model_name: str, answers_dict: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluate answers and compute scores.
    
    Args:
        model_name: Model name from JSON
        answers_dict: Dictionary mapping question id to answer
        
    Returns:
        Dictionary with scores and details
    """
    details = []
    dim1_scores = []
    dim2_scores = []
    dim3_scores = []
    
    for item in BENCHMARK:
        question_id = item["id"]
        
        # Get answer from dictionary, default to empty string
        answer = answers_dict.get(question_id, "")
        
        # Clean the answer
        cleaned = clean_answer(answer)
        
        # Score based on dimension
        dim = item["dim"]
        if dim == 1:
            score = eval_dim1(cleaned, item["ground_truths"])
            dim1_scores.append(score)
        elif dim == 2:
            score = eval_dim2(cleaned, item["correct"])
            dim2_scores.append(score)
        elif dim == 3:
            score = eval_dim3(cleaned, item["keyword_groups"])
            dim3_scores.append(score)
        
        details.append({
            "id": question_id,
            "dim": dim,
            "question": item["question"],
            "answer": answer,
            "score": score
        })
    
    # Calculate dimension averages
    avg_dim1 = sum(dim1_scores) / len(dim1_scores) if dim1_scores else 0.0
    avg_dim2 = sum(dim2_scores) / len(dim2_scores) if dim2_scores else 0.0
    avg_dim3 = sum(dim3_scores) / len(dim3_scores) if dim3_scores else 0.0
    
    # Calculate SUS score
    sus = avg_dim1 * 0.30 + avg_dim2 * 0.70
    
    return {
        "model": model_name,
        "scores": {
            "dim1_accuracy": avg_dim1,
            "dim2_accuracy": avg_dim2,
            "dim3_normalized": avg_dim3,
            "sus": sus
        },
        "details": details,
        "dim1_scores": dim1_scores,
        "dim2_scores": dim2_scores,
        "dim3_scores": dim3_scores
    }


def print_results(results: Dict[str, Any]) -> None:
    """
    Print results in same format as eval_soccer_understanding.py.
    
    Args:
        results: Dictionary with scores and details
    """
    scores = results["scores"]
    dim1_scores = results["dim1_scores"]
    dim2_scores = results["dim2_scores"]
    dim3_scores = results["dim3_scores"]
    
    print("\n" + "=" * 70)
    print(f"=== Soccer Understanding Benchmark (Manual Answers): {results['model']} ===")
    print("=" * 70)
    print(f"Dimension 1 (Rules)    Accuracy: {scores['dim1_accuracy']*100:5.1f}%  ({int(sum(dim1_scores))}/{len(dim1_scores)})")
    print(f"Dimension 2 (Spatial)  Accuracy: {scores['dim2_accuracy']*100:5.1f}%  ({int(sum(dim2_scores))}/{len(dim2_scores)})")
    print(f"Dimension 3 (Tactical) Score:    {scores['dim3_normalized']*100:5.1f}%  (normalized)")
    print("-" * 70)
    print(f"Overall SUS Score:               {scores['sus']*100:5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate manually annotated answers on Soccer Understanding Benchmark"
    )
    parser.add_argument(
        "--answers_file",
        type=str,
        required=True,
        help="Path to JSON file containing manual answers"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/soccer_understanding_{model_slug}_manual.json)"
    )
    
    args = parser.parse_args()
    
    # Load answers
    print(f"Loading answers from: {args.answers_file}")
    model_name, answers_dict = load_answers(args.answers_file)
    print(f"Model: {model_name}")
    print(f"Questions loaded: {len(answers_dict)}")
    
    # Determine output path
    output_path = get_output_path(model_name, args.output)
    
    # Create results directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Evaluate answers
    print("Evaluating answers...")
    results = evaluate_answers(model_name, answers_dict)
    
    # Save JSON output
    output_data = {
        "model": results["model"],
        "scores": results["scores"],
        "details": results["details"]
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print results
    print_results(results)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
