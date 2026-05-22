#!/usr/bin/env python3
"""Lightweight unit tests for Phase 2 — no GPU or data files required."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracking.dataset.multitask_dataset import TASKS, ACTION_VOCAB

# compute_f1_action は rouge_score 不要なので直接定義（train_action_alignment と同一ロジック）
_ACTION_VOCAB_SET = set(ACTION_VOCAB)

def compute_f1_action(pred, gt):
    if not gt.strip():
        return None
    gt_labels   = {w.strip() for w in gt.split(',')   if w.strip() in _ACTION_VOCAB_SET}
    pred_labels = {w.strip() for w in pred.split(',') if w.strip() in _ACTION_VOCAB_SET}
    if not gt_labels:
        return None
    if not pred_labels:
        return 0.0
    tp = len(pred_labels & gt_labels)
    p  = tp / len(pred_labels)
    r  = tp / len(gt_labels)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

try:
    from rouge_score import rouge_scorer as rs
    _rouge = rs.RougeScorer(['rougeL'], use_stemmer=True)
    def compute_rouge_l(pred, gt):
        if not gt.strip():
            return None
        return _rouge.score(gt, pred)['rougeL'].fmeasure
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False
    compute_rouge_l = None

_PASS = "\033[32m  PASS\033[0m"
_FAIL = "\033[31m  FAIL\033[0m"


def test_f1_perfect_match():
    assert compute_f1_action("pass, dribble", "pass, dribble") == 1.0

def test_f1_partial_match():
    score = compute_f1_action("pass", "pass, dribble")
    assert 0 < score < 1, f"expected 0 < score < 1, got {score}"

def test_f1_no_vocab_in_pred():
    assert compute_f1_action("some random text", "pass, dribble") == 0.0

def test_f1_empty_gt():
    assert compute_f1_action("pass", "") is None

def test_f1_order_invariant():
    a = compute_f1_action("pass, dribble", "dribble, pass")
    b = compute_f1_action("dribble, pass", "pass, dribble")
    assert a == b == 1.0, f"expected 1.0, got a={a} b={b}"

def test_rouge_l_exact():
    if not _HAS_ROUGE:
        return  # skip if rouge_score not installed
    gt = "The home team has ball possession in this sequence."
    assert compute_rouge_l(gt, gt) == 1.0

def test_rouge_l_empty_gt():
    if not _HAS_ROUGE:
        return
    assert compute_rouge_l("something", "") is None

def test_rouge_l_partial():
    if not _HAS_ROUGE:
        return
    gt = "The home team has ball possession in this sequence."
    pred = "The away team has ball possession in this sequence."
    score = compute_rouge_l(pred, gt)
    assert 0 < score < 1, f"expected partial score, got {score}"

def test_tasks_have_instructions():
    for task in TASKS:
        assert task['instruction'].strip(), f"Empty instruction: {task['name']}"

def test_action_instruction_contains_vocab():
    instr = TASKS[0]['instruction']
    for kw in ['pass', 'shot', 'dribble', 'tackle']:
        assert kw in instr, f"'{kw}' not found in action instruction"

def test_possession_instruction_contains_choices():
    instr = TASKS[1]['instruction']
    assert 'home' in instr and 'away' in instr

def test_zone_instruction_contains_template():
    instr = TASKS[2]['instruction']
    assert 'third' in instr and 'side' in instr

def test_pressure_instruction_contains_choices():
    instr = TASKS[3]['instruction']
    assert 'high' in instr and 'medium' in instr and 'low' in instr

def test_action_vocab_not_empty():
    assert len(ACTION_VOCAB) > 0

def test_all_tasks_have_label_field():
    for task in TASKS:
        assert 'label_field' in task, f"Missing label_field: {task['name']}"


_TESTS = [
    test_f1_perfect_match, test_f1_partial_match, test_f1_no_vocab_in_pred,
    test_f1_empty_gt, test_f1_order_invariant,
    test_rouge_l_exact, test_rouge_l_empty_gt, test_rouge_l_partial,
    test_tasks_have_instructions, test_action_instruction_contains_vocab,
    test_possession_instruction_contains_choices, test_zone_instruction_contains_template,
    test_pressure_instruction_contains_choices,
    test_action_vocab_not_empty, test_all_tasks_have_label_field,
]

if __name__ == '__main__':
    passed = 0
    for t in _TESTS:
        try:
            t()
            print(f"{_PASS}  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"{_FAIL}  {t.__name__}: {e}")
    print(f"\n{passed}/{len(_TESTS)} tests passed")
    sys.exit(0 if passed == len(_TESTS) else 1)
