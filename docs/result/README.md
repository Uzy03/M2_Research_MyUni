# Ablation Results

実験条件を変えながら各 Phase の評価指標を記録する。

- **データ**: SoccerNet-England-EPL-2014/2015 シーズン
- **評価指標**: Phase 1 = ADE（低いほど良い↓）、Phase 2/3 = F1 / ROUGE-L（高いほど良い↑）

---

## Phase 1: Trajectory Regression

| Experiment | Games | ADE↓ | Notes |
|---|---|---|---|
| baseline | 5 | 0.0708 | |

---

## Phase 2: Multi-task Action Alignment（学習後テスト）

| Experiment | Games | LoRA | rep_penalty | max_new_tokens | f1_action↑ | rouge_possession↑ | rouge_zone↑ | rouge_pressure↑ | Notes |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 5 | ON | 1.0 | 128 | 0.4123 | 0.1514 | 0.1443 | 0.1166 | |

---

## Phase 3: Zero-shot QA Inference

| Experiment | Games | rep_penalty | max_new_tokens | f1_action↑ | rouge_possession↑ | rouge_zone↑ | rouge_pressure↑ | Notes |
|---|---|---|---|---|---|---|---|---|
| baseline | 5 | 1.0 | 128 | 0.2771 | 0.1478 | 0.1665 | 0.1372 | instruction echo・繰り返し問題あり |

---

## 観察メモ

### baseline の生成問題

- **Instruction echo**: モデルが回答の後に instruction 文を繰り返す。例）`"The play is in the right side of the defensive third. Where on the field is the play occurring? Answer using the template..."`
- **Answer repetition**: 同じ回答文を何度も繰り返す。例）`"There is low pressure around the ball. / There is low pressure around the ball. / ..."`
- **原因**: `repetition_penalty=1.0`（ペナルティなし）かつ `max_new_tokens=128` が長すぎるため、短い正解を出力した後もトークンを生成し続ける。
- **改善案**: `repetition_penalty=1.3`、`max_new_tokens=40` に変更する。
