# Ablation Results

実験条件を変えながら各 Phase の評価指標を記録する。

- **データ**: SoccerNet-England-EPL-2014/2015 シーズン
- **評価指標**: Phase 1 = ADE（低いほど良い↓）、Phase 2/3 = F1 / ROUGE-L（高いほど良い↑）

---

## Phase 1: Trajectory Regression

| Experiment | Games | ADE↓ | Notes |
|---|---|---|---|
| baseline | 5 | 0.0708 | |
| rep=1.3, max=40 | 5 | 0.0566 | |
| task-specific max | 10 | 0.0545 | |

---

## Phase 2: Multi-task Action Alignment（学習後テスト）

| Experiment | Games | LoRA | rep_penalty | max_new_tokens | f1_action↑ | rouge_possession↑ | rouge_zone↑ | rouge_pressure↑ | Notes |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 5 | ON | 1.0 | 128 | 0.4123 | 0.1514 | 0.1443 | 0.1166 | |
| rep=1.3, max=40 | 5 | ON | 1.3 | 40 | 0.4018 | 0.1451 | 0.1426 | 0.1176 | Phase2はほぼ同等 |
| task-specific max | 10 | ON | 1.3 | action=80, others=40 | 0.3805 | 0.1502 | 0.1470 | 0.1249 | |

---

## Phase 3: Zero-shot QA Inference

| Experiment | Games | rep_penalty | max_new_tokens | f1_action↑ | rouge_possession↑ | rouge_zone↑ | rouge_pressure↑ | Notes |
|---|---|---|---|---|---|---|---|---|
| baseline | 5 | 1.0 | 128 | 0.2771 | 0.1478 | 0.1665 | 0.1372 | instruction echo・繰り返し問題あり |
| rep=1.3, max=40 | 5 | 1.3 | 40 | 0.1341 | 0.4142 | 0.3222 | 0.3088 | possession/zone/pressure 大幅改善、action低下（語彙列挙がmax=40で不足） |
| task-specific max | 10 | 1.3 | action=80, others=40 | 0.2235 | 0.4036 | 0.4428 | 0.2555 | action回復、zone大幅改善 |

---

## Phase 2: Multi-task Action Alignment（追加実験）

| Experiment | Games | LoRA | lora_rank | f1_action↑ | rouge_possession↑ | rouge_zone↑ | rouge_pressure↑ | Notes |
|---|---|---|---|---|---|---|---|---|
| LoRA拡張(k,o_proj) + rank32 | 10 | ON | 32 | - | - | - | - | Phase2単体未計測 |

---

## Phase 3: Zero-shot QA Inference（追加実験）

| Experiment | Games | rep_penalty | max_new_tokens | f1_action↑ | rouge_possession↑ | rouge_zone↑ | rouge_pressure↑ | Notes |
|---|---|---|---|---|---|---|---|---|
| self-attention fusion (InstructionTrackingFusion, num_layers=2) | 10 | 1.3 | action=80, others=40 | ≈0.22 (偽) | 高い (偽) | 高い (偽) | 高い (偽) | **失敗**: 全タスクで instruction 全文を出力。スコアは偶発的一致による見かけ値 |

**失敗の詳細**（InstructionTrackingFusion: bidirectional TransformerEncoder で tracking と instruction を融合）:
- action: `" List the soccer actions... Use only: block, clearance, ..."` を繰り返す（f1≈0.22 だが正解ではない）
- possession / zone / pressure: 全サンプルで同一の instruction テキストを出力（完全固定出力）
- ROUGE が高く見える理由: instruction テンプレートが答えの単語（left/right/high 等）を含むため偶発的に一致

**根本原因**（instruction collapse）:
fused tracking tokens が instruction 情報を含むため、LLM は `[fused_tracking | BOS | instruction]` を受け取ると「instruction の次には instruction が来る」と学習してしまい、answer ではなく instruction 全文を生成する。self-attention による双方向融合が tracking と instruction を不可分に混合したことが原因。

**教訓**: tracking 特徴量と instruction を融合する場合、LLM 入力から instruction を除去するか、融合前後で情報が完全に分離されていることを保証する必要がある。

---

## 観察メモ

### baseline の生成問題

- **Instruction echo**: モデルが回答の後に instruction 文を繰り返す。例）`"The play is in the right side of the defensive third. Where on the field is the play occurring? Answer using the template..."`
- **Answer repetition**: 同じ回答文を何度も繰り返す。例）`"There is low pressure around the ball. / There is low pressure around the ball. / ..."`
- **原因**: `repetition_penalty=1.0`（ペナルティなし）かつ `max_new_tokens=128` が長すぎるため、短い正解を出力した後もトークンを生成し続ける。
- **改善案**: `repetition_penalty=1.3`、`max_new_tokens=40` に変更する。

### rep=1.3, max=40 の観察

- possession/zone/pressure の ROUGE-L が2〜3倍に改善（echo が消えてクリーンな回答を出力）
- action F1 が 0.28 → 0.13 に低下（max_new_tokens=40 では21語彙を列挙しきれない）
- **次の改善案**: action タスクのみ `max_new_tokens` を大きくする（タスク別設定）

### task-specific max（10試合）の観察

- action F1 が 0.13 → 0.22 に回復（max=80 の効果）
- zone ROUGE-L が 0.32 → 0.44 に大幅改善（データ量増加の効果）
- possession・pressure はやや低下（誤差範囲内）
- 全体として 5 試合より安定した結果
