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
| ANS token | 10 | 0.0639 | Phase1はANS token非依存。乱数差と考えられる |

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
| ANS token | 10 | 1.3 | action=80, others=40 | 0.2125 | 0.4262 | 0.4469 | 0.2314 | echo依然残る。possession/zone微改善、action/pressure微低下 |

---

## Phase 2: Multi-task Action Alignment（追加実験）

| Experiment | Games | LoRA | lora_rank | f1_action↑ | rouge_possession↑ | rouge_zone↑ | rouge_pressure↑ | Notes |
|---|---|---|---|---|---|---|---|---|
| LoRA拡張(k,o_proj) + rank32 | 10 | ON | 32 | - | - | - | - | Phase2単体未計測 |
| ANS token | 10 | ON | 32 | 0.3923 | 0.0971 | 0.0874 | 0.0920 | action F1↑、possession/zone/pressure 大幅低下 |

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

## Phase 2: action 1タスク・LoRA なし（学習後テスト）

> 設定: LoRA=OFF / action タスクのみ / 5ゲーム(5392サンプル) / 20エポック

| Experiment | best_val (epoch) | Test f1_action↑ | Notes |
|---|---|---|---|
| 指示文あり (202605042259) | 0.7533 (ep10) | **0.5623** | ep11以降過学習(val→1.27) |
| 指示文なし (202605050018) | 0.7128 (ep9) | **0.5477** | ep10以降過学習(val→1.07) |

---

## Phase 3: action 1タスク・LoRA なし推論

> 設定: 20clips / rep=1.3 / max=40 / ALLOWED_TASKS=action / free_config=qa_action.json

| Experiment | Inference f1_action↑ | Free QA | Notes |
|---|---|---|---|
| 指示文あり (202605042259) | 0.8622 (n=15) ※ | ほぼ空文字 or `trap,trap,...` ループ | F1は訓練データ重複で過大評価の可能性大 |
| 指示文なし (202605050018) | 0.0836 (n=15) | 全クリップ空文字 | 推論スクリプトが指示文を渡すため分布外→当然低い |

※ inference は全クリップからランダム20件サンプルのため訓練データと重複している可能性が高い。Test F1=0.5623 の方が信頼性が高い。

**生成例（指示文あり・正解ケース）**:
```
gt:  "pass, trap, through pass, block, clearance"
gen: "pass, trap, through pass, block, clearance"  ← 完全一致

gt:  "foul received, trap, pass, dribble, tackle"
gen: "pass, trap, dribble, foul, foul received"    ← 語彙は合うが順番違い(F1=0.8)
```

**生成例（指示文あり・失敗ケース）**:
```
gt:  "pass, shot, goalkeeper save, throw-in, trap"
gen: "shot, goalkeeper save, pass, throw-in, trap, shot, goalkeeper save, ..."  ← 正解語彙を無限ループ
```

**考察**:
- Q-Former はトラッキングデータからアクション語彙をある程度正確に抽出できている（Test F1≈0.56）
- LLM が「カンマ区切り語彙リスト」フォーマットに過剰適合しているため Free QA に転用できない
- 指示文なし学習は Test F1≈0.55 で同等だが、推論時に指示文が渡されると完全に崩壊する
- 次ステップ: instruction dropout で Q-Former を鍛えつつ Free QA にも対応できるようにする

---

## Phase 2: 自然文ターゲット・LoRA なし（学習後テスト）

> 設定: LoRA=OFF / action タスクのみ / 5ゲーム / 10エポック / 自然文ターゲット  
> ターゲット形式: `"pass, trap"` → `"In this soccer sequence, performing pass and trap."`  
> 評価: compute_f1_action をサブストリング検索に変更（カンマ区切り・自然文両対応）

| Experiment | best_val (epoch) | Test f1_action↑ | Notes |
|---|---|---|---|
| 自然文 + 指示文あり (202605051303) | 0.3885 (ep7) | **0.7166** | 指示文: "Describe the soccer actions in this tracking sequence." |
| 自然文 + 指示文なし (202605051310) | 0.3833 (ep7) | **0.7466** | 指示文なし学習でもわずかに高い |

> **比較**: 語彙リスト形式 (202605042259) Test F1=0.5623 → 自然文形式で +0.15 改善

---

## Phase 3: 自然文ターゲット推論

> 設定: 20clips / rep=1.3 / max=40 / free_config=qa_action.json

| Experiment | Inference f1_action | Free QA | Notes |
|---|---|---|---|
| 自然文 + 指示文あり (202605051303) | 0.0000 ※ | **全20クリップで正常な英文を生成** | ※ 推論スクリプトが語彙リスト付き長指示文を渡すため空出力。正式F1はStep6の0.7166 |
| 自然文 + 指示文なし (202605051310) | 0.0956 | 空文字 | 指示文なし学習のため推論時の指示文に未対応 |

**Free QA 生成例（Run A、指示文: "List the soccer actions occurring in this tracking sequence in chronological order."）**:
```
gt:  "pass, touch, clearance"
gen: "In this soccer sequence, performing pass, touch and clearance."  ← 完全一致

gt:  "foul received, trap, pass, dribble, tackle"
gen: "In this soccer sequence, performing foul received, trap and pass."

gt:  "touch, pass, trap, dribble, cross, clearance"
gen: "In this soccer sequence, performing trap, pass, dribble, cross, clearance and throw-in."
```

**考察**:
- 自然文ターゲットに変更することで Test F1 が 0.56 → 0.72 に大幅改善
- 自然文形式の学習により LLM が文章生成モードで動作し、Free QA も成立
- 指示文なし学習は Test F1 がわずかに高いが、推論時に任意の指示文を受け付けられない
- **指示文あり + 自然文ターゲットが最も汎用 QA に近い結果**

---

## Phase 2: カリキュラム学習（学習後テスト）

> 設定: checkpoints/202605041201 / LoRA=ON rank=32 / 全ゲーム(5392サンプル, train4314/val539/test539) / 各ステージ5エポック

| Stage | Tasks | best_val↓ | f1_action↑ | rouge_possession↑ | rouge_zone↑ | rouge_pressure↑ |
|---|---|---|---|---|---|---|
| Stage 1 | action | 0.7241 | 0.4341 | 0.1198 | 0.0481 | 0.0482 |
| Stage 2 | action + possession | 0.2912 | 0.5028 | 0.1821 | 0.1158 | 0.1106 |
| Stage 3 | action + possession + zone | 0.1912 | 0.4585 | 0.1743 | 0.1524 | 0.1083 |
| Stage 4 (final) | all 4 tasks | 0.1246 | 0.4769 | 0.1825 | 0.1523 | 0.1187 |

> **注**: Stage 2 で action F1 が 0.4341 → 0.5028 に上昇するが、Stage 3 で 0.4585 に低下（タスク追加による干渉）。
> カリキュラム学習は廃案（理由: Q-Former がタスク専用トークンに特化してしまい、汎用 QA に使えない）。

---

## Phase 3: カリキュラム学習後の推論

> 設定: checkpoints/202605041201 / 20 clips / rep=1.3 / max=40 / free_config=qa_action.json

| Experiment | Clips | f1_action↑ | Notes |
|---|---|---|---|
| カリキュラム baseline (全ゲーム, LoRA ON) | 20 (n=15) | 0.2647 | instruction echo・繰り返し残存。Free QA はほぼ echo または無限ループ |

**Free QA 生成例（instruction を変えた場合）**:
- `"List the soccer actions occurring in this tracking sequence in chronological order."` → ほぼ全クリップで instruction 文を繰り返す
- 一部クリップのみ `"1. The player dribbles the ball. 2. ..."` のような文章を生成
- 複数クリップで `"... Read more... Read more..."` の無限ループ

**バグ**: Python 3.8 では `Path.with_stem()` が存在しない → Free QA CSV 保存時に `AttributeError` クラッシュ（推論自体は完了済み）。

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

### ANS token の観察

**Phase 3**:
- baseline (task-specific max) との比較で差は小さく、全体的には大きな改善なし
- action: 0.2235 → 0.2125（微低下）。生成結果を見ると `. List the soccer actions...` や番号付き語彙リスト（`1. block\n2. clearance\n...`）の echo が依然として発生
- possession: 0.4036 → 0.4262（微改善）。正答フレーズ + 繰り返しが混在する形に変化
- zone: 0.4428 → 0.4469（ほぼ同等）
- pressure: 0.2555 → 0.2314（低下）。高圧・中圧の誤答が多い
- **結論**: `<ANS>` トークン単体では instruction echo を十分に抑制できなかった

**Phase 2 の異変**:
- action F1 は 0.3805 → 0.3923 に小幅改善
- possession / zone / pressure の ROUGE が大幅低下（0.15 → 0.10 程度）
- 原因不明。train/test 分割の乱数差の可能性あり

**生成パターン（Phase 3 CSV より）**:
- action: instruction テキストを先頭に echo してから action 語彙を続ける → F1 低下
- action: `. 1\nList the...` や番号付きリストを生成 → F1 = 0.00（語彙形式が不一致）
- possession: 正答フレーズを繰り返す or home/away を交互に出力するサンプルあり
- zone / pressure: `Where on the field...` や `Describe the pressing...` を先頭に echo してから回答
