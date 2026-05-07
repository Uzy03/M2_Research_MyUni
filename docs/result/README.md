# Ablation Results

実験条件を変えながら各 Phase の評価指標を記録する。

- **データ**: SoccerNet-England-EPL-2014/2015 シーズン
- **評価指標**: Phase 1 = ADE（低いほど良い↓）、Phase 2/3 = F1 / ROUGE-L（高いほど良い↑）

---

## 全実験サマリー

> - **Phase 2 F1**: 学習後テストセットでの vocab F1（高いほど良い）
> - **Phase 3 F1（指示文変更）**: 推論時に学習外指示文（qa_action.json）を使用した vocab F1
> - **Free QA**: 多様な指示文（formation / commentary / first_action）に対する総合品質
>   - ○ = action・異ドメイン両方で正確な内容を生成（トラッキングデータを使用）
>   - △A = action クエリは正確・少なくとも1件意味のある応答あり、ただし異ドメイン指示には沈黙またはテンプレート固定
>   - △B = 指示の形式には反応（少なくとも1件意味のある応答あり）、内容は LLM 事前知識による幻覚
>   - × = **全件空文字** または **学習時定型文（"In this soccer sequence, performing X."）の繰り返しのみ**

| 実験 | LoRA | ep | Phase 2 F1↑ | Phase 3 F1↑ | Free QA | 備考 |
|---|---|---|---|---|---|---|
| 指示文なし + 自然文 (202605051310) | なし | 10 | 0.7466 | 0.0000 | × | 推論時に指示文が来ると全件空文字 |
| 指示文あり + 自然文 (202605051303) | なし | 10 | 0.7166 | 0.8371 | × | commentary=全件定型文・formation/first_action=ほぼ全件空文字 |
| LoRA rank=2 (202605061305) | 2 | 10 | 0.6207 | 0.0222 | × | ガーベジ・echo・完全崩壊 |
| LoRA rank=4 (202605060027) | 4 | 10 | 0.6894 | 0.1711 | △B | echo・幻覚（LLM 事前知識で回答） |
| LoRA rank=8 (202605061327) | 8 | 10 | 0.7212 | 0.1068 | × | mostly echo（Phase 4 未実行・Phase 3 での観察） |
| **No LoRA + 指示多様化 (202605061813)** | なし | 5 | **0.7345** | **0.8194** | **△A** | **Phase 3 最高 F1・action は正確だが formation 全件空文字** |
| No LoRA + 指示+回答多様化 (202605062134) | なし | 10 | 0.6615 | 0.7915 | × | commentary=全件テンプレート列挙・formation/first_action=空文字または同テンプレート |
| rank=4 + 指示多様化 (202605062135) | 4 | 5 | 0.6384 | 0.1923 | △B | formation/commentary 形式には反応・内容は全件幻覚 |
| rank=4 + 指示+回答多様化 (202605071626) | 4 | 10 | 0.5931 | 0.1438 | △B | 形式追従・content 幻覚（※異なるゲームデータ使用、10ep のみ） |

**結論**: 指示+回答多様化（202605062134）も Phase 4 では全件定型文／空文字で × に。**唯一 △A は No LoRA + 指示多様化（202605061813）のみ**（first_action に 1 件「The first action is trap」が確認）。LoRA あり／なしのトレードオフも、多様化を加えても解消されない → Token-Word Alignment（B-3）が本命。

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

> 設定: 20clips / rep=1.3 / max=40 / SENTENCE_FORMAT=1 / free_config=qa_action.json  
> 評価指示文（action タスク）: `"Describe the soccer actions in this tracking sequence."` (sentence_instruction)

| Experiment | Inference f1_action↑ | Free QA | Notes |
|---|---|---|---|
| 自然文 + 指示文あり (202605051303) | **0.8371** (n=15) | 全20クリップで正常な英文を生成 | sentence_instruction で正しく評価 |
| 自然文 + 指示文なし (202605051310) | 0.0000 (n=15) | 空文字 | 指示文なし学習のため推論時の指示文に未対応 |

**生成例（Run A）**:
```
gt:  "pass, touch, clearance"
gen: "In this soccer sequence, performing pass, touch and clearance."          F1=1.00

gt:  "pass, trap, dribble, clearance, touch"
gen: "In this soccer sequence, performing pass, trap, dribble, clearance and touch."  F1=1.00

gt:  "touch, pass, trap, dribble, cross, clearance"
gen: "In this soccer sequence, performing pass, trap, dribble, cross, clearance, touch and tackle."  F1=0.92

gt:  "pass, shot, goalkeeper save, throw-in, trap"
gen: "In this soccer sequence, performing shot, goalkeeper save and pass."     F1=0.75
```

**Free QA 生成例（Run A、指示文: "List the soccer actions occurring in this tracking sequence in chronological order." ← 学習時と異なる指示文）**:
```
gt:  "pass, touch, clearance"
gen: "In this soccer sequence, performing pass, touch and clearance."

gt:  "foul received, trap, pass, dribble, tackle"
gen: "In this soccer sequence, performing foul received, trap and pass."
```

**考察**:
- 自然文ターゲット + 指示文あり: Inference F1=0.8371 で高精度な sentence 形式出力を実現
- 学習時と異なる指示文（qa_action.json）でも正しい sentence 形式で出力 → 指示文への汎化が起きている
- 指示文なし学習は Test F1=0.7466 と高いが、推論時に指示文を渡すと崩壊する
- **正式な評価軸**: Test F1 (train.log Step6) + Free QA 目視確認。Phase 3 Inference F1 は参考値

---

## Phase 2: 自然文ターゲット・LoRA rank=4（学習後テスト）

> 設定: LoRA=ON rank=4 / action タスクのみ / 5ゲーム / 10エポック / 自然文ターゲット / 指示文あり

| Experiment | best_val (epoch) | Test f1_action↑ | Notes |
|---|---|---|---|
| 自然文 + LoRA rank=4 (202605060027) | 0.4007 (ep7) | **0.6894** | ep8以降 val 急上昇（0.52）→ 過学習 |

> **比較**: LoRA なし (202605051303) Test F1=0.7166 → LoRA rank=4 で -0.027 低下

---

## Phase 3: 自然文ターゲット・LoRA rank=4 推論

> 設定: 20clips / SENTENCE_FORMAT=1 / free_config=qa_action.json（学習指示文と異なる）  
> 学習指示文: `"Describe the soccer actions in this tracking sequence."` (sentence_instruction)  
> 推論指示文: `"List the soccer actions occurring in this tracking sequence in chronological order."` (qa_action.json)

| Experiment | Inference f1_action↑ | Free QA | Notes |
|---|---|---|---|
| 自然文 + LoRA rank=4 (202605060027) | 0.1711 (n=15) | 指示文 echo・番号付きリスト・幻覚生成 | LoRA なし (0.8371) から大幅低下 |

**Free QA 生成例（LoRA rank=4）**:
```
gen: "List the soccer actions occurring in this tracking sequence in chronological or..."  ← 指示文をそのまま echo
gen: "Here are the results. 1. The first action. 2. The second action. 3. The third a..."  ← プレースホルダー
gen: "1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11"                                                  ← 番号のみ
gen: "1. Kick-off 2. Pass 3. Tackle 4. Foul 5. Offside 6. Backpass 7."                   ← LLM 事前知識による幻覚
gen: "1. Pass (player passes the ball to a teammate) 2. Dribble (p..."                    ← LLM 事前知識による幻覚
```

**考察**:
- LoRA で LLM の応答能力が上がり「番号付きリスト」等の指示フォーマットには反応できるようになった
- しかし小データ（5ゲーム）では Q-Former 特徴量を使うより LLM の事前知識で答える方が loss が下がりやすく、トラッキング埋め込みを無視した幻覚生成が発生
- LoRA なしでは学習分布外の指示文に対して「沈黙」で対応したが、LoRA ありでは「echo」で対応 → 指示文への反応力は上がったが有害な方向に作用
- **結論**: LoRA rank=4 でも 5ゲームの小データでは過学習・幻覚生成が避けられない。LoRA なしの方が安定して高品質な出力を示す

---

## Phase 4: Zero-shot スタイル汎化テスト

> 設定: 20clips / SENTENCE_FORMAT=1 / free_config=configs/qa_commentary.json  
> 指示文: `"Describe the play in this soccer sequence in a lively commentary style."`

| Experiment | f1_action↑ | Free QA (commentary) | Notes |
|---|---|---|---|
| 指示文あり (202605051303) | **0.8371** (n=15) | 全20クリップで文章を生成（スタイルは学習フォーマット固定） | sentence_instruction で評価 |
| 指示文なし (202605051310) | - | 全20クリップ空文字 | 指示文なし学習のため推論指示文に未対応 |

**生成例（Run A、commentary 指示に対する出力）**:
```
instruction: "Describe the play in this soccer sequence in a lively commentary style."
gen: "In this soccer sequence, performing pass, touch and clearance."        ← 学習フォーマットのまま
gen: "In this soccer sequence, performing throw-in, trap and pass."          ← 実況風にならず
gen: "In this soccer sequence, performing shot, block, pass and trap."
gen: "In this soccer sequence, performing corner kick and shot."
```

**考察**:
- Run A は未知の指示文に対しても全クリップで文章を生成 → **指示文の有無に対する汎化**はできている
- ただし出力スタイルは学習時のフォーマット `"In this soccer sequence, performing..."` から変化しない → **スタイル指示への追従**は未達
- 原因: モデルは「アクションを sentence 形式で出力する」パターンに特化して学習しており、スタイルの変え方を学習していない

---

## Phase 4: 指示文内容理解テスト（フォーメーション質問）

> 設定: 202605051303/phase4 / SENTENCE_FORMAT=1 / free_config=configs/qa_formation.json  
> 指示文: `"What formation is the attacking team using in this soccer sequence? Answer with a formation like 4-4-2, 4-3-3, or 3-5-2."`

**生成結果**:
```
[15/20クリップ] → 空文字
[2/20クリップ] → "In this soccer sequence, formation and pass."       ← "formation" を action 語彙として誤認
[2/20クリップ] → "In this soccer sequence, which is the attacking team using in this soccer seque..."  ← 指示文が漏れ出し
[1/20クリップ] → "In this soccer sequence, which is the attacking team."
```

**結論: モデルは指示文の内容を理解していない**

- `"4-4-2"` や `"4-3-3"` のようなフォーメーション回答は**一件も生成されなかった**
- 学習した `"In this soccer sequence, performing X."` テンプレートから脱出できない
- 指示文の単語（"formation", "attacking team"）がアクション語彙として誤認されてテンプレートに埋め込まれるケースが発生
- **指示文トークンの役割**: 「何かを生成すべき」というトリガーとして機能しているが、内容の解釈には使われていない

---

## Phase 4: 指示文内容理解テスト②（最初のアクションのみ）

> 設定: 202605051303/phase4 / SENTENCE_FORMAT=1 / free_config=configs/qa_first_action.json  
> 指示文: `"What is the first action that occurs in this soccer sequence? Answer with a single action word only."`  
> ※ `--tasks none` により f1_action 評価なし（phase3 で計測済みのため不要）

**生成結果**:
```
[18/20クリップ] → 空文字
[1/20クリップ] → "In this soccer sequence, answering a shot."   ← "Answer" → "answering" に変化してテンプレートに混入
[1/20クリップ] → "In this soccer sequence, performing pass and trap."  ← 複数アクション（first only 指示を無視）
```

**単一アクション語だけを返したクリップはゼロ。**

**Phase 4 全実験の比較まとめ（LoRA なし）**:

| 指示文 | 非空出力 | 内容正解 | 考察 |
|---|---|---|---|
| commentary（同ドメイン・スタイル変更） | 20/20 | 0/20 | 訓練分布に近い → テンプレートをそのまま出力 |
| first action only（同ドメイン・数量制限） | 2/20 | 0/20 | "Answer" が動詞として漏れ出し |
| formation（異ドメイン） | 5/20 | 0/20 | "formation" が名詞として漏れ出し |

**総合結論（LoRA なし）: モデルは指示文の内容を理解していない**
- 指示文が訓練分布に近いほど出力が多く、遠いほど空文字になる
- LLM は指示文の意味ではなく**テンプレートパターンとの類似度**で出力するかを決めている
- Q-Former の出力はアクション抽出のみに特化しており、指示文内容への追従は不可能

---

## Phase 4: 指示文内容理解テスト・LoRA rank=4（フォーメーション質問）

> 設定: 202605060027/phase4 / SENTENCE_FORMAT=1 / free_config=configs/qa_formation.json

**生成結果**:
```
[2023102106_0937] "The attacking team is using a 4-3-3 formation. The attacking team is using a..."
[2023102002_0204] "The attacking team is using a 4-3-3 formation. In this sequence, the attacking..."
[2023102003_1044] "The attacking team is using the 4-3-3 formation. Answer: The attacking team is..."
[2023102003_0866] "The attacking team is using the 4-3-3 formation. Answer: 4-3..."
[2023102003_0181] "The attacking team is using a 4-4-2 formation, which means they are playing wit..."
[2023102106_0166] "The attacking team is using the formation 4-4-2."
← 残り14クリップは指示文 echo または "4-4-2, 4-3-3, or..." と例示をそのまま出力
```

**LoRA なし vs LoRA rank=4 比較（formation 質問）**:

| | LoRA なし (202605051303) | LoRA rank=4 (202605060027) |
|---|---|---|
| 非空出力 | 5/20 | **20/20** |
| formation 形式の回答 | 0/20 | **6/20以上** |
| 生成例 | `"In this soccer sequence, formation and pass."` | `"The attacking team is using a 4-3-3 formation."` |

**考察**:
- LoRA によって「指示文内容を読んで回答形式を変える」能力が明確に向上した
- フォーメーション形式（`"4-3-3"`, `"4-4-2"`）の回答が複数クリップで出現 → 指示文を読んでいる証拠
- ただし回答内容は LLM の事前知識による幻覚（トラッキングデータから導出していない）

---

## Phase 4: 指示文スタイル汎化テスト・LoRA rank=4（commentary）

> 設定: 202605060027/phase4 / SENTENCE_FORMAT=1 / free_config=configs/qa_commentary.json

**生成結果（抜粋）**:
```
"And we're off! The play is 'Soccer Sequence' and we're going to describe it in..."  ← commentary 形式
"AND WE'RE OFF..." ← commentary 形式
"And here comes the pass, straight to Johnson! He's got the ball and he's makin..."  ← commentary 形式（実況風）
"The sequence starts with a pass from the goalkeeper to a midfielder, who then p..."  ← narrative 形式
"This is a test. This is a test. This is a test..."                                   ← 繰り返しループ
"The sequence is: 1-2-3, 4-5-6, ..."                                                  ← 数列（崩壊）
```

---

## Phase 4: 指示文内容理解テスト・LoRA rank=4（最初のアクションのみ）

> 設定: 202605060027/phase4 / SENTENCE_FORMAT=1 / free_config=configs/qa_first_action.json

**生成結果（抜粋）**:
```
"Kick"
"Kickoff"
"The first action that occurs in this soccer sequence is: KICK."
"(e.g. "Kick") Kick"
"The first action that occurs in this soccer sequence is "Pass"."
← 残りは指示文 echo または "What is the second action..." と後続質問を自己生成
```

- **20/20 クリップで非空出力**（LoRA なしは 2/20）
- **単一アクション語での回答が多数**（LoRA なし は 0/20）→ 「単語1つで答えよ」という数量指示を理解
- ただし全クリップが `"Kick"` / `"Kickoff"` → LLM 事前知識（サッカー開始 = キックオフ）による幻覚。トラッキングデータ不使用

**LoRA なし vs LoRA rank=4 総合比較（Phase 4 全指示文）**:

| 指示文 | LoRA なし | LoRA rank=4 |
|---|---|---|
| commentary | 20/20・形式固定（training template） | 20/20・**commentary 形式を試みる** |
| formation | 5/20・形式誤り | 20/20・**formation 形式で回答** |
| first action only | 2/20・数量指示を無視 | 20/20・**単語1つで回答** |

**総合結論**:
- **LoRA なし**: トラッキングデータを正確に使う・指示文スタイル/内容を無視（アクション Test F1=0.7166）
- **LoRA rank=4**: 指示文を読んでスタイル・形式を変える・トラッキングデータを無視して LLM 事前知識で幻覚（アクション Test F1=0.6894）
- 両者のトレードオフが明確。「Q-Former 特徴量の活用」と「指示文追従」を同時に実現するには小データでは不十分であり、より多くのデータまたはアーキテクチャ上の工夫が必要

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

---

## Phase 2: LoRA rank アブレーション（自然文・5ゲーム）

> 設定: 自然文ターゲット / action タスクのみ / 5ゲーム(5392サンプル) / 10エポック / SENTENCE_FORMAT=1  
> LoRA rank=2: 202605061305 / LoRA rank=8: 202605061327

| Experiment | LoRA rank | best_val (epoch) | Test f1_action↑ | Notes |
|---|---|---|---|---|
| 自然文 + LoRA なし (202605051303) | — | 0.3885 (ep7) | **0.7166** | 比較ベースライン |
| 自然文 + LoRA rank=2 (202605061305) | 2 | 0.3845 (ep8) | **0.6207** | val は ep3以降不安定・上下 |
| 自然文 + LoRA rank=4 (202605060027) | 4 | 0.4007 (ep7) | **0.6894** | ep8以降 val 急上昇 |
| 自然文 + LoRA rank=8 (202605061327) | 8 | 0.3756 (ep7) | **0.7212** | 全実験中最高 F1 |

**LoRA rank 別 val 推移（rank=8 が最も安定）**:

| Epoch | rank=2 val | rank=4 val | rank=8 val |
|---|---|---|---|
| 1 | 0.4684 | 0.4975 | 0.4783 |
| 3 | 0.4394 | 0.4681 | 0.4488 |
| 5 | 0.4564 | 0.4643 | 0.4170 |
| 7 | 0.3942 ← best | 0.4007 ← best | **0.3756 ← best** |
| 9 | 0.4020 | (過学習) | 0.4437 |
| 10 | 0.4234 | — | 0.5052 |

---

## Phase 3: LoRA rank アブレーション推論（自然文・5ゲーム）

> 設定: 20clips / SENTENCE_FORMAT=1 / free_config=qa_action.json（学習指示文と異なる長い指示文）  
> ※ phase3 f1_action は SENTENCE_FORMAT=1 時は参考値（学習指示文と不一致のため低め）

| Experiment | LoRA rank | Inference f1_action↑ | Free QA 品質 |
|---|---|---|---|
| LoRA なし (202605051303) | — | 0.8371 | 全20クリップで自然文生成（スタイル固定） |
| LoRA rank=2 (202605061305) | 2 | 0.0222 | echo ほぼ全件・ガーベジ多数（`###`・`- - -`・無限ループ） |
| LoRA rank=4 (202605060027) | 4 | 0.1711 | echo / 番号付きリスト / 幻覚生成 |
| LoRA rank=8 (202605061327) | 8 | 0.1068 | mostly echo、2-3件で不完全な文章生成 |

**rank=2 Free QA 生成例**（代表的な失敗パターン）:
```
[2023102002_0204]  # # # # # # # # # # # # # # # #                ← ガーベジ
[2023102003_1044]  [List] [List] [List] [List] [List] ...          ← ループ
[2023102002_0839]  ................................                  ← ドット
[2023102002_0712]  In this order. In this order. In this order...  ← ループ
[2023102106_0630]  List the soccer actions... in chronological or  ← echo
```

**rank=8 Free QA 生成例**（rank=2 より若干改善、ただし mostly echo）:
```
[2023102002_0912]  1. The soccer player kicks the ball. 2. The soccer player runs...  ← 幻覚生成（LLM 事前知識）
[2023102002_0204]  The actions are: 1. Tracking 2. Sequencing 3. Actions...          ← 指示文語彙を分解
[2023102002_0217]  Soccer actions include dribbling, passing, shooting...             ← 幻覚生成
← 残り17件はほぼ echo
```

**LoRA rank アブレーション総合考察**:

| | Test F1 | Free QA | prefix 使用 |
|---|---|---|---|
| No LoRA | 0.7166 | テンプレート固定（安定） | **Yes（ただし指示文無視）** |
| rank=2 | 0.6207 | ガーベジ/echo（最悪） | ほぼなし |
| rank=4 | 0.6894 | 指示文 echo・幻覚 | なし（LLM 事前知識に依存） |
| rank=8 | **0.7212** | mostly echo・一部幻覚 | わずかに試みる |

- **rank=2**: パラメータ少なすぎ → LLM の指示文追従能力が引き出せず、かつ Q-Former 特徴量も使えない
- **rank=4**: パラメータ中程度 → 指示文に反応するが Q-Former 特徴量を無視して幻覚
- **rank=8**: パラメータ多め → **Test F1 がノー LoRA を上回る（0.7212）**、Free QA はまだ不安定
- **→ Instruction Diversification のベースには rank=8 が最有力**（最高 F1 かつわずかに生成能力あり）

---

## Phase 2: Instruction Diversification アブレーション（自然文・5ゲーム・5エポック）

> 設定: 自然文ターゲット / action タスクのみ / 5ゲーム / **5エポック** / SENTENCE_FORMAT=1 / INSTRUCTION_DIVERSE=1  
> 指示文バリエーション6種類からランダムサンプリング  
> Run A: LoRA なし (202605061813) / Run B: LoRA rank=8 (202605061807)

| Experiment | LoRA | best_val (epoch) | Test f1_action↑ | vs. 多様化なし比較 |
|---|---|---|---|---|
| No LoRA + 多様化なし (202605051303) | — | 0.3885 (ep7) | 0.7166 | ベースライン |
| **No LoRA + 多様化 (202605061813)** | — | 0.3801 (ep5) | **0.7345** | **+0.0179↑** |
| rank=8 + 多様化なし (202605061327) | 8 | 0.3756 (ep7) | 0.7212 | ベースライン |
| rank=8 + 多様化 (202605061807) | 8 | 0.3885 (ep5) | **0.5604** | **-0.1608↓（大幅劣化）** |

---

## Phase 3: Instruction Diversification アブレーション推論

> 設定: 20clips / SENTENCE_FORMAT=1 / free_config=qa_action.json（学習バリエーション外の指示文）

| Experiment | LoRA | Inference f1_action↑ | Free QA |
|---|---|---|---|
| No LoRA + 多様化なし (202605051303) | — | 0.8371 | テンプレート固定（全20件）|
| **No LoRA + 多様化 (202605061813)** | — | **0.8194** | **全20件で自然文生成・1件のみ番号リスト** |
| rank=8 + 多様化なし (202605061327) | 8 | 0.1068 | mostly echo |
| rank=8 + 多様化 (202605061807) | 8 | 0.0413 | 全件ガーベジ/ループ/断片 |

**Run A（No LoRA + 多様化）Free QA 生成例（学習外指示文に対する出力）**:
```
instruction: "List the soccer actions occurring in this tracking sequence in chronological order."
                ↑ 学習時に使っていない指示文

[2023102106_0937]  In this soccer sequence, performing pass, touch and clearance.
[2023102002_0912]  In this soccer sequence, performing throw-in and trap.
[2023102002_0204]  In this soccer sequence, performing pass, trap, block and through pass.
[2023102104_0198]  In this soccer sequence, performing trap, pass, block, clearance and throw-in.
[2023102003_1044]  In this soccer sequence, performing pass, trap, dribble, foul and foul received.
[2023102003_0181]  1. Pass 2. Trap 3. Cross 4. Clearance 5. Corner kick ... ← 1件のみ番号リスト
← 残り19件は全て "In this soccer sequence, performing..." 形式
```

**Run B（rank=8 + 多様化）Free QA 生成例（崩壊パターン）**:
```
[2023102106_0937]  in the. in the. in the. in the. ...          ← 無限ループ
[2023102002_0204]  of, for of, of. of, of. of. of. ...          ← 断片ループ
[2023102003_0181]  ofs and**s**s**s****s***...                  ← 完全崩壊
```

**Instruction Diversification アブレーション考察**:

| | Test F1 | Free QA 品質 | 原因 |
|---|---|---|---|
| No LoRA + 多様化 | **0.7345** | **全20件で自然文（最良）** | prefix 依存が維持され、多様な指示に汎化 |
| No LoRA + 多様化なし | 0.7166 | テンプレート固定（良好） | prefix 依存は維持、指示文は無視 |
| rank=8 + 多様化なし | 0.7212 | mostly echo | LoRA が指示文暗記に誘導 |
| rank=8 + 多様化 | 0.5604 | 全件崩壊 | LoRA + 多様化の同時学習が小データ5エポックで過負荷 |

---

## Phase 4: 指示文内容理解テスト・No LoRA + Instruction Diversification (202605061813)

> 設定: SENTENCE_FORMAT=1 / INSTRUCTION_DIVERSE=1 / LoRA なし

| 指示文 | 非空出力 | 内容正解 | 生成例 |
|---|---|---|---|
| formation | 0/20 | 0/20 | 全件空文字（沈黙） |
| commentary | 20/20 | 0/20 | `"In this soccer sequence, performing X."` テンプレート固定 |
| first action | 3/20 | 1/20 | 1件 `"The first action that occurs in this soccer sequence is trap."` ← 指示内容を理解！ |

**No LoRA + 多様化 vs rank=4（多様化なし）の Phase 4 比較**:

| 指示文 | No LoRA + 多様化なし | No LoRA + 多様化 | rank=4 多様化なし |
|---|---|---|---|
| formation | 5/20・テンプレート混入 | **0/20（悪化）** | **20/20・formation 形式** |
| commentary | 20/20・テンプレート固定 | 20/20・テンプレート固定 | 20/20・commentary 形式を試みる |
| first action | 2/20・テンプレート | 3/20・1件内容正解 | **20/20・単語1つ回答** |

**考察**:
- 指示文多様化は「多様な指示に対して何かを出力する」能力を向上させたが、「出力フォーマットを指示内容に合わせる」能力は向上しなかった
- 多様化した指示文がすべて「アクションを答えて」系のため、モデルはどの指示に対してもアクション記述で答えることを学習した
- フォーマット切替（formation → formation 形式）には LoRA による LLM 側の fine-tune が必要
- **first action で 1 件のみ内容正解**（"The first action that occurs in this soccer sequence is trap."）→ 微弱だが指示内容理解の萌芽
- **次ステップ**: rank=4 + 指示文多様化（形式切替 × prefix 活用の両立を狙う）

**総合結論**: **No LoRA + Instruction Diversification が最良構成**
- Test F1=0.7345（全実験最高）
- 学習時未使用の指示文（qa_action.json）に対しても全20件で正常な自然文生成
- これが「prefix を読んで指示に汎化する」という B-0 の目標を達成した最初の実験
- LoRA + 多様化の組み合わせは 5 エポック・小データでは過負荷。LoRA を使う場合はエポック数を増やすか、多様化後に LoRA を追加する2段階 fine-tune が必要

---

## Phase 2: 指示文+回答多様化 / rank=4+指示文多様化 アブレーション

> Run 1 (202605062134): No LoRA + INSTRUCTION_DIVERSE=1 + ANSWER_DIVERSE=1 / 10ep  
> Run 2 (202605062135): LoRA rank=4 + INSTRUCTION_DIVERSE=1 / 5ep

| Experiment | LoRA | ep | Test F1↑ | Phase3 Free QA |
|---|---|---|---|---|
| No LoRA + 指示多様化のみ (202605061813) | — | 5 | 0.7345 | 全件自然文（単一フォーマット） |
| **No LoRA + 指示+回答多様化 (202605062134)** | — | 10 | 0.6615 | **多様なフォーマット混在** |
| rank=4 + 指示多様化 (202605062135) | 4 | 5 | 0.6384 | mostly echo |

---

## Phase 4: 指示文+回答多様化 / rank=4+指示文多様化 比較

| 指示文 | No LoRA+指示多様化 (202605061813) | No LoRA+指示+回答多様化 (202605062134) | rank=4+指示多様化 (202605062135) |
|---|---|---|---|
| formation | 0/20・沈黙 | 6/20・アクション記述のみ | **15/20・"4-3-3"/"4-4-2" 形式** |
| commentary | 20/20・テンプレート固定 | 20/20・アクション記述のまま | **20/20・commentary 形式を試みる** |
| first action | 3/20・1件内容正解 | 2/20・アクション列挙 | **20/20・形式は理解（内容は Kickoff 幻覚）** |

**Run 1（No LoRA + 指示+回答多様化）Phase 3 生成例**（多様なフォーマット）:
```
"Throw-in, Trap and Pass were performed."          ← variant 2（大文字化）
"This sequence shows pass followed by trap..."     ← variant 3（シーケンス）
"Soccer actions: goalkeeper save flick-on."        ← variant 5（キーワード列挙）
"In this soccer sequence, performing pass..."      ← variant 0（既存）
```

**Run 2（rank=4 + 指示多様化）Phase 4 生成例**:
```
[formation]   "In this soccer sequence, the attacking team is using a 4-3-3 formation."
[formation]   "The attacking team is using a 4-4-2 formation."
[commentary]  "Oh, what a beautiful pass! The player has got the ball and is making a run..."
[commentary]  "And here's the midfield maestro, controlling the tempo of the game..."
[first_action] "The single action word is 'pass'."  ← 1件のみ prefix 使用の可能性
[first_action] "The first action that occurs is Kickoff."  ← 大半は幻覚
```

**考察**:
- **回答多様化の効果**: Phase 3 で出力フォーマットが多様化した（6バリアントが混在）→ ただし F1 が 0.7345→0.6615 に低下（モデルが「どのフォーマットで答えるべきか」の学習も必要になりコストが増加）
- **rank=4 + 指示多様化**: 5ep でも formation/commentary/first_action の形式理解が維持された。多様化なし rank=4 と同様の指示追従能力を持ちつつ、prefix を無視する問題は変わらず
- **根本的なトレードオフ**: No LoRA = prefix 使う・フォーマット切替不可 / LoRA = フォーマット切替できる・prefix 無視、は指示文・回答多様化を加えても解消されない
- **Token-Word Alignment（Phase B-3）がこのトレードオフを解消する本命**：Q-Former トークンを LLM 意味空間にアラインすることで、LoRA なしでも LLM が prefix を「意味ある情報」として読めるようにする
