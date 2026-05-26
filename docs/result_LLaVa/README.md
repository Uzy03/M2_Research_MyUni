# LLaVA-style Ablation Results

- **データ**: SoccerNet-England-EPL 全シーズン（5ゲーム）
- **固定条件**: LoRA なし / SENTENCE_FORMAT=1 / INSTRUCTION_DIVERSE=1 / 10エポック
- **アブレーション軸**: init重み(Phase1 vs Phase1.5) × hub(Q-Former vs Linear) × Phase2.5(あり vs なし) × Phase2.5.1(空間補助タスク追加)

---

## スコア一覧

> P2/P2.5 F1 は train-time test split（同分布）で評価。P3 F1 は学習外クリップでの推論。Judge スコアは 0-1 スケール（LLaMA-3-8B-Instruct による 0-100 点採点を正規化、n=20）。formation/def.line は spatial_labels.json（ルールベース K-Means）を Ground Truth として使用。

| init | hub | P2.5 | P3 run | P2 F1↑ | P2.5 F1↑ | P3 F1↑ | formation↑ | commentary↑ | att.intent↑ | def.intent↑ | def.line↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase1 | Q-Former | なし | 202605172348 | 0.7533 | - | 0.6691 | 0.0000 | 0.2375 | 0.0125 | 0.0125 | 0.0000 |
| Phase1 | Q-Former | あり | 202605180023 | 0.7533 | 0.7881 | 0.6604 | 0.0500 | 0.3375 | **0.7500** | **0.7500** | 0.1875 |
| Phase1 | Linear | なし | 202605180002 | 0.7019 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Phase1 | Linear | あり | 202605180039 | 0.7019 | **0.9351** | **0.6775** | 0.1000 | **0.6750** | 0.7125 | 0.6750 | 0.0000 |
| **LLM Baseline** | | | (metadata only) | - | - | - | **0.4250** | 0.7125 | **0.7500** | **0.7500** | **0.2750** |
| Phase1.5 | Q-Former | なし | 202605141535 | 0.6116 | - | 0.6004 | - | - | - | - | - |
| Phase1.5 | Q-Former | あり | - | 0.6116 | - | - | - | - | - | - | - |
| Phase1.5 | Linear | なし | 202605141549 | 0.6516 | - | 0.0000 | - | - | - | - | - |
| Phase1.5 | Linear | あり | - | 0.6516 | - | - | - | - | - | - |
| Phase1 | Q-Former | P2.5.1 | 202605202300 | 0.7533 | 0.8414 | **0.8615** | **0.1875** | **0.4625** | **0.7500** | **0.7500** | **0.2000** |
| Phase1 | Linear | P2.5.1 | 202605211053 | 0.7019 | 0.8842 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0375 | - |

---

## LLM-as-a-Judge サッカー理解度ベンチマーク（Soccer Understanding Benchmark）

> Judge モデルとして使用する LLM のサッカー理解度を定量評価。スクリプト: `tracking/eval_soccer_understanding.py`
> **SUS = D1×0.30 + D2×0.70**（客観ラベルのある次元のみ。D3 戦術推論はラベル不在のため補足参考値）

| モデル | プロンプト | D1 Rules↑ | D2 Spatial↑ | D3 Tactical（参考）↑ | **SUS↑** | 備考 |
|---|---|---|---|---|---|---|
| LLaMA-3-8B-Instruct | zero-shot | 25.0% | 60.0% | 58.3% | **49.5%** | **Judge 採用** |
| LLaMA-3-8B-Instruct | few-shot | 50.0% | 20.0% | 33.3% | 29.0% | D2 大幅悪化・不採用 |
| LLaMA-3.1-8B-Instruct | zero-shot | 25.0% | 40.0% | 33.3% | 35.5% | LLaMA-3 に劣るため不採用 |

### D1 個別結果（LLaMA-3-8B）
- Q1 バックパス間接FK: ✓（"Indirect free kick to the opposing team."）
- Q2 スローイン・オフサイド例外: ✗（誤ってオフサイドと判定）
- Q3 ペナルティエリアライン上のファウル: ✗（"indirect free kick" と回答、正解は penalty）
- Q4 ゴールキック→自陣ゴール: ✗（"indirect free kick" と回答、正解は corner kick）

### D2 個別結果（LLaMA-3-8B）
- Q1 4-1-4-1系: ✓（4-3-3と回答、許容範囲内）
- Q2 3-4-2-1系: ✗（3-2-2-3と回答）
- Q3 4-1-2-1-2: ✓（"442.21.2" → dashless変換で4-4-2にマッチ）
- Q4 4-2-3-1: ✗（"442(2)31" → dashless変換で不一致）
- Q5 3-1-4-2系: ✓（"352" → dashless変換で3-5-2にマッチ）

### D1 個別結果（LLaMA-3.1-8B）
- Q1 バックパス間接FK: ✓
- Q2 スローイン・オフサイド例外: ✗（誤ってオフサイドと判定）
- Q3 ペナルティエリアライン上のファウル: ✗（"indirect free kick" と回答）
- Q4 ゴールキック→自陣ゴール: ✗（"indirect free kick" と回答）

### D2 個別結果（LLaMA-3.1-8B）
- Q1 4-1-4-1系: ✓（4-3-3と回答）
- Q2 3-4-2-1系: ✗（3-5-2と回答）
- Q3 4-1-2-1-2: ✗（"4-1-2-2-2 or 4-1-2-2 (also known as 4-2-3-1)"と冗長回答→regex不一致）
- Q4 4-2-3-1: ✓
- Q5 3-1-4-2系: ✗（4-2-3-1と回答）

### 考察
- **LLaMA-3-8B が SUS=49.5% で LLaMA-3.1-8B（35.5%）を上回り、Judge として採用**
- LLaMA-3.1 は D2 で大きく下回る。原因: "4-1-2-2-2 or ..." のような冗長回答で regex マッチ失敗
- **D1 は両モデルとも 25%**。Q3（ファウルの位置）・Q4（ゴールキック→自陣ゴール）でともに誤り → ルール理解に共通の穴
- **D2 は LLaMA-3（60%）> LLaMA-3.1（40%）**。3バック系・ダイヤモンドの認識が苦手
- **D3（参考）も LLaMA-3（58.3%）> LLaMA-3.1（33.3%）**。LLaMA-3.1 は false-9 を "target man" と誤答
- commentary・intent タスクの Judge としては LLaMA-3-8B を使用可能。formation タスクには限界あり

---

## Free QA 品質

> ○=正確 / △=概ね正確だが形式に問題あり / △B=形式追従・内容幻覚 / ×=空文字・崩壊・定型文固定

| init | hub | P2.5 | P3 run | formation | commentary | first_action | 総合 |
|---|---|---|---|---|---|---|---|
| Phase1 | Q-Former | なし | 202605141356 | × | × (定型文) | × (定型文) | × |
| Phase1 | Q-Former | あり | 202605142333 | × | △B | △ | △B |
| Phase1 | Linear | なし | 202605141507 | × | × (崩壊) | × (崩壊) | × |
| Phase1 | Linear | あり | 202605142214 | △ | **○** | △ | **○** |
| Phase1.5 | Q-Former | なし | 202605141535 | × (空) | × (定型文) | × (空) | × |
| Phase1.5 | Q-Former | あり | - | - | - | - | - |
| Phase1.5 | Linear | なし | 202605141549 | × | × (崩壊) | × (崩壊) | × |
| Phase1.5 | Linear | あり | - | - | - | - | - |

---

## 考察

### Linear hub の崩壊と復活

- Phase2のみ（actionラベル学習）では hublinear の P3 F1=0.0000。出力は "and learn, and learn..." の繰り返し。
- 原因: mean pool で (B, T, 768) → (B, 768) に潰すと視覚情報がほぼ消え、LLM がハルシネーションループに入る。
- **Phase2.5 QA学習を加えると完全復活**（F1: 0.0000 → 0.6775）。QA多様データが繰り返し崩壊への正則化として機能した可能性。

### 以前の実験との比較（docs/result/README.md）

- 以前の全実験で commentary が○になったことは一度もない（LoRAなし→全件定型文、LoRA rank=4→全件幻覚）
- **今回 hublinear + Phase2.5（LoRAなし）が commentary 初の○を達成**
- Phase2.5 QA学習が「トラッキング → 自由文生成」のブレークスルーをもたらした

### hubqformer vs hublinear（Phase2.5あり）

- P3 F1: hublinear 0.6775 > hubqformer 0.6604（わずかに hublinear が上）
- Free QA: hublinear ○ vs hubqformer △B（hublinear が大きく上回る）
- hubqformer + Phase2.5 では commentary にアクションラベル形式が混入（"performing trap, pass, clearance..."）、formation で別タスク回答が混入するなどフォーマット退行が起きている
- **Phase2.5なしは全パターン×**: Q-Formerも含め、質問の種類に関係なくアクション定型文を出力するだけ。Phase2.5が Free QA 汎化の必須条件。
- **現時点のベスト構成: Phase1 + Linear + Phase2.5**

### Phase2.5.1（空間補助タスク追加）の効果

- **学習設定**: Phase2.5 重みから継続学習、allowed_tasks=action/formation/def_line + QA、5エポック
- **Q-Former**: P3 F1 が 0.6604 → **0.8615** に大幅改善。formation 0.05→**0.1875**、commentary 0.3375→**0.4625**、def.line 0.1875→**0.2000** と全タスクで改善
- **Linear**: 壊滅的忘却が発生。P3 F1=0.0000、FreeQA は全タスク崩壊（ランダムな文字列を生成）
- **結論**: Q-Former は空間タスク追加に頑健で有効。Linear は脆弱で QA 能力を喪失しやすい。以降の実験は Q-Former を採用する

### LLM Baseline との比較（連続スコア版、n=20）

- **attacking/defensive_intent**: Q-Former+P2.5 が baseline と同スコア（0.7500）、Linear+P2.5 も 0.6750-0.7125 でほぼ同等。トラッキング特徴から戦術意図の推論ができている
- **commentary**: Linear+P2.5 (0.6750) ≈ baseline (0.7125)。連続評価により以前のバイナリ（0.90）より厳しいスコアに。Q-Former+P2.5 は 0.3375 と大きく下回る
- **formation**: baseline が 0.4250 でトップ（partial credit で上昇）。両モデルとも 0.0500-0.1000 に留まり、選手の空間配置推論が苦手なことが数値で明確化
- **defensive_line**: baseline 0.2750 に対し Linear+P2.5 は 0.0000、Q-Former+P2.5 は 0.1875。formation 同様、空間的な絶対位置の推論が困難
- **Phase2.5なし**: 5タスク全て 0.00（Linear は commentary も崩壊）。Phase2.5 が Free QA の絶対条件
