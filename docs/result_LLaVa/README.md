# LLaVA-style Ablation Results

- **データ**: SoccerNet-England-EPL 全シーズン（5ゲーム）
- **固定条件**: LoRA なし / SENTENCE_FORMAT=1 / INSTRUCTION_DIVERSE=1 / 10エポック
- **アブレーション軸**: init重み(Phase1 vs Phase1.5) × hub(Q-Former vs Linear) × Phase2.5(あり vs なし)

---

## スコア一覧

> P2/P2.5 F1 は train-time test split（同分布）で評価。P3 F1 は学習外クリップでの推論。Judge スコアは 0-1 スケール（LLaMA-3-8B-Instruct による 0-100 点採点を正規化、n=20）。formation/def.line は spatial_labels.json（ルールベース K-Means）を Ground Truth として使用。

| init | hub | P2.5 | P3 run | P2 F1↑ | P2.5 F1↑ | P3 F1↑ | formation↑ | commentary↑ | att.intent↑ | def.intent↑ | def.line↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase1 | Q-Former | なし | 202605172348 | 0.7533 | - | 0.6691 | 0.00 | 0.24 | 0.01 | 0.01 | 0.00 |
| Phase1 | Q-Former | あり | 202605180023 | 0.7533 | 0.7881 | 0.6604 | 0.05 | 0.34 | **0.75** | **0.75** | 0.19 |
| Phase1 | Linear | なし | 202605180002 | 0.7019 | - | 0.0000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Phase1 | Linear | あり | 202605180039 | 0.7019 | **0.9351** | **0.6775** | 0.10 | **0.68** | 0.71 | 0.68 | 0.00 |
| **LLM Baseline** | | | (metadata only) | - | - | - | **0.43** | 0.71 | **0.75** | **0.75** | **0.28** |
| Phase1.5 | Q-Former | なし | 202605141535 | 0.6116 | - | 0.6004 | - | - | - | - | - |
| Phase1.5 | Q-Former | あり | - | 0.6116 | - | - | - | - | - | - | - |
| Phase1.5 | Linear | なし | 202605141549 | 0.6516 | - | 0.0000 | - | - | - | - | - |
| Phase1.5 | Linear | あり | - | 0.6516 | - | - | - | - | - | - | - |

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

### LLM Baseline との比較（連続スコア版、n=20）

- **attacking/defensive_intent**: Q-Former+P2.5 が baseline と同スコア（0.75）、Linear+P2.5 も 0.68-0.71 でほぼ同等。トラッキング特徴から戦術意図の推論ができている
- **commentary**: Linear+P2.5 (0.68) ≈ baseline (0.71）。連続評価により以前のバイナリ（0.90）より厳しいスコアに。Q-Former+P2.5 は 0.34 と大きく下回る
- **formation**: baseline が 0.43 でトップ（partial credit で上昇）。両モデルとも 0.05-0.10 に留まり、選手の空間配置推論が苦手なことが数値で明確化
- **defensive_line**: baseline 0.28 に対し Linear+P2.5 は 0.00、Q-Former+P2.5 は 0.19。formation 同様、空間的な絶対位置の推論が困難
- **Phase2.5なし**: 5タスク全て 0.00（Linear は commentary も崩壊）。Phase2.5 が Free QA の絶対条件
