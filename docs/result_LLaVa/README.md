# LLaVA-style Ablation Results

- **データ**: SoccerNet-England-EPL 全シーズン（5ゲーム）
- **固定条件**: LoRA なし / SENTENCE_FORMAT=1 / INSTRUCTION_DIVERSE=1 / 10エポック
- **アブレーション軸**: init重み(Phase1 vs Phase1.5) × hub(Q-Former vs Linear) × Phase2.5(あり vs なし)

---

## スコア一覧

> P2/P2.5 F1 は train-time test split（同分布）で評価。P3 F1 は学習外クリップでの推論。P2.5 ROUGE-L/BLEU は QA テスト分割の生成評価。

| init | hub | P2.5 | P2 ckpt | P3 run | P2 F1↑ | P2.5 F1↑ | P2.5 ROUGE-L↑ | P2.5 BLEU↑ | P3 F1↑ |
|---|---|---|---|---|---|---|---|---|---|
| Phase1 | Q-Former | なし | phase2_init1_div1_hubqformer | 202605141356 | 0.7533 | - | - | - | 0.6691 |
| Phase1 | Q-Former | あり | phase2_5_init1_div1_hubqformer | 202605142333 | 0.7533 | 0.7881 | 0.3042 | 0.1165 | 0.6604 |
| Phase1 | Linear | なし | phase2_init1_div1_hublinear | 202605141507 | 0.7019 | - | - | - | 0.0000 |
| Phase1 | Linear | あり | phase2_5_init1_div1_hublinear | 202605142214 | 0.7019 | **0.9351** | 0.3083 | 0.1187 | **0.6775** |
| Phase1.5 | Q-Former | なし | phase2_init15_div1_hubqformer | 202605141535 | 0.6116 | - | - | - | 0.6004 |
| Phase1.5 | Q-Former | あり | phase2_5_init15_div1_hubqformer | - | 0.6116 | - | - | - | - |
| Phase1.5 | Linear | なし | phase2_init15_div1_hublinear | 202605141549 | 0.6516 | - | - | - | 0.0000 |
| Phase1.5 | Linear | あり | phase2_5_init15_div1_hublinear | - | 0.6516 | - | - | - | - |

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
