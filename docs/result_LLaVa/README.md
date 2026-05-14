# LLaVA-style Ablation Results

- **データ**: SoccerNet-England-EPL 全シーズン（5ゲーム）
- **固定条件**: LoRA なし / SENTENCE_FORMAT=1 / INSTRUCTION_DIVERSE=1 / 10エポック
- **アブレーション軸**: init重み(Phase1 vs Phase1.5) × hub(Q-Former vs Linear) × Phase2.5(あり vs なし)

---

## 全実験サマリー

> Free QA 評価基準: ○=action・異ドメイン両方正確 / △A=action正確・異ドメイン沈黙 / △B=形式追従・内容幻覚 / ×=全件空文字またはエコー

| 実験タグ | init | hub | Phase2.5 | Phase2 F1↑ | Phase3 F1↑ | Free QA | 備考 |
|---|---|---|---|---|---|---|---|
| init1_div1_hubqformer | Phase1 | Q-Former | なし | - | **0.6691** | △A | |
| init1_div1_hubqformer | Phase1 | Q-Former | あり | - | - | - | |
| init1_div1_hublinear | Phase1 | Linear | なし | - | 0.0000 | × | 繰り返し崩壊 |
| init1_div1_hublinear | Phase1 | Linear | あり | - | - | - | |
| init15_div1_hubqformer | Phase1.5 | Q-Former | なし | - | 0.6004 | × | Free QA 空文字 |
| init15_div1_hubqformer | Phase1.5 | Q-Former | あり | - | - | - | |
| init15_div1_hublinear | Phase1.5 | Linear | なし | - | 0.0000 | - | 繰り返し崩壊 |
| init15_div1_hublinear | Phase1.5 | Linear | あり | - | - | - | |

---

## Phase 2: 学習後テスト

| 実験タグ | best_val (epoch) | Test f1_action↑ |
|---|---|---|
| init1_div1_hubqformer | - | - |
| init1_div1_hublinear | - | - |
| init15_div1_hubqformer | - | - |
| init15_div1_hublinear | - | - |

---

## Phase 3: 推論（学習外指示文）

| 実験タグ | Phase2.5 | Inference f1_action↑ | Free QA (qa_action.json) |
|---|---|---|---|
| init1_div1_hubqformer | なし | **0.6691** (47.1s, 2.35s/clip) | - |
| init1_div1_hubqformer | あり | - | - |
| init1_div1_hublinear | なし | 0.0000 (134.8s, 6.74s/clip) | - |
| init1_div1_hublinear | あり | - | - |
| init15_div1_hubqformer | なし | 0.6004 (35.8s, 1.79s/clip) | - |
| init15_div1_hubqformer | あり | - | - |
| init15_div1_hublinear | なし | 0.0000 (127.3s, 6.36s/clip) | - |
| init15_div1_hublinear | あり | - | - |

---

## Phase 4: Free QA スタイル汎化

| 実験タグ | Phase2.5 | formation | commentary | first_action |
|---|---|---|---|---|
| init1_div1_hubqformer | なし | - | - | ○ (action文生成OK) |
| init1_div1_hubqformer | あり | - | - | - |
| init1_div1_hublinear | なし | - | - | × (繰り返し崩壊) |
| init1_div1_hublinear | あり | - | - | - |
| init15_div1_hubqformer | なし | - | - | × (空文字列) |
| init15_div1_hubqformer | あり | - | - | - |
| init15_div1_hublinear | なし | - | - | - (未確認) |
| init15_div1_hublinear | あり | - | - | - |

---

## 補足・考察

### 2026-05-14: Linear hub 完全崩壊

- **hublinear** は Phase 3 F1=0.0000。出力は "and learn, and learn, and learn..." の繰り返しループ。
- 原因: TrackingEncoder 出力 (B, T, 768) を mean(dim=1) で (B, 768) に潰すと、視覚情報がほぼ消える。LLM が visual token から手がかりを得られず hallucination ループに入る。
- **hublinear も学習時間が長い** (6.74s/clip vs Q-Former 2.35s/clip) のは、max_new_tokens まで繰り返しを生成しているため。
- **hubqformer** は正常。init1 > init15 (0.6691 vs 0.6004)。
- Phase 1.5 init (init15_hubqformer) は Phase 3 F1 は出るが **Phase 4 Free QA が空文字列**。Phase 2 の action 学習に過適合している可能性。
- **結論**: Linear hub は採用不可。Q-Former が必須。LLaVA の Linear projection が機能するのは画像の密な特徴量があるからであり、トラッキング座標のような疎な時系列には適さない。
