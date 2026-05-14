# LLaVA-style Ablation Results

- **データ**: SoccerNet-England-EPL 全シーズン（5ゲーム）
- **固定条件**: LoRA なし / SENTENCE_FORMAT=1 / INSTRUCTION_DIVERSE=1 / 10エポック
- **アブレーション軸**: init重み(Phase1 vs Phase1.5) × hub(Q-Former vs Linear) × Phase2.5(あり vs なし)

---

## 全実験サマリー

> Free QA 評価基準: ○=action・異ドメイン両方正確 / △A=action正確・異ドメイン沈黙 / △B=形式追従・内容幻覚 / ×=全件空文字またはエコー

| 実験タグ | init | hub | Phase2.5 | Phase2 F1↑ | Phase3 F1↑ | Free QA | 備考 |
|---|---|---|---|---|---|---|---|
| init1_div1_hubqformer | Phase1 | Q-Former | なし | - | - | - | |
| init1_div1_hubqformer | Phase1 | Q-Former | あり | - | - | - | |
| init1_div1_hublinear | Phase1 | Linear | なし | - | - | - | |
| init1_div1_hublinear | Phase1 | Linear | あり | - | - | - | |
| init15_div1_hubqformer | Phase1.5 | Q-Former | なし | - | - | - | |
| init15_div1_hubqformer | Phase1.5 | Q-Former | あり | - | - | - | |
| init15_div1_hublinear | Phase1.5 | Linear | なし | - | - | - | |
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
| init1_div1_hubqformer | なし | - | - |
| init1_div1_hubqformer | あり | - | - |
| init1_div1_hublinear | なし | - | - |
| init1_div1_hublinear | あり | - | - |
| init15_div1_hubqformer | なし | - | - |
| init15_div1_hubqformer | あり | - | - |
| init15_div1_hublinear | なし | - | - |
| init15_div1_hublinear | あり | - | - |

---

## Phase 4: Free QA スタイル汎化

| 実験タグ | Phase2.5 | formation | commentary | first_action |
|---|---|---|---|---|
| init1_div1_hubqformer | なし | - | - | - |
| init1_div1_hubqformer | あり | - | - | - |
| init1_div1_hublinear | なし | - | - | - |
| init1_div1_hublinear | あり | - | - | - |
| init15_div1_hubqformer | なし | - | - | - |
| init15_div1_hubqformer | あり | - | - | - |
| init15_div1_hublinear | なし | - | - | - |
| init15_div1_hublinear | あり | - | - | - |

---

## 補足・考察

<!-- 実験が進むにつれて追記 -->
