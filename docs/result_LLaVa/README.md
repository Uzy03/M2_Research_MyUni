# LLaVA-style Ablation Results (v1)

- **データ**: SoccerNet-England-EPL 全シーズン（5ゲーム）
- **固定条件**: Phase2.5あり / LoRA なし / SENTENCE_FORMAT=1 / INSTRUCTION_DIVERSE=1 / Phase2=10エポック / Phase2.5=5エポック
- **アブレーション軸**: pool_mode（mean_pool vs player_tokens） × hub（Q-Former vs Linear）
- **アーキテクチャ (v1)**: エンコーダ処理順を修正（temporal per-player → N-pool）/ Linear hub を per-token 射影に修正 / Phase 1 から Q-Former を除去

---

## スコア一覧

> P2/P2.5 F1 は train-time test split（同分布）で評価。P3 F1 は学習外クリップでの推論。Judge スコアは 0-1 スケール（**GPT-4o** による 0-100 点採点を正規化、n=20）。

| pool_mode | hub | P3 run | P2 F1↑ | P2.5 F1↑ | P3 F1↑ | formation↑ | commentary↑ | att.intent↑ | def.intent↑ | def.line↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| mean_pool | Q-Former | 202605291101 | 0.5822 | 0.6533 | **0.7059** | **0.1375** | 0.2875 | 0.6250 | 0.1750 | **0.0875** |
| mean_pool | Linear | 202605291120 | 0.7036 | **0.7091** | 0.0444 | 0.0875 | **0.3625** | 0.6500 | 0.1375 | **0.0875** |
| player_tokens | Q-Former | 202605290954 | 0.6984 | 0.6398 | 0.6807 | 0.0625 | 0.3000 | **0.7625** | **0.4750** | 0.0000 |
| player_tokens | Linear | 202605291015 | **0.7133** | 0.6366 | 0.2167 | 0.0000 | 0.3250 | 0.5250 | 0.0625 | 0.0000 |
| **LLM Baseline** | — | (metadata only) | — | — | — | 0.5875 | 0.7625 | 0.9000 | 0.9125 | 0.5125 |

> **太字** = モデル側（LLM Baseline 除く）の列最高値

---

## 考察

### P3 F1（アクション分類）

- **mean_pool × Q-Former が最高（0.7059）**。encoder の時系列表現を Q-Former が効率よく cross-attend できている
- **Linear hub は mean_pool / player_tokens ともに P3 F1 が低い**（0.04, 0.22）。Phase 2.5 後も Action 分類精度の回復が旧実験より遅い傾向
- player_tokens × Q-Former は 0.6807 で Q-Former 系としては mean_pool と同水準

### GPT-4o Judge スコア

- **attacking intent**: player_tokens × Q-Former が 0.7625 で最高。N 個の選手トークンに cross-attend することで戦術意図の手がかりを保持できている
- **defensive intent**: 同じく player_tokens × Q-Former が 0.4750 と突出。mean_pool 系は 0.14-0.18 止まり
- **commentary**: mean_pool × Linear が 0.3625 で最高。player_tokens 系は 0.30-0.33 で大差なし
- **formation / def.line**: 全パターンで 0.00-0.14 に留まり LLM Baseline（0.59 / 0.51）と大差。空間的絶対位置の推論は現アーキテクチャの共通課題

### player_tokens の効果

- att.intent / def.intent において mean_pool に対して明確な優位（+0.14〜+0.30）
- P3 F1 は mean_pool × Q-Former とほぼ同等（0.7059 vs 0.6807）
- → **戦術意図タスクには player_tokens × Q-Former が有効**。formation/def.line は依然として課題

### Linear hub の崩壊

- mean_pool × Linear: P3 F1=0.0444（推論時に出力崩壊）
- player_tokens × Linear: P3 F1=0.2167（部分回復するも低水準）
- Q-Former は両 pool_mode で安定。Linear は per-token 射影に修正後も Phase 2.5 だけでは汎化が不十分

### LLM Baseline との比較

- att.intent / def.intent で player_tokens × Q-Former が 0.76 / 0.48 まで迫る（Baseline=0.90 / 0.91）
- commentary は最高でも 0.36（Baseline=0.76）と差が大きい
- formation / def.line は全モデルで Baseline に遠く及ばず → TrajPrism 方式の QA 改良 or 4択タスク化が必要

---

## LLM-as-a-Judge サッカー理解度ベンチマーク（Soccer Understanding Benchmark）

> Judge モデルとして使用する LLM のサッカー理解度を定量評価。スクリプト: `tracking/eval_soccer_understanding.py`
> **SUS = D1×0.30 + D2×0.70**（客観ラベルのある次元のみ。D3 戦術推論はラベル不在のため補足参考値）

> D2 評価方式: **MCQ（4択）** に変更済み（v2）。旧 regex 方式は false positive が発生したため廃止。
> GPT-4o は ChatGPT ブラウザ経由で手動実施（`tracking/eval_soccer_understanding_from_file.py` で採点）。

| モデル | プロンプト | D1 Rules↑ | D2 Spatial (MCQ)↑ | D3 Tactical（参考）↑ | **SUS↑** | 備考 |
|---|---|---|---|---|---|---|
| **GPT-4o** | manual (zero-shot) | **100%** | **100%** | 41.7%※ | **100%** | **Judge 採用** |
| LLaMA-3-8B-Instruct | zero-shot | 25.0% | 60.0% | 58.3% | 49.5% | Judge 不採用 |
| LLaMA-3.1-8B-Instruct | zero-shot | 25.0% | 60.0% | 33.3% | 49.5% | Judge 不採用 |
| LLaMA-3-8B-Instruct | few-shot | 50.0% | 40.0% | 33.3% | 43.0% | D2 悪化・不採用 |

※ GPT-4o の D3 が低いのは D3Q4 の回答が空文字だったため（JSON 生成が途切れた可能性）。

**結論**: 8B クラス LLM は SUS≈50% でランダムに近く Judge として不十分。GPT-4o（SUS=100%）を Judge に採用することを定量的に正当化できた。
