# Research Requirements & Roadmap

## 研究の最終ゴール

**「トラッキングデータの暗黙的な戦術パターンを Q-Former が学習し、LLM が人間定義の統計量では説明できない戦術的文脈を解説できる」**

### 分類器 + GPT-4 との差別化

**統計量アプローチ（情報の要約）**:
```
トラッキングデータ → [人間が定義した数式] → テキスト → GPT-4
```
- プレス強度・ゾーン・速度などは計算できる
- しかし「22人の幾何学的パターン」「スペースが生まれる予兆」「守備を引きつける無駄走り」は**定義の爆発**が起きる。人間が数式で列挙しきれない
- 新しい戦術が出るたびに計算式を書き直す必要がある

**Tracking-LLM（情報のロスレス圧縮）**:
```
トラッキングデータ → [Q-Former の潜在空間] → LLM
```
- Q-Former の 32 トークンは人間が定義できない暗黙的パターンをデータから自動学習
- 統計量では表しきれない非線形な位置関係・タイミング・予兆を LLM に直接渡す
- 新しい競技・戦術にもデータさえあれば適応可能

**キラーフレーズ**: 「統計量を GPT-4 に入れるのはデータの『要約』を読ませているだけ。本手法は LLM にデータの『現場』を直接見せている。」

### 注意：ゾーン・方向・プレッシャーの位置付け
これらは差別化の本体ではなく、**Alignment のための基礎**（Q-Former が座標を理解していることを示す手段）。本当の差別化は「人間が定義できない暗黙的な戦術パターンを Q-Former が学習できるか」であり、LLM-as-a-Judge で「統計量 + GPT-4 では出せない記述が現れるか」で実証する。

### 新たな差別化軸：Token-Word Alignment（対照学習）
「ゾーンやプレスは統計量で出せる→GPT-4 に渡せばよい」という反論に対する回答:
- 統計量は離散化した時点で**速度・角度・空間の連続的ニュアンス**を捨てている
- 分類器は "pass" と分類した瞬間に「その pass がどれだけ素早く・どれほど危険なエリアへ」という連続情報を消失させる
- 本手法では Q-Former トークンを LLM の意味空間に**対照学習でアライン**することで、連続的な物理情報を LLM の推論回路に直接入力できる
- これが「**定義の爆発を回避しながら情報のロスレス圧縮を実現する**」差別化の核心（Phase B-3）

---

## Phase A：現アーキテクチャの限界把握（〜2週間）

### A-1. LoRA rank アブレーション
**目的**: アクション検出と指示文追従のトレードオフがどの rank で最適化されるか確認。

**アプローチ**:
- rank=2, 8 を 5ゲーム・sentence format で学習（rank=4 比較済み）
- 評価: Phase 2 Test F1 ＋ Phase 4 での formation / commentary / first_action 出力品質
- 実行: `make run_from_phase2 MAX_GAMES=5 EPOCHS_PHASE2=20 SENTENCE_FORMAT=1 OPEN_LORA=1 LORA_RANK=2 GPU=X`
- 実行: `make run_from_phase2 MAX_GAMES=5 EPOCHS_PHASE2=20 SENTENCE_FORMAT=1 OPEN_LORA=1 LORA_RANK=8 GPU=X`

**判断基準**: F1 低下が 0.05 以内で instruction following が改善すれば LoRA 採用を検討。

---

### A-2. データ量アブレーション
**目的**: 試合数を増やすだけで F1 向上・instruction following 改善が起きるか確認。

**アプローチ**:
- 25ゲーム（LoRA なし・sentence format）で学習
- 評価: Test F1（5ゲーム 0.7166 との比較）＋ Phase 4 出力
- 実行: `make run_from_phase2 MAX_GAMES=25 EPOCHS_PHASE2=20 SENTENCE_FORMAT=1 GPU=X`

**判断基準**: F1 が 0.80 以上になれば Phase B の Alignment Tuning の土台として十分。

---

### A-3. Zone・方向ラベルの追加
**目的**: Q-Former が学習する情報を増やし、Stage 1 の出力をより豊かにする。

**アプローチ**:
- `SoccerNet_script/add_task_labels.py` でボール座標から以下を自動計算して clips.json に追加:
  - `zone`: `defensive` / `middle` / `attacking` × `left` / `center` / `right`（3×3 = 9ゾーン）
  - `direction`: `forward` / `backward` / `lateral`（ボールの移動方向）
  - `possession`: `home` / `away`
- `multitask_dataset.py` に zone・direction タスクを追加
- action + zone + direction のマルチタスク学習
- 評価: zone ROUGE-L ＋ action F1（干渉がないか確認）

**判断基準**: zone ROUGE-L > 0.6 が目標（定型文タスクなら正解なら 0.8 以上出るはず）。

---

## Phase B：Alignment Tuning（〜1.5ヶ月）

### B-0. Instruction Diversification（最重要・最初に実装）
**目的**: 「指示文を変えると prefix を無視して LLM 単独で答える」問題を根本解決する。

**なぜ起きるか**: 学習時の指示文が固定（例: `"Describe the play"`）だと、LLM は「この文字列が来たら定型文を返す」という暗記をしてしまい、prefix を読む必要がなくなる。指示文が変わった瞬間に暗記が通じなくなり prefix を無視して幻覚/echo に逃げる。

**アプローチ**:
- 同一トラッキングクリップに対して、複数の指示文バリエーションをランダムに選択して学習:
  ```
  "What action is taking place?"
  "Describe the current soccer scene."
  "What is happening on the pitch?"
  "Analyze this play sequence."
  "Summarize what you see in this tracking data."
  ```
- 指示文が何であっても「答えるには prefix を読むしかない」状況を意図的に作る
- `multitask_dataset.py` の `action` タスクに `instruction_variants` リストを追加し `random.choice` でサンプリング
- 評価: **学習時に使っていない未知の指示文**（例: `"Can you explain the scene?"`）を入力し、prefix に基づく自然文が出れば汎化成功

**注意**: これをやらずに文章化ターゲットや対照学習を入れても、prefix 無視問題は解決しない。

**判断基準**: 未知の指示文 5 種類で 4/5 以上が prefix 依存の自然文を出力すること。

---

### B-1. Self-Rationalization ターゲット設計
**目的**: CoT スタイルで「物理的事実 → 解釈」の順番を学習させ、座標情報を LLM に自然に伝える。

**補足**: 教師データはテキスト文字列として LLM に渡る。Q-Former の出力（prefix）は連続ベクトルであり「pass trap...」という文字列ではない。LLM が prefix を「意味不明なベクトル」として無視しないよう、B-0 と B-3 が必要。

**アプローチ**:
- トラッキング座標から以下を自動計算してターゲット先頭に付加:
  - ボールゾーン（どのエリアか）
  - 攻撃方向（左チャンネル・右チャンネル・中央縦）
  - プレッシャー強度（ボール周辺 5m 以内の相手選手数）
  - 保持チーム（ホーム/アウェイ）
- 学習ターゲットを以下のように拡張:

  ```
  現在: "In this soccer sequence, performing pass and trap."
  提案: "The home team is in the midfield right channel.
         3 opponents are within 5m (high pressure).
         → The home team advances through the right channel under high pressure, performing pass and trap."
  ```
- 「物理的事実 → 矢印 → 解釈」の CoT 構造で Q-Former に因果関係を学習させる
- プレイヤー個人の特定は行わない（順不同性の問題を回避）
- ターゲット生成スクリプトを作成し clips.json を更新

**判断基準**: Phase 4 で「右チャンネルで」「プレッシャー下で」といった物理的文脈が出力に含まれること。

---

### B-2. Slot-based Q-Former 設計
**目的**: 32 個のトークンに役割を割り当て、Q-Former が「何を」どのトークンに圧縮するかを明示化する。

**アプローチ**:
- トークンスロットの役割分担:
  - Token 0-7: action（何が起きているか）
  - Token 8-15: spatial（どこで、どの方向で）
  - Token 16-23: intensity（プレッシャー・速度・密度）
  - Token 24-31: summary（全体の文脈・戦術的意図）
- 各スロットに補助損失（auxiliary loss）を追加:
  - action スロット → action ラベルの cross-entropy
  - spatial スロット → zone ラベルの cross-entropy
  - intensity スロット → pressure の回帰損失
- メイン損失（生成）に補助損失を小さい λ で加算: `L = L_gen + λ * L_aux`
- これにより、Q-Former が「どのトークンに何を入れるか」が自然と学習される

**判断基準**: action スロットの分類精度 > 0.7 ＋ Zone ラベル精度 > 0.6 が目標。

---

### B-3. Token-Word Alignment（対照学習）
**目的**: Q-Former のトークンを LLM の意味空間に直接アラインし、「分類器 + GPT-4」では得られない連続的なニュアンスを保持する。

**核心アイデア**: 「分類器」は "pass" と出した瞬間に連続的な物理情報（速度・角度・空間）を捨てる。対照学習でトークンを LLM の意味空間にプッシュすれば、LLM の推論能力を活用しながら連続的な情報を保持できる。

**アプローチ**:
- アクション語（"pass", "shot", "clearance" 等）の LLM 埋め込み E(w) を取得（凍結）
- Q-Former の action スロット（Token 0-7）の平均プールベクトルを線形 proj で同次元に射影
- アラインメント損失:
  ```
  L_align = 1 - cos(proj(mean(h_action)), E(w_gt))
  ```
  ここで `h_action` = Token 0-7 の Q-Former 出力、`w_gt` = GT アクションの LLM 単語埋め込み
- 最終損失: `L = L_gen + λ_align * L_align + λ_slot * L_slot`（λ は小さい値、生成がメイン）
- この構造を **Structured Prefix Tuning** として位置付ける（Soft Prompt の意味的解釈可能版）

**差別化のロジック**:
- 分類器: 「pass」と分類 → 離散ラベルになった瞬間に連続情報消失 → GPT-4 はテキストからしか推論できない
- 本手法: Q-Former トークン → LLM 意味空間に対照学習でアライン → LLM が「passっぽさの強度」「速度感」を連続量として推論できる

**判断基準**: アライン損失が収束し、Phase 4 LLM-as-a-Judge スコアが「分類器 + GPT-4」ベースラインを上回ること。

---

### B-4. LLM-as-a-Judge 評価の導入
**目的**: F1 では測れない「物理的根拠に基づく解説の質」を評価する。

**アプローチ**:
- GPT-4o に以下の基準で 1〜5 点評価させるスクリプトを作成:
  1. 物理的正確性（座標から導かれる事実と矛盾していないか）
  2. 戦術的妥当性（サッカーの文脈として自然か）
  3. 指示文への追従（質問に答えているか）
  4. 統計量では表せないニュアンス（速度感・予兆・空間的緊張感が含まれるか）
- 比較対象: 「分類器（action labels テキスト） + GPT-4」を baseline として並走評価
- これが「分類器 + GPT-4」との差別化を示す中心的な実験

---

## Phase C：2段推論プロトタイプ（〜3ヶ月）

### C-1. Stage 1 → Stage 2 パイプライン実装
**目的**: Stage 1 の自然言語出力を Stage 2 LLM へのテキスト入力として渡し、任意の指示に回答できるか確認。

**アプローチ**:
- Stage 1: 現モデルで action + zone + direction + pressure を自然言語で生成
  - 出力例: "The home team advances through the right channel in the midfield under high pressure, performing pass and trap."
- Stage 2: Stage 1 出力テキスト + 任意の指示文 → 凍結 LLaMA-3-8B で推論
  - 追加学習なし。LLM の事前知識でテキストから推論させる
  - 例: "Given: [Stage 1 output]. Question: Describe this play as a commentator."
- 評価: 実況・戦術解説・プレー予測の 3 種類の指示で LLM-as-a-Judge

**判断基準**: LLM-as-a-Judge スコアが「分類器 + GPT-4」を上回ること（物理的根拠の豊かさが差になるはず）。

---

## 現状の実験結果サマリー

| Experiment | Test F1 | Instruction Following | Notes |
|---|---|---|---|
| 語彙リスト + LoRA なし (202605042259) | 0.5623 | NG | instruction echo |
| 語彙リスト + LoRA なし (202605050018) | 0.5477 | NG | 指示文なし学習 |
| 自然文 + LoRA なし (202605051303) | **0.7166** | 部分的（形式のみ） | 指示文あり |
| 自然文 + LoRA なし (202605051310) | 0.7466 | NG | 指示文なし学習 |
| 自然文 + LoRA rank=2 (202605061305) | 0.6207 | NG（ガーベジ/echo） | rank 小さすぎ |
| 自然文 + LoRA rank=4 (202605060027) | 0.6894 | **形式・数量を理解** | 内容は幻覚 |
| 自然文 + LoRA rank=8 (202605061327) | **0.7212** | mostly echo | 全実験最高 F1 |

---

## 技術的制約・注意事項

- Python 3.8 環境: `Path.with_stem()` 不使用、`Path.parent / (stem + suffix)` を使う
- GPU サーバー: `solar.arch.cs.kumamoto-u.ac.jp -p 2222`、プロジェクトは `~/M2_Research_MyUni/`
- LLM 凍結が基本方針（LoRA は実験的に解禁）
- 順不同性の問題: プレイヤー個人の特定は現アーキテクチャでは不可。チーム・ゾーン・役割ベースの記述で回避
