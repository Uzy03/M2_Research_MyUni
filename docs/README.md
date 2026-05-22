# Tracking-LLM: トラッキングデータによるサッカーマルチタスク QA

## 概要

本研究は、サッカーのトラッキングデータ（選手・ボールの位置情報）を LLM が理解できる表現に変換し、**指示文を切り替えるだけでフォーメーション認識・戦術解説・実況生成などのマルチタスク QA を実現する**モデルを提案する。

Claude や GPT-4V が画像について自由に質問に答えられるように、本モデルはトラッキングデータについてゼロショットで QA する。

```mermaid
graph LR
    T["トラッキングデータ\n(x, y, speed × 23選手)"]
    M["Tracking-LLM"]
    Q1["フォーメーションは？"]
    Q2["このプレーの意図は？"]
    Q3["実況してください"]
    A1["4-2-3-1で守備的に..."]
    A2["右サイドを崩して..."]
    A3["左サイドから突破！..."]

    T --> M
    Q1 --> M
    Q2 --> M
    Q3 --> M
    M --> A1
    M --> A2
    M --> A3
```

---

## 動機

### なぜ LLM か？

従来のサッカー AI はタスクごとに専用モデルを学習していた（フォーメーション認識モデル、実況生成モデル、など）。一方、LLM はテキスト・画像・動画を統一的に扱い、**タスクごとのラベルなしにゼロショット QA** を実現している。

本研究はこれをトラッキングデータに拡張する。

| | 従来手法 | 本提案 |
|---|---|---|
| フォーメーション認識 | 専用分類モデル | 指示文を変えるだけ |
| 実況生成 | 専用生成モデル | 指示文を変えるだけ |
| 戦術解説 | 未対応（ラベルがない） | ゼロショット |
| 新タスク追加 | 再学習が必要 | 追加学習不要 |

### BLIP-2 との類比

```mermaid
graph TB
    subgraph 画像版["画像版 (BLIP-2 / GPT-4V)"]
        I["画像"] --> VE["Vision Encoder\n(ViT)"]
        VE --> QF1["Q-Former"]
        QF1 --> LLM1["LLM"]
        LLM1 --> A1["VQA・キャプション\n・ゼロショット QA"]
    end

    subgraph tracking版["本提案 (Tracking-LLM)"]
        TR["トラッキングデータ\n(B, T, 23, 5)"] --> TE["TrackingEncoder\n(Spatial-Temporal Transformer)"]
        TE --> QF2["Q-Former"]
        QF2 --> LLM2["LLM\n(LLaMA-3-8B)"]
        LLM2 --> A2["サッカー QA\n(ゼロショット)"]
    end
```

**鍵となる洞察**: LLM はすでにサッカーの知識（フォーメーション・戦術・ルール）を持っている。必要なのは「トラッキングトークンを LLM が読める形に変換する alignment」のみ。

---

## アーキテクチャ

```mermaid
graph LR
    subgraph input["入力"]
        TRK["Tracking NPY\n(T, 23, 5)\nx, y, speed\nteam_flag, is_ball"]
        INST["指示文\n'フォーメーションは？'"]
    end

    subgraph encoder["TrackingEncoder"]
        EMB["Player Embedding\n5 → 256"]
        SP["Spatial Transformer\n選手間の関係"]
        TMP["Temporal Transformer\n時系列関係"]
        PROJ["出力射影\n256 → 768"]
        EMB --> SP --> TMP --> PROJ
    end

    subgraph qformer["Q-Former (BERT-base, 2層)"]
        QT["Query Tokens\n(32, 768)"]
        CA["Cross-Attention\n with Tracking Features"]
        PFX["Prefix Tokens\n(32, 768)"]
        QT --> CA --> PFX
    end

    subgraph llm["LLM (LLaMA-3-8B)"]
        LPROJ["llama_proj\n768 → 4096"]
        DEC["LLM Decoder\n(凍結 or 軽量 LoRA)"]
        OUT["回答テキスト"]
        LPROJ --> DEC --> OUT
    end

    TRK --> EMB
    PROJ --> CA
    PFX --> LPROJ
    INST --> DEC
```

### 各コンポーネントの役割

| コンポーネント | 役割 | パラメータ数 |
|---|---|---|
| TrackingEncoder | 位置情報 → 時空間特徴 | ~1M |
| Q-Former | 可変長 tracking → 固定 32 トークン | ~14M |
| llama_proj | Q-Former 次元 → LLM 次元 | ~3M |
| LLM | 指示理解 + 回答生成 | 8B（凍結） |

---

## 学習パイプライン

### Phase 1: 軌跡回帰 Pretraining（自己教師あり）

```mermaid
flowchart LR
    subgraph data1["データ (J1リーグ, 95試合)"]
        D1["tracking.csv\n各フレームの選手座標"]
    end

    subgraph model1["モデル"]
        TE1["TrackingEncoder"]
        QF1["Q-Former"]
        MP["Mean Pool\n(32, 768) → (768)"]
        REG["回帰 MLP\n768 → K×N×2"]
    end

    subgraph loss1["損失"]
        MSE["MSE Loss\n予測座標 vs 実座標"]
    end

    D1 -- "過去20秒\n(コンテキスト)" --> TE1 --> QF1 --> MP --> REG --> MSE
    D1 -- "次の5秒\n(正解)" --> MSE

    style data1 fill:#e8f4f8
    style model1 fill:#f8f0e8
    style loss1 fill:#f0f8e8
```

**目的**: 大量の J1 データで TrackingEncoder + Q-Former に「トラッキングデータの空間的・時間的パターン」を学習させる。ラベル不要。

### Phase 2: LLM Alignment（弱教師あり）

```mermaid
flowchart LR
    subgraph data2["データ (J1 play.csv)"]
        D2["アクションラベル\n'パス' / 'シュート'\n'ドリブル' etc."]
    end

    subgraph model2["モデル"]
        TE2["TrackingEncoder\n(Phase 1 重みで初期化)"]
        QF2["Q-Former\n(Phase 1 重みで初期化)"]
        LP["llama_proj\n(学習対象)"]
        LLM2["LLM\n(凍結)"]
    end

    subgraph loss2["損失"]
        LM["Language Model Loss\n正解トークンのみ"]
    end

    D2 -- "指示文:\n'このプレーは？'" --> LLM2
    D2 -- "tracking" --> TE2 --> QF2 --> LP --> LLM2 --> LM
    D2 -- "正解: 'パス'" --> LM

    style data2 fill:#e8f4f8
    style model2 fill:#f8f0e8
    style loss2 fill:#f0f8e8
```

**目的**: tracking トークンと LLM の言語空間を接続する。play.csv の自動抽出ラベルのみ使用（人手アノテーション不要）。

### Phase 3: ゼロショット QA（推論）

```mermaid
flowchart TD
    TRK["トラッキングデータ"]
    TE["TrackingEncoder"]
    QF["Q-Former"]
    LP["llama_proj"]

    TRK --> TE --> QF --> LP

    LP --> LLM

    subgraph LLM["LLM"]
        I1["指示: フォーメーションは？"] --> O1["4-2-3-1\nボランチ2枚で守備的に構成"]
        I2["指示: 相手の意図は？"] --> O2["右サイドのスペースを使って\nクロスを狙っている"]
        I3["指示: 今のプレーを実況して"] --> O3["右サイドから中央へ\nワンツーで崩しにかかる！"]
        I4["指示: 次の5秒の座標を予測して"] --> O4["p0:[(0.45,0.32),...]\np1:[(0.51,0.28),...]"]
    end
```

---

## データ

| データセット | 試合数 | 内容 | 用途 |
|---|---|---|---|
| J1リーグ tracking (SoccerData) | 95試合 | tracking.csv, play.csv, players.csv | Phase 1 pretraining, Phase 2 alignment |
| SoccerNet tracking | 44クリップ | MOT形式 + 実況テキスト | 補助学習・評価 |
| SoccerReplay-1988 | 1,988クリップ | 映像 + アクションラベル | 補助学習 |

### J1データのフォーマット

```
tracking.csv: GameID, Frame, HA, SysTarget, No, X, Y, Speed
play.csv:     フレーム番号, アクションID, アクション名, 試合状態
players.csv:  ホームアウェイF, 背番号, スタメン
```

---

## 評価

| タスク | メトリクス | データ |
|---|---|---|
| 軌跡予測 | ADE / FDE | J1 test split |
| アクション分類 | Accuracy / F1 | play.csv test split |
| ゼロショット QA | Human eval / BLEU | 手動作成問題セット |

---

## 研究上の貢献

1. **新モダリティの LLM 統合**: トラッキングデータ（時系列座標）を LLM が扱える最初のフレームワーク
2. **自己教師あり pretraining**: 軌跡回帰により大量のラベルなしデータを活用
3. **ゼロショット マルチタスク**: 単一モデルで指示文を変えるだけで複数タスクに対応
4. **実用性**: 実際の J1 データで検証

---

## 実装状況

- [x] TrackingEncoder（Spatial-Temporal Transformer）
- [x] Q-Former + LLM パイプライン（matchvoice_model_tracking）
- [x] SoccerData 前処理スクリプト（preprocess_soccerdata.py）
- [x] 軌跡回帰モデル（TrajectoryRegressionModel）
- [x] 軌跡回帰学習スクリプト（train_trajectory_regression.py）
- [ ] play.csv を使った alignment 学習
- [ ] ゼロショット QA 評価スクリプト
- [ ] フォーメーション自動ラベリング
