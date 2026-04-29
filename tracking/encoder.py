import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TrackingEncoder(nn.Module):
    """
    Spatial-Temporal Transformer エンコーダ
    
    SoccerDataのトラッキングデータ（選手・ボールの座標×時系列）を受け取り、
    VisionTimesformerと同じ出力形状 (B, T, 768) を返す。
    
    Args:
        num_players (int): 選手数。デフォルト23（選手22 + ボール1）
        in_features (int): 入力特徴次元。デフォルト5（x_norm, y_norm, speed_norm, team_flag, is_ball）
        d_model (int): トランスフォーマー内部次元
        nhead (int): マルチヘッドアテンションのヘッド数
        num_spatial_layers (int): 空間トランスフォーマーの層数
        num_temporal_layers (int): 時間トランスフォーマーの層数
        out_features (int): 出力特徴次元（VisionTimesformerに合わせて768）
    """

    def __init__(
        self,
        num_players: int = 23,
        in_features: int = 5,
        d_model: int = 256,
        nhead: int = 4,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 2,
        out_features: int = 768,
    ):
        super().__init__()

        self.num_players = num_players
        self.in_features = in_features
        self.d_model = d_model

        # 入力特徴をモデル次元に埋め込む
        self.player_embed = nn.Linear(in_features, d_model)

        # 空間トランスフォーマー（各タイムスタンプで選手間の関係をモデル）
        spatial_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.spatial_transformer = TransformerEncoder(
            spatial_layer, num_layers=num_spatial_layers
        )

        # 時間トランスフォーマー（タイムステップ間の時系列関係をモデル）
        temporal_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal_transformer = TransformerEncoder(
            temporal_layer, num_layers=num_temporal_layers
        )

        # 出力投影層
        self.out_proj = nn.Linear(d_model, out_features)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): 入力トラッキングデータ。shape (B, T, N, F)
                B: バッチサイズ
                T: タイムステップ数
                N: 選手数（num_players）
                F: 特徴次元（in_features）
            mask (torch.Tensor, optional): マスク。shape (B, T, N) のbool tensor。
                Trueが欠損選手。Noneなら全て有効。

        Returns:
            torch.Tensor: 出力特徴。shape (B, T, out_features=768)
        """
        B, T, N, F = x.shape

        # 入力を埋め込む (B, T, N, d_model)
        x = self.player_embed(x)

        # 空間アテンション：各タイムステップで選手間の関係をモデル
        # (B, T, N, d_model) -> (B*T, N, d_model)
        x = x.reshape(B * T, N, self.d_model)

        # マスクの処理
        src_key_padding_mask = None
        if mask is not None:
            # (B, T, N) -> (B*T, N)
            src_key_padding_mask = mask.reshape(B * T, N)
            # If all players in a frame are masked, unmask them to avoid NaN
            # in softmax (softmax of all -inf = NaN in PyTorch)
            all_masked = src_key_padding_mask.all(dim=-1, keepdim=True)
            src_key_padding_mask = src_key_padding_mask & ~all_masked

        # 空間トランスフォーマー
        x = self.spatial_transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 選手次元でmean pooling（マスク済み選手を除外）(B*T, N, d_model) -> (B*T, d_model)
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(-1)  # (B*T, N, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # (B*T, d_model) -> (B, T, d_model)
        x = x.reshape(B, T, self.d_model)

        # 時間アテンション：タイムステップ間の時系列関係をモデル
        x = self.temporal_transformer(x)

        # 出力投影 (B, T, d_model) -> (B, T, out_features=768)
        x = self.out_proj(x)

        return x
