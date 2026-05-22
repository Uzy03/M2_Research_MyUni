成果物一覧

- tracking/model/trajectory_regression_model.py
  - compute_loss: when no valid targets, now returns a device-aware zero tensor with requires_grad=True to avoid NaN when pred contains NaN.

- tracking/train_trajectory_regression.py
  - Training loop: added gradient clipping torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) after loss.backward() to prevent gradient explosion.

注意事項

- 変更は指示箇所のみ。その他は未変更。
- 変更後は訓練時のNaN発生が低減されるが、データのNaN発生源は別途確認推奨。
