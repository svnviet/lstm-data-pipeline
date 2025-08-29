from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
import pandas as pd
from .dtos import DatasetSplit


class Predictor:
    """Utility for inference on new dataframes using fitted scalers & model."""

    def __init__(self, model, feature_cols: Sequence[str], target_cols: Sequence[str], window_size: int, x_scaler, y_scaler):
        self.model = model
        self.feature_cols = tuple(feature_cols)
        self.target_cols = tuple(target_cols)
        self.window_size = int(window_size)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def _to_windows(self, scaled_X: np.ndarray) -> np.ndarray:
        w = self.window_size
        N = scaled_X.shape[0]
        if N <= w:
            raise ValueError(f"Need N > window_size. Got N={N}, window_size={w}")
        M = N - w
        windows = np.stack([scaled_X[i:i+w, :] for i in range(M)], axis=0)
        return windows  # (M, w, F)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assumes df is already preprocessed like training (Time parsed/sorted, numeric types).
        Returns a DataFrame aligned to df.index[self.window_size:] containing predictions (REAL units).
        """
        for c in self.feature_cols:
            if c not in df.columns:
                raise KeyError(f"Missing feature '{c}' in new dataframe")

        raw_X = df[list(self.feature_cols)].to_numpy(dtype=float)
        scaled_X = self.x_scaler.transform(raw_X)
        X_win = self._to_windows(scaled_X)

        y_pred_s = self.model.predict(X_win, verbose=0)
        y_pred_inv = self.y_scaler.inverse_transform(y_pred_s)

        out_idx = df.index[self.window_size: self.window_size + y_pred_inv.shape[0]]
        pred_df = pd.DataFrame(y_pred_inv, index=out_idx, columns=self.target_cols)
        return pred_df

    def predict_next_from_tail(self, df_tail: pd.DataFrame) -> np.ndarray:
        """
        Given the last `window_size` rows of features, predict the next target values (REAL units).
        """
        if len(df_tail) != self.window_size:
            raise ValueError(f"df_tail must have exactly window_size={self.window_size} rows")
        raw = df_tail[list(self.feature_cols)].to_numpy(dtype=float)
        scaled = self.x_scaler.transform(raw)
        X = np.expand_dims(scaled, axis=0)  # (1, w, F)
        y_pred_s = self.model.predict(X, verbose=0)[0]
        y_inv = self.y_scaler.inverse_transform(y_pred_s.reshape(1, -1))[0]
        return y_inv  # shape (n_targets,)
