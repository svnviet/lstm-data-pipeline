from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .dtos import SplitConfig, DatasetSplit, ScalerBundle


@dataclass
class SequenceSplitter:
    config: SplitConfig
    x_scaler: MinMaxScaler = MinMaxScaler()
    y_scaler: MinMaxScaler = MinMaxScaler()

    def split(self,
              df: pd.DataFrame,
              feature_cols: Sequence[str],
              target_cols: Sequence[str]) -> DatasetSplit:
        ratios = self.config.ratios
        assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1.0"

        for c in list(feature_cols) + list(target_cols):
            if c not in df.columns:
                raise KeyError(f"Column '{c}' not in DataFrame. Available: {list(df.columns)}")

        raw_X = df[list(feature_cols)].to_numpy(dtype=float)
        raw_y = df[list(target_cols)].to_numpy(dtype=float)

        scaled_X = self.x_scaler.fit_transform(raw_X)
        scaled_y = self.y_scaler.fit_transform(raw_y)

        N = len(df)
        w = self.config.window_size
        if N <= w:
            raise ValueError(f"Need N > window_size. Got N={N}, window_size={w}")

        M = N - w
        n_train = int(M * ratios[0])
        n_test = int(M * ratios[1])

        train_targets = range(w, w + n_train)
        test_targets = range(w + n_train, w + n_train + n_test)

        def build_split(target_indices: range):
            X, y, z, idx = [], [], [], []
            for i in target_indices:
                X.append(scaled_X[i - w:i, :])
                y.append(scaled_y[i, :])
                idx.append(i)
            return np.array(X), np.array(y), np.array(idx, dtype=int)

        X_train, y_train, idx_train = build_split(train_targets)
        X_test, y_test, idx_test = build_split(test_targets)

        scalers = ScalerBundle(self.x_scaler, self.y_scaler)
        return DatasetSplit(
            X_train, y_train, idx_train,
            X_test, y_test, idx_test,
            tuple(feature_cols), tuple(target_cols), w, scalers
        )
