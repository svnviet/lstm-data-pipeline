from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .dtos import DatasetSplit, ScalerBundle, SplitConfig


@dataclass
class SequenceSplitter:
    """
    Builds time-ordered Train/Val/Test windows.
    Uses PROVIDED scalers (no refit) when resuming.
    """

    config: SplitConfig

    def split(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        window_size: int,
        ratios: Sequence[float],
        x_scaler: object,
        y_scaler: object,
    ) -> DatasetSplit:
        # enforce config (and validate), but keep the signature compatible
        w = window_size
        assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1.0"

        for c in list(feature_cols) + list(target_cols):
            if c not in df.columns:
                raise KeyError(
                    f"Column '{c}' not in DataFrame. Available: {list(df.columns)}"
                )

        raw_X = df[list(feature_cols)].to_numpy(dtype=float)
        raw_y = df[list(target_cols)].to_numpy(dtype=float)

        # transform with PROVIDED scalers (do not refit here)
        scaled_X = x_scaler.transform(raw_X)
        scaled_y = y_scaler.transform(raw_y)

        N = len(df)
        if N <= w:
            raise ValueError(f"Need N > window_size. Got N={N}, window_size={w}")

        # number of *target* positions we can produce
        M = N - w

        # NOTE: order is train, val, test  (not train, test, val)
        n_train = int(M * ratios[0])
        n_val = int(M * ratios[1])
        n_test = M - n_train - n_val

        def build(target_indices: range):
            X, y, z, idx = [], [], [], []
            for i in target_indices:
                X.append(scaled_X[i - w : i, :])
                y.append(scaled_y[i, :])
                z.append(raw_y[i, :])  # unscaled targets (for inspection)
                idx.append(i)
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            z = np.asarray(z, dtype=np.float32)
            idx = np.asarray(idx, dtype=np.int64)
            return X, y, z, idx

        train_targets = range(w, w + n_train)
        val_targets = range(w + n_train, w + n_train + n_val)
        test_targets = range(w + n_train + n_val, w + M)

        X_train, y_train, z_train, idx_train = build(train_targets)
        X_val, y_val, z_val, idx_val = build(val_targets)
        X_test, y_test, z_test, idx_test = build(test_targets)

        scalers = ScalerBundle(x_scaler=x_scaler, y_scaler=y_scaler)
        return DatasetSplit(
            X_train,
            y_train,
            z_train,
            idx_train,
            X_test,
            y_test,
            z_test,
            idx_test,
            X_val,
            y_val,
            z_val,
            idx_val,
            tuple(feature_cols),
            tuple(target_cols),
            w,
            scalers,
        )
