from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    window_size: int = 60
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)  # train, test, val


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    verbose: int = 1


@dataclass(frozen=True)
class ScalerBundle:
    x_scaler: Any
    y_scaler: Any


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    z_train: np.ndarray
    idx_train: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    z_test: np.ndarray
    idx_test: np.ndarray

    X_val: np.ndarray
    y_val: np.ndarray
    z_val: np.ndarray
    idx_val: np.ndarray

    feature_cols: Tuple[str, ...]
    target_cols: Tuple[str, ...]
    window_size: int
    scalers: ScalerBundle


@dataclass(frozen=True)
class TrainReport:
    test_loss_scaled_mse: float
    mape_per_target: Dict[str, float]
    accuracy_per_target: Dict[str, float]
    history: Optional[Dict[str, list]] = None
