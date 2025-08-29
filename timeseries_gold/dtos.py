from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    window_size: int = 60
    ratios: Tuple[float, float] = (0.7, 0.3)


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
    idx_train: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    idx_test: np.ndarray

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
