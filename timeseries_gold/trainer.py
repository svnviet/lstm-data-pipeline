from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from .dtos import TrainConfig, DatasetSplit, TrainReport


class Trainer:
    def __init__(self, model):
        self.model = model

    def fit(self, ds: DatasetSplit, cfg: TrainConfig) -> TrainReport:
        hist = self.model.fit(
            ds.X_train, ds.y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_data=(ds.X_val, ds.y_val),
            verbose=cfg.verbose,
        )

        test_loss = float(self.model.evaluate(ds.X_test, ds.y_test, verbose=0))

        # Predictions (scaled)
        y_pred_s = self.model.predict(ds.X_test, verbose=0)

        # Back to real units (inverse of y scaler)
        y_test_inv = ds.scalers.y_scaler.inverse_transform(ds.y_test)
        y_pred_inv = ds.scalers.y_scaler.inverse_transform(y_pred_s)

        # Metrics per target
        mape_raw = mean_absolute_percentage_error(y_test_inv, y_pred_inv, multioutput='raw_values')
        acc_raw = 1.0 - mape_raw

        mape_per_target = {t: float(m) for t, m in zip(ds.target_cols, mape_raw)}
        acc_per_target = {t: float(a) for t, a in zip(ds.target_cols, acc_raw)}

        history_dict = {k: list(map(float, v)) for k, v in hist.history.items()}

        return TrainReport(
            test_loss_scaled_mse=test_loss,
            mape_per_target=mape_per_target,
            accuracy_per_target=acc_per_target,
            history=history_dict,
        )
