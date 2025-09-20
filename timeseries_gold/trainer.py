# trainer.py

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
from typing import Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_percentage_error

from . import ModelBuilder
from .data import CsvPreprocessor
from .dtos import DatasetSplit, ScalerBundle, SplitConfig, TrainConfig, TrainReport
from .split import SequenceSplitter


class Trainer:
    def __init__(self, model=None):
        self.model = model
        self.builder = ModelBuilder()

    # ----------------------------
    # Build default callbacks
    # ----------------------------
    @staticmethod
    def _default_callbacks(outdir: str, monitor: str = "val_loss"):
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        return [
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1,
            ),
            EarlyStopping(
                monitor=monitor,
                patience=20,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=os.path.join(outdir, "model.keras"),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
        ]

    # ----------------------------
    # Shared training logic
    # ----------------------------
    def _run_fit(
        self,
        run_dir: str,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int,
        initial_epoch: int,
        batch_size: int,
        verbose: int,
        callbacks: Optional[Sequence],
    ):
        # Make sure model is compiled
        self.builder.ensure_compiled(self.model)

        cbs = list(callbacks) if callbacks else self._default_callbacks(run_dir)

        hist = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=cbs,
            shuffle=False,
        )

        # Save last-epoch model & training history
        self.model.save(os.path.join(run_dir, "model.keras"))
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f)

        return hist

    # ----------------------------
    # Standard fresh training
    # ----------------------------
    def fit(
        self, ds: DatasetSplit, cfg: TrainConfig, base_dir: str = "artifacts"
    ) -> Tuple[str, TrainReport]:
        run_dir = os.path.join(
            base_dir, dt.datetime.now().strftime("run_%Y%m%dT%H%M%S")
        )
        pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

        X_train, y_train = np.asarray(ds.X_train, np.float32), np.asarray(ds.y_train, np.float32)
        X_val, y_val = np.asarray(ds.X_val, np.float32), np.asarray(ds.y_val, np.float32)

        hist = self._run_fit(
            run_dir,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=cfg.epochs,
            initial_epoch=0,
            batch_size=cfg.batch_size,
            verbose=cfg.verbose,
            callbacks=None,
        )

        # Save scalers + meta
        joblib.dump(ds.scalers.x_scaler, os.path.join(run_dir, "x_scaler.joblib"))
        joblib.dump(ds.scalers.y_scaler, os.path.join(run_dir, "y_scaler.joblib"))
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "feature_cols": ds.feature_cols,
                    "target_cols": ds.target_cols,
                    "window_size": ds.window_size,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        report = self._evaluate_and_report(ds, hist.history)
        return run_dir, report

    # ----------------------------
    # Resume training when ds & model are already in memory
    # ----------------------------
    def resume(
        self,
        ds: DatasetSplit,
        epochs_more: int,
        initial_epoch: int = 0,
        batch_size: Optional[int] = None,
        verbose: Optional[int] = None,
        callbacks: Optional[Sequence] = None,
        base_dir: str = "artifacts",
    ) -> Tuple[str, TrainReport]:
        run_dir = os.path.join(
            base_dir, dt.datetime.now().strftime("resume_%Y%m%dT%H%M%S")
        )
        pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

        X_train, y_train = np.asarray(ds.X_train, np.float32), np.asarray(ds.y_train, np.float32)
        X_val, y_val = np.asarray(ds.X_val, np.float32), np.asarray(ds.y_val, np.float32)

        hist = self._run_fit(
            run_dir,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=initial_epoch + epochs_more,
            initial_epoch=initial_epoch,
            batch_size=batch_size or 32,
            verbose=1 if verbose is None else verbose,
            callbacks=callbacks,
        )

        report = self._evaluate_and_report(ds, hist.history)
        return run_dir, report

    # ----------------------------
    # One-shot resume from saved artifacts
    # ----------------------------
    def resume_from_artifacts(
        self,
        artifact_dir: str,
        csv_path: str,
        epochs_more: int = 50,
        initial_epoch: Optional[int] = None,
        batch_size: int = 32,
        ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        callbacks: Optional[Sequence] = None,
        save_to: Optional[str] = None,
        verbose: int = 1,
    ) -> Tuple[str, TrainReport]:

        self.model, x_scaler, y_scaler, meta, prev_epoch_hist, orig_fmt = (
            self.builder._load_artifacts(artifact_dir)
        )
        feature_cols = meta["feature_cols"]
        target_cols = meta["target_cols"]
        window_size = int(meta["window_size"])

        cp = CsvPreprocessor()
        df = cp.preprocess(cp.load(csv_path))

        splitter = SequenceSplitter(config=SplitConfig(window_size=window_size, ratios=ratios))
        ds = splitter.split(
            df=df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            window_size=window_size,
            ratios=ratios,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )

        start_epoch = int(initial_epoch) if initial_epoch is not None else int(prev_epoch_hist)
        end_epoch = start_epoch + int(epochs_more)

        run_dir = save_to or os.path.join(
            artifact_dir, "resume_" + dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

        X_train, y_train = np.asarray(ds.X_train, np.float32), np.asarray(ds.y_train, np.float32)
        X_val, y_val = np.asarray(ds.X_val, np.float32), np.asarray(ds.y_val, np.float32)

        hist = self._run_fit(
            run_dir,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=end_epoch,
            initial_epoch=start_epoch,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        # Save scalers + meta alongside model
        self.model.save(os.path.join(run_dir, f"model.{orig_fmt}"))
        joblib.dump(x_scaler, os.path.join(run_dir, "x_scaler.joblib"))
        joblib.dump(y_scaler, os.path.join(run_dir, "y_scaler.joblib"))
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        report = self._evaluate_and_report(ds, hist.history)
        return run_dir, report

    # ----------------------------
    # Helpers
    # ----------------------------
    def _evaluate_and_report(self, ds: DatasetSplit, history: Dict[str, list]) -> TrainReport:
        X_test = np.asarray(ds.X_test, np.float32)
        y_test = np.asarray(ds.y_test, np.float32)

        eval_res = self.model.evaluate(X_test, y_test, verbose=0, return_dict=True)
        test_loss = float(eval_res["loss"]) if isinstance(eval_res, dict) else float(eval_res[0])

        y_pred_s = self.model.predict(X_test, verbose=0)
        y_test_inv = ds.scalers.y_scaler.inverse_transform(y_test)
        y_pred_inv = ds.scalers.y_scaler.inverse_transform(y_pred_s)

        mape_raw = mean_absolute_percentage_error(y_test_inv, y_pred_inv, multioutput="raw_values")
        acc_raw = 1.0 - mape_raw

        mape_per_target = {t: float(m) for t, m in zip(ds.target_cols, mape_raw)}
        acc_per_target = {t: float(a) for t, a in zip(ds.target_cols, acc_raw)}

        history_dict = {k: [float(x) for x in v] for k, v in history.items()}
        return TrainReport(
            test_loss_scaled_mse=test_loss,
            mape_per_target=mape_per_target,
            accuracy_per_target=acc_per_target,
            history=history_dict,
        )
