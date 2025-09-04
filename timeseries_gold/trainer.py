from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import os, json, pathlib, datetime as dt
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_percentage_error

# Keras 3 / TF 2.20
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.saving import load_model

from .split import SequenceSplitter
from .dtos import TrainConfig, DatasetSplit, TrainReport, ScalerBundle, SplitConfig
from .data import CsvPreprocessor


class Trainer:
    def __init__(self, model):
        self.model = model

    # ----------------------------
    # Standard fresh training
    # ----------------------------
    def fit(self, ds: DatasetSplit, cfg: TrainConfig) -> TrainReport:
        # make sure inputs are float32
        X_train = np.asarray(ds.X_train, dtype=np.float32)
        y_train = np.asarray(ds.y_train, dtype=np.float32)
        X_val = np.asarray(ds.X_val, dtype=np.float32)
        y_val = np.asarray(ds.y_val, dtype=np.float32)

        # rely on the caller to have compiled the model; otherwise compile here
        if not getattr(self.model, "optimizer", None):
            self.model.compile(
                optimizer=Adam(1e-3),
                loss="mse",
                metrics=["mae"],
                run_eagerly=True,  # avoid graph-mode numpy() crash when resuming
                jit_compile=False,
            )

        hist = self.model.fit(
            X_train, y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_data=(X_val, y_val),
            verbose=cfg.verbose,
            shuffle=False,  # time series
        )
        return self._evaluate_and_report(ds, hist.history)

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
    ) -> TrainReport:
        end_epoch = initial_epoch + int(epochs_more)

        X_train = np.asarray(ds.X_train, dtype=np.float32)
        y_train = np.asarray(ds.y_train, dtype=np.float32)
        X_val = np.asarray(ds.X_val, dtype=np.float32)
        y_val = np.asarray(ds.y_val, dtype=np.float32)

        if not getattr(self.model, "optimizer", None):
            self.model.compile(
                optimizer=Adam(1e-3),
                loss="mse",
                metrics=["mae"],
                run_eagerly=True,
                jit_compile=False,
            )

        hist = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=end_epoch,
            initial_epoch=initial_epoch,
            batch_size=batch_size or 32,
            verbose=1 if verbose is None else verbose,
            callbacks=list(callbacks) if callbacks else None,
            shuffle=False,
        )
        return self._evaluate_and_report(ds, hist.history)

    # ----------------------------
    # One-shot resume from saved artifacts (model + scalers + meta)
    # ----------------------------
    @staticmethod
    def resume_from_artifacts(
            artifact_dir: str,
            csv_path: str,
            epochs_more: int = 50,
            initial_epoch: Optional[int] = None,
            batch_size: int = 32,
            ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),  # train, val, test
            callbacks: Optional[Sequence] = None,
            save_to: Optional[str] = None,
            verbose: int = 1,
    ) -> tuple[str, TrainReport]:

        # 1) Load artifacts
        model, x_scaler, y_scaler, meta, prev_epoch_hist, orig_fmt = Trainer._load_artifacts(artifact_dir)
        feature_cols = meta["feature_cols"]
        target_cols = meta["target_cols"]
        window_size = int(meta["window_size"])

        # 2) Preprocess CSV
        cp = CsvPreprocessor()
        df = cp.preprocess(cp.load(csv_path))

        # 3) Build windows WITHOUT refitting scalers
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

        # 4) Continue training (force eager to avoid numpy() crash)
        start_epoch = int(initial_epoch) if initial_epoch is not None else int(prev_epoch_hist)
        end_epoch = start_epoch + int(epochs_more)

        planned_cbs = list(callbacks) if callbacks else [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ]
        Trainer._print_model_resume_info(
            model=model,
            artifact_dir=artifact_dir,
            orig_fmt=orig_fmt,
            ds=ds,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            epochs_more=int(epochs_more),
            planned_optimizer="Adam(1e-3)",
            planned_loss="mse",
            planned_metrics=("mae",),
            callbacks=planned_cbs,
        )

        model.compile(
            optimizer=Adam(1e-3),
            loss="mse",
            metrics=["mae"],
            run_eagerly=True,
            jit_compile=False,
        )

        X_train = np.asarray(ds.X_train, dtype=np.float32)
        y_train = np.asarray(ds.y_train, dtype=np.float32)
        X_val = np.asarray(ds.X_val, dtype=np.float32)
        y_val = np.asarray(ds.y_val, dtype=np.float32)

        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=end_epoch,
            initial_epoch=start_epoch,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=list(callbacks) if callbacks else [
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
                EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
            ],
            shuffle=False,
        )

        # 5) Report
        trainer = Trainer(model)
        report = trainer._evaluate_and_report(ds, hist.history)

        # 6) Save updated artifacts
        outdir = save_to or os.path.join(
            artifact_dir,
            "resume_" + dt.datetime.now().strftime("%Y%m%dT%H%M%SZ"),
        )
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

        model_path = os.path.join(outdir, f"model.{orig_fmt}")
        model.save(model_path)
        joblib.dump(x_scaler, os.path.join(outdir, "x_scaler.joblib"))
        joblib.dump(y_scaler, os.path.join(outdir, "y_scaler.joblib"))
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(outdir, "history.json"), "w") as f:
            json.dump({k: list(map(float, v)) for k, v in hist.history.items()}, f)

        return outdir, report

    # ----------------------------
    # Helpers
    # ----------------------------
    def _evaluate_and_report(self, ds: DatasetSplit, history: Dict[str, list]) -> TrainReport:
        test_loss = float(self.model.evaluate(ds.X_test, ds.y_test, verbose=0))

        y_pred_s = self.model.predict(ds.X_test, verbose=0)
        y_test_inv = ds.scalers.y_scaler.inverse_transform(ds.y_test)
        y_pred_inv = ds.scalers.y_scaler.inverse_transform(y_pred_s)

        mape_raw = mean_absolute_percentage_error(y_test_inv, y_pred_inv, multioutput="raw_values")
        acc_raw = 1.0 - mape_raw

        mape_per_target = {t: float(m) for t, m in zip(ds.target_cols, mape_raw)}
        acc_per_target = {t: float(a) for t, a in zip(ds.target_cols, acc_raw)}

        history_dict = {k: list(map(float, v)) for k, v in history.items()}
        return TrainReport(
            test_loss_scaled_mse=test_loss,
            mape_per_target=mape_per_target,
            accuracy_per_target=acc_per_target,
            history=history_dict,
        )

    @staticmethod
    def _load_artifacts(art_dir: str):
        # Prefer modern .keras, fall back to .h5
        model = None
        fmt = None
        for ext in ("keras", "h5"):
            p = os.path.join(art_dir, f"model.{ext}")
            if os.path.exists(p):
                model = load_model(p, compile=False)  # we'll recompile with eager
                fmt = ext
                break
        if model is None:
            raise FileNotFoundError("No model.h5 or model.keras found in artifacts dir")

        def _pick(*names):
            for n in names:
                q = os.path.join(art_dir, n)
                if os.path.exists(q):
                    return q
            raise FileNotFoundError(f"Missing {' or '.join(names)} in {art_dir}")

        x_scaler = joblib.load(_pick("x_scaler.joblib", "x_scaler.bin"))
        y_scaler = joblib.load(_pick("y_scaler.joblib", "y_scaler.bin"))

        with open(os.path.join(art_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        prev_epoch = 0
        hist_path = os.path.join(art_dir, "history.json")
        if os.path.exists(hist_path):
            try:
                with open(hist_path, "r") as f:
                    hist = json.load(f)
                # number of completed epochs = length of history
                prev_epoch = int(len(hist.get("loss", [])))
            except Exception:
                pass

        return model, x_scaler, y_scaler, meta, prev_epoch, fmt

    @staticmethod
    def _print_model_resume_info(
        *,
        model,
        artifact_dir: str,
        orig_fmt: str,
        ds: DatasetSplit,
        start_epoch: int,
        end_epoch: int,
        epochs_more: int,
        planned_optimizer: str = "Adam(1e-3)",
        planned_loss: str = "mse",
        planned_metrics: Sequence[str] = ("mae",),
        callbacks: Optional[Sequence] = None,
    ) -> None:
        print("\n=== Resume Training: Model & Data Info ===")
        print(f"Artifacts: {artifact_dir} (model.{orig_fmt})")

        # Model basics
        try:
            name = getattr(model, "name", "model")
            in_shape = getattr(model, "input_shape", None)
            out_shape = getattr(model, "output_shape", None)
            total_params = model.count_params()
            trainable_params = int(sum(int(np.prod(v.shape)) for v in model.trainable_weights))
            non_trainable_params = int(sum(int(np.prod(v.shape)) for v in model.non_trainable_weights))
        except Exception:
            name = getattr(model, "name", "model")
            in_shape = out_shape = total_params = trainable_params = non_trainable_params = "?"
        print(f"Model: {name}")
        print(f"  Input shape:  {in_shape}")
        print(f"  Output shape: {out_shape}")
        print(f"  Params: total={total_params} | trainable={trainable_params} | non-trainable={non_trainable_params}")

        # Layer stack (best-effort)
        try:
            for lyr in model.layers:
                oshape = getattr(lyr, "output_shape", "?")
                print(f"    - {lyr.name:24s} {lyr.__class__.__name__:20s} -> {oshape}")
        except Exception:
            pass

        # Dataset shapes / sizes
        print("Data:")
        print(f"  window_size={ds.window_size} | features={len(ds.feature_cols)} | targets={len(ds.target_cols)}")
        print(f"  Train: X={ds.X_train.shape}, y={ds.y_train.shape}")
        print(f"  Val:   X={ds.X_val.shape},   y={ds.y_val.shape}")
        print(f"  Test:  X={ds.X_test.shape},  y={ds.y_test.shape}")

        # Resume plan
        print("Resume plan:")
        print(f"  start_epoch={start_epoch} -> end_epoch={end_epoch} (add {epochs_more} epochs)")
        print("Planned compile:")
        print(f"  optimizer={planned_optimizer}, loss={planned_loss}, metrics={list(planned_metrics)}")
        if callbacks:
            names = []
            for cb in callbacks:
                try:
                    names.append(cb.__class__.__name__)
                except Exception:
                    names.append(str(cb))
            print(f"Callbacks: {names}")
        print("=========================================\n")
