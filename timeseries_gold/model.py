# model.py
from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Tuple

import joblib
import tensorflow as tf
from keras import Model, layers
from keras.models import load_model
from keras.optimizers import Adam


class ModelBuilder:
    def lstm_stacked(
        self,
        window_size: int,
        n_features: int,
        n_targets: int,
        *,
        units: Tuple[int, int, int] = (128, 128, 64),
        dense_units: Optional[int] = 32,
        dropout: float = 0.0,
        compile_: bool = False,
        # compile args (used only if compile_=True)
        lr: float = 1e-3,
        loss: str = "mse",
        metrics: Sequence[str] = ("mae", "mse"),
        run_eagerly: bool = True,  # helps avoid numpy() crash on resume
        jit_compile: bool = False,
    ) -> Model:
        """
        Build an LSTM regressor:
          - 3 stacked LSTMs (return_sequences on first two)
          - Optional Dense head
          - Linear output with n_targets units

        Set compile_=True if you want this function to compile the model.
        Otherwise, call ModelBuilder.ensure_compiled(...) later.
        """
        inp = layers.Input(
            shape=(window_size, n_features), dtype="float32", name="inputs"
        )

        x = layers.LSTM(units[0], return_sequences=True, name="lstm_1")(inp)
        if dropout:
            x = layers.Dropout(dropout, name="drop_1")(x)

        x = layers.LSTM(units[1], return_sequences=True, name="lstm_2")(x)
        if dropout:
            x = layers.Dropout(dropout, name="drop_2")(x)

        x = layers.LSTM(units[2], return_sequences=False, name="lstm_3")(x)

        if dense_units and dense_units > 0:
            x = layers.Dense(dense_units, activation="relu", name="dense")(x)

        out = layers.Dense(n_targets, name="outputs")(x)

        model = Model(inp, out, name="lstm_stacked_regressor")

        if compile_:
            model.compile(
                optimizer=Adam(lr),
                loss=self._constrained_ohlc_loss(5.0),
                metrics=list(metrics),
                run_eagerly=run_eagerly,
                jit_compile=jit_compile,
            )

        return model

    def ensure_compiled(
        self,
        model: Model,
        *,
        lr: float = 1e-3,
        loss: str = "mse",
        metrics: Sequence[str] = ("mae",),
        run_eagerly: bool = True,
        jit_compile: bool = False,
    ) -> Model:
        """
        Compile only if not already compiled.
        Use this for retraining after load_model(..., compile=False).
        """
        if not getattr(model, "optimizer", None):
            model.compile(
                optimizer=Adam(lr),
                loss=self._constrained_ohlc_loss(5.0),
                metrics=list(metrics),
                run_eagerly=run_eagerly,
                jit_compile=jit_compile,
            )
        return model

    def _constrained_ohlc_loss(self, lambda_penalty: float = 5.0):
        """
        Custom loss for OHLC prediction with constraints:
        - High >= max(Open, Close, Low)
        - Low  <= min(Open, Close, High)

        Args:
            lambda_penalty: weight of the constraint penalty relative to MSE.

        Returns:
            A loss function to be used in model.compile(loss=...).
        """

        def loss_fn(y_true, y_pred):
            # split into columns
            o_p, h_p, l_p, c_p, v_p = tf.split(y_pred, 5, axis=-1)

            # base regression loss (MSE across all outputs)
            mse = tf.reduce_mean(tf.square(y_true - y_pred))

            # constraint 1: High must be >= Open, Close, Low
            high_violation = (
                tf.nn.relu(o_p - h_p) + tf.nn.relu(c_p - h_p) + tf.nn.relu(l_p - h_p)
            )

            # constraint 2: Low must be <= Open, Close, High
            low_violation = (
                tf.nn.relu(l_p - o_p) + tf.nn.relu(l_p - c_p) + tf.nn.relu(l_p - h_p)
            )

            # average penalty
            penalty = tf.reduce_mean(high_violation + low_violation)

            return mse + lambda_penalty * penalty

        return loss_fn

    def load(
        self,
        path: str,
        *,
        compile_: bool = False,
        lr: float = 1e-3,
        metrics: Sequence[str] = ("mae", "mse"),
        run_eagerly: bool = True,
        jit_compile: bool = False,
    ) -> Model:
        """
        Load a model from a .keras file.
        If compile_=True, recompile with constrained_ohlc_loss.
        Otherwise, load with compile=False and use ensure_compiled later.
        """
        if compile_:
            model = load_model(
                path,
                custom_objects={"loss_fn": self._constrained_ohlc_loss(5.0)},
                compile=False,  # recompile manually to avoid errors
            )
            model.compile(
                optimizer=Adam(lr),
                loss=self._constrained_ohlc_loss(5.0),
                metrics=list(metrics),
                run_eagerly=run_eagerly,
                jit_compile=jit_compile,
            )
        else:
            model = load_model(
                path,
                custom_objects={"loss_fn": self._constrained_ohlc_loss(5.0)},
                compile=False,
            )

        return model

    def _load_artifacts(self, art_dir: str):
        custom_objects = {"loss_fn": self._constrained_ohlc_loss(5.0)}
        model = None
        fmt = None
        for ext in ("keras", "h5"):
            p = os.path.join(art_dir, f"model.{ext}")
            if os.path.exists(p):
                model = load_model(p, custom_objects=custom_objects, compile=True)
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
                prev_epoch = int(len(hist.get("loss", [])))
            except Exception:
                pass

        return model, x_scaler, y_scaler, meta, prev_epoch, fmt
