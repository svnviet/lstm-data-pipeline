# model.py
from __future__ import annotations
from typing import Sequence, Tuple, Optional

from keras import Model
from keras import layers
from keras.optimizers import Adam


class ModelBuilder:
    @staticmethod
    def lstm_stacked(
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
        metrics: Sequence[str] = ("mae",),
        run_eagerly: bool = True,   # helps avoid numpy() crash on resume
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
        inp = layers.Input(shape=(window_size, n_features), dtype="float32", name="inputs")

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
                loss=loss,
                metrics=list(metrics),
                run_eagerly=run_eagerly,
                jit_compile=jit_compile,
            )

        return model

    @staticmethod
    def ensure_compiled(
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
                loss=loss,
                metrics=list(metrics),
                run_eagerly=run_eagerly,
                jit_compile=jit_compile,
            )
        return model
