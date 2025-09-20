from __future__ import annotations

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from keras.saving import load_model  # FIX: Keras 3 path
from sklearn.preprocessing import StandardScaler

from timeseries_gold.visualiza import Visualizer
from .data import CsvPreprocessor
from .dtos import SplitConfig, TrainConfig
from .model import ModelBuilder
from .predict import Predictor
from .split import SequenceSplitter
from .trainer import Trainer

FEATURE_COLS = [
    # --- Base OHLCV ---
    "Open",
    "High",
    "Low",
    "Close",
    "TickVolume",
    # --- Returns ---
    "Close_ret",
    "Open_ret",
    "High_ret",
    "Low_ret",
    # --- Volatility ---
    "Range",
    # --- Time features ---
    "minute_of_day",
    "slot5",
    "slot5_sin",
    "slot5_cos",
    "minutes_from_open",
    "minutes_to_close",
    "percent_session_elapsed",
    "is_open",
    "is_close",
    # --- Day of week ---
    "day_of_week",
    "dow_sin",
    "dow_cos",
    "is_monday",
    "is_friday",
    # --- Technical indicators ---
    "SMA10",
    "SMA20",
    "EMA10",
    "VWAP",
    "RSI14",
    "MACD",
    "MACD_signal",
    "MACD_diff",
    "Bollinger_high",
    "Bollinger_low",
    "Bollinger_mavg",
    "ATR14",
]

TARGET_COLS = ["Open", "High", "Low", "Close", "TickVolume"]

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), f"artifacts_{datetime.now().strftime('%Y%m%d')}"
)


def _fit_scalers_on_train_only(
        df: pd.DataFrame, feature_cols, target_cols, window_size: int, ratios
):
    """
    Fit StandardScalers using ONLY the training span:
    - X scaler on rows [0 : w + n_train)
    - y scaler on rows [w : w + n_train)   (targets exist only for indices >= w)
    """
    w = int(window_size)
    r_train, r_val, r_test = ratios
    N = len(df)
    if N <= w:
        raise ValueError(f"Need N > window_size. Got N={N}, window_size={w}")
    M = N - w

    n_train = int(M * r_train)

    # Training span
    X_train_span = df[feature_cols].iloc[: w + n_train]
    y_train_span = df[target_cols].iloc[w: w + n_train]

    # StandardScaler instead of MinMaxScaler
    x_scaler = StandardScaler().fit(X_train_span.to_numpy(dtype=float))
    y_scaler = StandardScaler().fit(y_train_span.to_numpy(dtype=float))
    return x_scaler, y_scaler


def run_training(csv_path: str) -> None:
    os.makedirs(MODEL_PATH, exist_ok=True)

    # 1) Load & preprocess
    prep = CsvPreprocessor()
    df_raw = prep.load(csv_path)
    df = prep.preprocess(df_raw)  # UTC parse, sort asc, numeric, drop extras

    # 2) Split (train/val/test), fitting scalers on TRAIN ONLY
    window_size = 60
    ratios = (0.7, 0.2, 0.1)

    x_scaler, y_scaler = _fit_scalers_on_train_only(
        df, FEATURE_COLS, TARGET_COLS, window_size, ratios
    )

    splitter = SequenceSplitter(SplitConfig(window_size=window_size, ratios=ratios))
    ds = splitter.split(
        df=df,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        window_size=window_size,
        ratios=ratios,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )

    # 3) Model (FIX: correct parameter names)
    model = model_builder.lstm_stacked(
        window_size=ds.window_size,
        n_features=len(ds.feature_cols),
        n_targets=len(ds.target_cols),
    )

    model.summary()

    # 4) Train & report
    trainer = Trainer(model)
    cfg = TrainConfig(epochs=100, batch_size=32, verbose=1)
    report = trainer.fit(ds, cfg)

    # 5) Quick inference sanity checks
    predictor = Predictor(
        model=model,
        feature_cols=ds.feature_cols,
        target_cols=ds.target_cols,
        window_size=ds.window_size,
        x_scaler=ds.scalers.x_scaler,
        y_scaler=ds.scalers.y_scaler,
    )
    pred_df = predictor.predict_dataframe(df)
    print("Pred head:\n", pred_df.head())

    tail = df.iloc[-ds.window_size:]
    next_pred = predictor.predict_next_from_tail(tail)
    print("Next-step prediction (real units, order TARGET_COLS):", next_pred)

    # 6) Save artifacts
    model.save(os.path.join(MODEL_PATH, "model.keras"))  # FIX: use .keras
    joblib.dump(ds.scalers.x_scaler, os.path.join(MODEL_PATH, "x_scaler.joblib"))
    joblib.dump(ds.scalers.y_scaler, os.path.join(MODEL_PATH, "y_scaler.joblib"))
    with open(os.path.join(MODEL_PATH, "meta.json"), "w") as f:
        json.dump(
            {
                "feature_cols": list(ds.feature_cols),
                "target_cols": list(ds.target_cols),
                "window_size": int(ds.window_size),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # (Optional) Save training history
    if getattr(report, "history", None):
        with open(os.path.join(MODEL_PATH, "history.json"), "w") as f:
            json.dump({k: list(map(float, v)) for k, v in report.history.items()}, f)

    print("Saved artifacts to:", MODEL_PATH)


def load_model_predict():
    """
    Load artifacts and run future prediction on a new CSV.
    """
    FUTURE_STEPS = 30
    ART_NAME = "artifacts_20250918"
    ART_DIR = "/Users/vietnguyen/Projects/prediction/artifacts_20250918"  # adjust as needed

    # 1) Load model & scalers (FIX: .keras + Keras 3 loader)
    model = load_model(os.path.join(ART_DIR, "model.keras"), compile=False)
    x_scaler = joblib.load(os.path.join(ART_DIR, "x_scaler.joblib"))
    y_scaler = joblib.load(os.path.join(ART_DIR, "y_scaler.joblib"))

    # 2) Load metadata
    with open(os.path.join(ART_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    window_size = int(meta["window_size"])

    # 3) Build predictor
    predictor = Predictor(
        model=model,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=window_size,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )

    # 4) Load & preprocess new data (FIX: keep preprocessing consistent)
    cp = CsvPreprocessor()
    df_new = cp.preprocess(cp.load("new_gold_data.csv"))

    # 5) Predict future
    future_df = predictor.predict_future(df_new, steps=FUTURE_STEPS)

    # 6) Save
    future_df.to_csv(f"future_predictions_{ART_NAME}.csv", index=False)
    print("Predictions saved to future_predictions.csv")


def retrain_model() -> None:
    trainer = Trainer()
    outdir, report = trainer.resume_from_artifacts(
        artifact_dir="timeseries_gold/artifacts_20250918",
        csv_path="xauusd_M1_exness_2025.csv",
        epochs_more=50,
        # initial_epoch=300,  # or None to auto-detect from history.json
        batch_size=32,
    )
    print("Saved to:", outdir)
    print(report.mape_per_target)


def show_visualization() -> None:
    vis = Visualizer("new_gold_data.csv", "real_data.csv", "future_predictions_artifacts_20250907.csv", n_history=200)
    vis.plot_predictions_all()


if __name__ == "__main__":
    csv_path = os.environ.get("GOLD_CSV", "xauusd_M1_exness_2025.csv")
    model_builder = ModelBuilder()

    # run_training(csv_path)
    # load_model_predict()
    retrain_model()

    # show_visualization()
