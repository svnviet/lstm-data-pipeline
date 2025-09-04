from __future__ import annotations
import os
from datetime import datetime

from timeseries_gold.data import CsvPreprocessor
from timeseries_gold.split import SequenceSplitter
from timeseries_gold.model import ModelBuilder
from timeseries_gold.trainer import Trainer
from timeseries_gold.dtos import SplitConfig, TrainConfig
from timeseries_gold.predict import Predictor

import joblib
import json

FEATURE_COLS = ["Open", "High", "Low", "Close", "TickVolume"]
TARGET_COLS = ["Open", "High", "Low", "Close", "TickVolume"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), f"artifacts_{datetime.now().strftime('%Y%m%d')}")


def run_training(csv_path: str) -> None:
    # 1) Load & preprocess
    prep = CsvPreprocessor()
    df_raw = prep.load(csv_path)
    df = prep.preprocess(df_raw)

    # 2) Split
    splitter = SequenceSplitter(SplitConfig(window_size=60, ratios=(0.7, 0.3)))
    ds = splitter.split(df, FEATURE_COLS, TARGET_COLS)

    # 3) Model
    model = ModelBuilder.lstm_stacked(ds.window_size, len(ds.feature_cols), len(ds.target_cols))
    model.summary()

    # 4) Train & report
    report = Trainer(model).fit(ds, TrainConfig(epochs=300, batch_size=32, verbose=1))

    print("Test Loss (scaled MSE):", report.test_loss_scaled_mse)
    print("Test MAPE per target:", report.mape_per_target)
    print("Test Accuracy per target:", report.accuracy_per_target)

    # 5) Inference on the test portion (or full df)
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

    # 6) Example: predict the very next step from the tail window
    tail = df.iloc[-ds.window_size:]
    next_pred = predictor.predict_next_from_tail(tail)
    print("Next-step prediction (real units, order TARGET_COLS):", next_pred)

    # 7) Save the model
    model.save(os.path.join(MODEL_PATH, "model.keras"))
    joblib.dump(ds.scalers.x_scaler, os.path.join(MODEL_PATH, "x_scaler.joblib"))
    joblib.dump(ds.scalers.y_scaler, os.path.join(MODEL_PATH, "y_scaler.joblib"))
    with open((os.path.join(MODEL_PATH, "meta.json")), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS, "target_cols": TARGET_COLS, "window_size": ds.window_size}, f)


def load_model_predict():
    import os, json, joblib
    import pandas as pd
    from keras.models import load_model

    MODEL_PATH = "timeseries_gold/artifacts_20250903"
    FUTURE_STEPS = 30

    def load_predictor() -> Predictor:
        # 1) Load model & scalers
        model = load_model(os.path.join(MODEL_PATH, "model.h5"))
        x_scaler = joblib.load(os.path.join(MODEL_PATH, "x_scaler.joblib"))
        y_scaler = joblib.load(os.path.join(MODEL_PATH, "y_scaler.joblib"))

        # 2) Load metadata
        with open(os.path.join(MODEL_PATH, "meta.json"), "r") as f:
            meta = json.load(f)

        feature_cols = meta["feature_cols"]
        target_cols = meta["target_cols"]
        window_size = meta["window_size"]

        # 3) Return Predictor object
        return Predictor(
            model=model,
            feature_cols=feature_cols,
            target_cols=target_cols,
            window_size=window_size,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )

    # === Usage ===

    # Load predictor
    predictor = load_predictor()

    # Load new data
    df_new = pd.read_csv("new_gold_data.csv", parse_dates=["Time"], index_col="Time")
    df_new = df_new.sort_index()  # must be sorted ascending

    future_df = predictor.predict_future(df_new, steps=30)

    # 7) Save to CSV if needed
    future_df.to_csv("future_predictions.csv", index=False)
    print("Predictions saved to future_predictions.csv")


if __name__ == "__main__":
    # Example run
    csv_path = os.environ.get("GOLD_CSV", "xauusd_M1_exness_2025-08-01.csv")
    run_training(csv_path)

    # load_model_predict()
