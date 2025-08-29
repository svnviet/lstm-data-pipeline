from __future__ import annotations
import os
from timeseries_gold.data import CsvPreprocessor
from timeseries_gold.split import SequenceSplitter
from timeseries_gold.model import ModelBuilder
from timeseries_gold.trainer import Trainer
from timeseries_gold.dtos import SplitConfig, TrainConfig
from timeseries_gold.predict import Predictor

FEATURE_COLS = ["Open", "High", "Low", "Close", "TickVolume"]
TARGET_COLS = ["Open", "Close", "High", "Low"]


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
    report = Trainer(model).fit(ds, TrainConfig(epochs=100, batch_size=32, verbose=1))

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


if __name__ == "__main__":
    # Example run
    csv_path = os.environ.get("GOLD_CSV", "xauusd_M1_exness_2025-08-25.csv")
    run_training(csv_path)
