import json
from pathlib import Path
from typing import List, Union, Literal

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

Mode = Literal["rolling", "one"]


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = (
        out[cols]
        .replace({r"[,\s]": ""}, regex=True)  # strip thousands & spaces
        .apply(pd.to_numeric, errors="coerce")
    )
    return out


def _build_windows(
        df: pd.DataFrame,
        feature_cols: List[str],
        window_size: int,
        x_scaler,
        timestamp_col: str | None = "Time",
):
    if timestamp_col and timestamp_col in df.columns:
        # keep original time for alignment
        times = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        order = times.argsort(kind="mergesort")
        df = df.iloc[order].reset_index(drop=True)
        times = times.iloc[order].reset_index(drop=True)
    else:
        times = None

    # numeric cleanup
    df = _ensure_numeric(df, feature_cols)

    # transform with fitted scaler
    X_all = df[feature_cols].to_numpy(dtype=float)
    Xs = x_scaler.transform(X_all)

    N = len(Xs)
    if N <= window_size:
        raise ValueError(f"Need > {window_size} rows, got {N}.")

    # sliding windows
    M = N - window_size
    X_windows = np.empty((M, window_size, len(feature_cols)), dtype=float)
    for i in range(M):
        X_windows[i] = Xs[i: i + window_size, :]

    # target indices correspond to rows [window_size .. N-1]
    target_idx = np.arange(window_size, N)
    target_times = times.iloc[target_idx] if times is not None else None
    return X_windows, target_idx, target_times, df


def load_artifacts(
        model_path: Union[str, Path],
        x_scaler_path: Union[str, Path],
        y_scaler_path: Union[str, Path],
        meta_json_path: Union[str, Path] | None = None,
):
    model = load_model(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    meta = {}
    if meta_json_path and Path(meta_json_path).exists():
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return model, x_scaler, y_scaler, meta


def predict_from_df(
        model,
        x_scaler,
        y_scaler,
        df_new: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        window_size: int,
        timestamp_col: str = "Time",
        mode: Mode = "rolling",
) -> pd.DataFrame:
    """
    Returns a DataFrame with predictions in ORIGINAL units.
    - rolling: predicts for every possible target time (len = N - window_size)
    - one: predicts only the next step based on the last window
    """
    # sanity checks
    for c in feature_cols:
        if c not in df_new.columns:
            raise KeyError(f"Missing feature column '{c}' in new data.")
    X_windows, target_idx, target_times, df_ord = _build_windows(
        df_new, feature_cols, window_size, x_scaler, timestamp_col
    )

    if mode == "one":
        X_in = X_windows[-1:]  # (1, window, n_feat)
        y_pred_s = model.predict(X_in, verbose=0)  # scaled
        y_pred = y_scaler.inverse_transform(y_pred_s)  # real units
        # try to infer next timestamp
        if timestamp_col in df_ord.columns:
            ts = pd.to_datetime(df_ord[timestamp_col], errors="coerce", utc=True)
            ts = ts.dropna()
            try:
                freq = pd.infer_freq(ts)
                next_time = (ts.iloc[-1] + pd.tseries.frequencies.to_offset(freq)) if freq else pd.NaT
            except Exception:
                next_time = pd.NaT
        else:
            next_time = pd.NaT

        out = pd.DataFrame(y_pred, columns=[f"{c}_pred" for c in target_cols])
        if not pd.isna(next_time):
            out.insert(0, timestamp_col, next_time)
        return out.reset_index(drop=True)

    # rolling predictions
    y_pred_s = model.predict(X_windows, verbose=0)  # (M, n_targets), scaled
    y_pred = y_scaler.inverse_transform(y_pred_s)  # real units

    out = pd.DataFrame(y_pred, columns=[f"{c}_pred" for c in target_cols])
    if target_times is not None:
        out.insert(0, timestamp_col, target_times.to_numpy())
    out.insert(0, "TargetIndex", target_idx)
    return out.reset_index(drop=True)


def load_and_predict(
        data_source: Union[str, Path, pd.DataFrame],
        model_path: Union[str, Path],
        x_scaler_path: Union[str, Path],
        y_scaler_path: Union[str, Path],
        feature_cols: List[str],
        target_cols: List[str],
        window_size: int,
        timestamp_col: str = "Time",
        mode: Mode = "rolling",
        meta_json_path: Union[str, Path] | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load model+scalers, load CSV/DF, and predict.
    """
    model, x_scaler, y_scaler, _ = load_artifacts(model_path, x_scaler_path, y_scaler_path, meta_json_path)

    if isinstance(data_source, (str, Path)):
        df_new = pd.read_csv(data_source)
    else:
        df_new = data_source.copy()

    # optional time parsing if present
    if timestamp_col in df_new.columns:
        df_new[timestamp_col] = pd.to_datetime(df_new[timestamp_col], utc=True, errors="coerce")

    return predict_from_df(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        df_new=df_new,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=window_size,
        timestamp_col=timestamp_col,
        mode=mode,
    )
