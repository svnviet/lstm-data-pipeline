from typing import Sequence
import numpy as np
import pandas as pd
import ta


class Predictor:
    def __init__(self, model, feature_cols: Sequence[str], target_cols: Sequence[str],
                 window_size: int, x_scaler, y_scaler):
        self.model = model
        self.feature_cols = list(feature_cols)   # FIX: use list
        self.target_cols = list(target_cols)     # FIX: use list
        self.window_size = int(window_size)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler


    # --- Helper: add engineered features automatically ---
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features (returns, volatility, time, day-of-week, technicals)."""
        df = df.copy()

        # --- Returns ---
        for col in ["Close", "Open", "High", "Low"]:
            if f"{col}_ret" not in df.columns:
                df[f"{col}_ret"] = df[col].pct_change()

        # --- Volatility (Range) ---
        if "Range" not in df.columns:
            df["Range"] = df["High"] - df["Low"]

        # --- Time features ---
        if "minute_of_day" not in df.columns:
            df["minute_of_day"] = df["Time"].dt.hour * 60 + df["Time"].dt.minute
        if "slot5" not in df.columns:
            df["slot5"] = df["minute_of_day"] // 5
            df["slot5_sin"] = np.sin(2 * np.pi * df["slot5"] / 288)
            df["slot5_cos"] = np.cos(2 * np.pi * df["slot5"] / 288)

        # Session boundaries (UTC 01:05 → 23:54)
        session_open, session_close = 65, 1434
        df["minutes_from_open"] = df["minute_of_day"] - session_open
        df["minutes_to_close"] = session_close - df["minute_of_day"]
        df["percent_session_elapsed"] = (
            df["minutes_from_open"] / (session_close - session_open)
        ).clip(0, 1)

        # Flags
        df["is_open"] = (df["minute_of_day"] == session_open).astype(int)
        df["is_close"] = (df["minute_of_day"] == session_close).astype(int)

        # --- Day-of-week features ---
        df["day_of_week"] = df["Time"].dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        # --- Technical Indicators ---
        if "SMA10" not in df.columns:
            df["SMA10"] = df["Close"].rolling(10).mean()
        if "SMA20" not in df.columns:
            df["SMA20"] = df["Close"].rolling(20).mean()
        if "EMA10" not in df.columns:
            df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
        if "VWAP" not in df.columns:
            df["VWAP"] = (df["Close"] * df["TickVolume"]).cumsum() / df["TickVolume"].cumsum()

        if "RSI14" not in df.columns:
            df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        if "MACD" not in df.columns:
            macd = ta.trend.MACD(close=df["Close"])
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
            df["MACD_diff"] = macd.macd_diff()

        if "Bollinger_high" not in df.columns:
            boll = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
            df["Bollinger_high"] = boll.bollinger_hband()
            df["Bollinger_low"] = boll.bollinger_lband()
            df["Bollinger_mavg"] = boll.bollinger_mavg()

        if "ATR14" not in df.columns:
            atr = ta.volatility.AverageTrueRange(
                high=df["High"], low=df["Low"], close=df["Close"], window=14
            )
            df["ATR14"] = atr.average_true_range()

        # Drop NaNs introduced by indicators
        return df.dropna()

    # --- Helper: convert to sliding windows ---
    def _to_windows(self, scaled_X: np.ndarray) -> np.ndarray:
        w = self.window_size
        N = scaled_X.shape[0]
        if N <= w:
            raise ValueError(f"Need N > window_size. Got N={N}, window_size={w}")
        M = N - w
        return np.stack([scaled_X[i:i + w, :] for i in range(M)], axis=0)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_features(df)

        for c in self.feature_cols:
            if c not in df.columns:
                raise KeyError(f"Missing feature '{c}' in new dataframe (expected {c})")

        raw_X = df[list(self.feature_cols)].to_numpy(dtype=float)
        scaled_X = self.x_scaler.transform(raw_X)
        X_win = self._to_windows(scaled_X)

        y_pred_s = self.model.predict(X_win, verbose=0)
        y_pred_inv = self.y_scaler.inverse_transform(y_pred_s)

        out_idx = df.index[self.window_size: self.window_size + y_pred_inv.shape[0]]
        return pd.DataFrame(y_pred_inv, index=out_idx, columns=self.target_cols)

    def predict_next_from_tail(self, df_tail: pd.DataFrame) -> np.ndarray:
        df_tail = self._add_features(df_tail)

        if len(df_tail) != self.window_size:
            raise ValueError(f"df_tail must have exactly window_size={self.window_size} rows")

        raw = df_tail[list(self.feature_cols)].to_numpy(dtype=float)
        scaled = self.x_scaler.transform(raw)
        X = np.expand_dims(scaled, axis=0)

        y_pred_s = self.model.predict(X, verbose=0)[0]
        return self.y_scaler.inverse_transform(y_pred_s.reshape(1, -1))[0]

    def predict_future(self, df: pd.DataFrame, steps: int = 30) -> pd.DataFrame:
        df = self._add_features(df)

        for c in self.feature_cols:
            if c not in df.columns:
                raise KeyError(f"Missing feature '{c}' in new dataframe (expected {c})")

        # working copy
        df_work = df.copy()
        preds = []

        for _ in range(steps):
            # last window_size rows
            seq = df_work.tail(self.window_size)
            X_in = seq[self.feature_cols].to_numpy(dtype=float)
            X_in = self.x_scaler.transform(X_in).reshape(1, self.window_size, len(self.feature_cols))

            # predict next OHLCV
            y_pred_s = self.model.predict(X_in, verbose=0)[0]
            y_pred = self.y_scaler.inverse_transform(y_pred_s.reshape(1, -1))[0]

            # build next row
            next_row = {col: val for col, val in zip(self.target_cols, y_pred)}
            next_row["Time"] = df_work["Time"].iloc[-1] + pd.Timedelta(minutes=1)

            # append and recompute engineered features
            df_work = pd.concat([df_work, pd.DataFrame([next_row])], ignore_index=True)
            df_work = self._add_features(df_work)

            preds.append(y_pred)

        return pd.DataFrame(preds, columns=self.target_cols)
