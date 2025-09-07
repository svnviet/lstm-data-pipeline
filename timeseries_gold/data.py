from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable
import ta


class CsvPreprocessor:
    """
    Loads and cleans the Exness M1 CSV:
    - parse Time to UTC
    - sort by Time & reset index
    - drop Spread/RealVolume (if present)
    - add engineered features (OHLCV, returns, volatility, time, technicals, day-of-week)
    - session assumed 01:05–23:54 UTC, Monday–Friday
    """

    def __init__(self, drop_cols: Iterable[str] = ("Spread", "RealVolume")):
        self.drop_cols = tuple(drop_cols)

        # All engineered features (now 29 total, including day-of-week)
        self.feature_cols = [
            # Returns
            "Close_ret", "Open_ret", "High_ret", "Low_ret",
            # Volatility
            "Range",
            # Time features
            "minute_of_day", "slot5", "slot5_sin", "slot5_cos",
            "minutes_from_open", "minutes_to_close", "percent_session_elapsed",
            "is_open", "is_close",
            # Day of week
            "day_of_week", "dow_sin", "dow_cos", "is_monday", "is_friday",
            # Technical indicators
            "SMA10", "SMA20", "EMA10", "VWAP",
            "RSI14", "MACD", "MACD_signal", "MACD_diff",
            "Bollinger_high", "Bollinger_low", "Bollinger_mavg",
            "ATR14"
        ]

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        empty_rows = df.isnull().all(axis=1)
        if empty_rows.any():
            print("Empty rows found at indices:")
            print(empty_rows[empty_rows].index)
        else:
            print("No empty rows found in the dataset.")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # --- Time handling ---
        if "Time" not in df.columns:
            raise KeyError("Expected 'Time' column in CSV")
        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        df.sort_values(by="Time", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Drop extra cols if present
        df = df.drop(columns=list(self.drop_cols), errors="ignore")

        # numeric columns (all except Time)
        num_cols = df.columns.drop(["Time"]).tolist()
        df[num_cols] = df[num_cols].replace({',': ''}, regex=True)
        df[num_cols] = df[num_cols].astype("float64")

        # Add engineered features
        df = self._add_features(df)

        # Drop NaNs introduced by rolling indicators
        df = df.dropna().reset_index(drop=True)

        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # --- Returns ---
        for col in ["Close", "Open", "High", "Low"]:
            df[f"{col}_ret"] = df[col].pct_change()

        # --- Volatility (Range) ---
        df["Range"] = df["High"] - df["Low"]

        # --- Time features ---
        df["minute_of_day"] = df["Time"].dt.hour * 60 + df["Time"].dt.minute
        df["slot5"] = df["minute_of_day"] // 5
        df["slot5_sin"] = np.sin(2 * np.pi * df["slot5"] / 288)
        df["slot5_cos"] = np.cos(2 * np.pi * df["slot5"] / 288)

        # Define daily session boundaries (UTC 01:05 → 23:54)
        session_open = 65  # 01:05 = 60 + 5
        session_close = 1434  # 23:54 = 23*60 + 54

        df["minutes_from_open"] = df["minute_of_day"] - session_open
        df["minutes_to_close"] = session_close - df["minute_of_day"]
        df["percent_session_elapsed"] = (
                df["minutes_from_open"] / (session_close - session_open)
        ).clip(0, 1)

        # Flags
        df["is_open"] = (df["minute_of_day"] == session_open).astype(int)
        df["is_close"] = (df["minute_of_day"] == session_close).astype(int)

        # --- Day-of-week features ---
        df["day_of_week"] = df["Time"].dt.dayofweek  # 0=Mon … 4=Fri
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        # --- Technical Indicators ---
        df["SMA10"] = df["Close"].rolling(10).mean()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

        # RSI (14)
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        # MACD (12, 26, 9)
        macd = ta.trend.MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        # Bollinger Bands (20, 2)
        boll = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["Bollinger_high"] = boll.bollinger_hband()
        df["Bollinger_low"] = boll.bollinger_lband()
        df["Bollinger_mavg"] = boll.bollinger_mavg()

        # ATR (14)
        atr = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14
        )
        df["ATR14"] = atr.average_true_range()

        return df
