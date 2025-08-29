from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Sequence


class CsvPreprocessor:


    """
    Loads and cleans the Exness M1 CSV in the same way as your script:
    - parse Time to UTC
    - sort by Time & reset index
    - drop Spread/RealVolume (if present)
    - strip commas from numeric cols and cast to float64
    """

    def __init__(self, drop_cols: Iterable[str] = ("Spread", "RealVolume")):
        self.drop_cols = tuple(drop_cols)

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
        # Time handling
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

        return df
