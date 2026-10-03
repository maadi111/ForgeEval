"""
Reference Solution B — Fix via rolling(closed='left').

Strategy: use pandas' ``closed='left'`` parameter so the rolling window
is half-open on the right, meaning the current row is always excluded.
This is semantically equivalent to Solution A but uses a different API.

Both approaches are independently valid and should receive full credit.
"""

import pandas as pd

FEATURE_COLS: list[str] = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7d",
    "rolling_std_7d",
    "rolling_mean_14d",
    "day_of_week",
    "month",
    "day_of_month",
]

TARGET_COL: str = "sales"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Lag features
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)

    # Rolling statistics — FIX: closed='left' excludes the current observation.
    # The window [i-6 … i] becomes [i-7 … i-1], strictly past values only.
    df["rolling_mean_7d"] = df["sales"].rolling(window=7, closed="left").mean()
    df["rolling_std_7d"] = df["sales"].rolling(window=7, closed="left").std()
    df["rolling_mean_14d"] = df["sales"].rolling(window=14, closed="left").mean()

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    return df.dropna().reset_index(drop=True)
