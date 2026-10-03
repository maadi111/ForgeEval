"""
Reference Solution A — Fix via .shift(1) before rolling.

Strategy: shift the sales series by one period before computing rolling
statistics.  This guarantees that rolling_mean_7d at position i is the
average of sales[i-7 … i-1], strictly excluding the current row.
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

    # Rolling statistics — FIX: shift(1) ensures the current row is excluded.
    # After shifting, position i holds the value from i-1, so rolling(7) at i
    # averages positions [i-7 … i-1] of the original series.
    sales_lagged = df["sales"].shift(1)
    df["rolling_mean_7d"] = sales_lagged.rolling(window=7).mean()
    df["rolling_std_7d"] = sales_lagged.rolling(window=7).std()
    df["rolling_mean_14d"] = sales_lagged.rolling(window=14).mean()

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    return df.dropna().reset_index(drop=True)
