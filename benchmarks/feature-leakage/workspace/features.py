"""
Feature pipeline for daily sales forecasting.

BUG (intentional): The rolling statistics include the current row's value,
meaning features at time T incorporate data that would not be available
at prediction time. This inflates validation metrics while production
performance collapses.

Your task: repair this pipeline so that every feature at time T uses only
information available strictly before T.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix from raw sales data.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``date`` (str / date) and ``sales`` (numeric).

    Returns
    -------
    pd.DataFrame
        Input rows enriched with feature columns.  Rows with NaN features
        are dropped.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Lag features (correct: uses strictly past values) ---
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)

    # --- Rolling statistics (BUG: window includes the current row) ----------
    # rolling(7).mean() at position i averages rows [i-6 … i], which means
    # sales[i] — the value we are trying to predict — is part of the feature.
    # This information is unavailable at real prediction time.
    df["rolling_mean_7d"] = df["sales"].rolling(window=7).mean()
    df["rolling_std_7d"] = df["sales"].rolling(window=7).std()
    df["rolling_mean_14d"] = df["sales"].rolling(window=14).mean()

    # --- Calendar features (correct: derived from the date, not the target) --
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    return df.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column declarations (shared by train.py and predict.py)
# ---------------------------------------------------------------------------

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
