"""
Public test 2 — Rolling feature direction check.

This test gives the candidate a hint that rolling features may use future
data, without revealing exactly how to fix the problem.

It checks whether rolling_mean_7d matches a naive (leaky) computation,
and flags a warning.  The threshold is deliberately loose so the test
passes for a partial fix — the hidden tests enforce full correctness.
"""

import pathlib
import sys

import pandas as pd
import pytest

WORKSPACE = pathlib.Path(__file__).parents[2] / "workspace"
DATA = pathlib.Path(__file__).parents[2] / "data" / "train.csv"
sys.path.insert(0, str(WORKSPACE))

from features import build_features


def test_rolling_mean_differs_from_same_day_value():
    """
    A leaky rolling_mean_7d at position i equals mean(sales[i-6 … i]).
    A correct implementation must differ from this at most positions.

    The test is deliberately lenient (>50% of rows differ) to guide the
    candidate without giving away the hidden-test threshold.
    """
    df = pd.read_csv(DATA)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    featured = build_features(df_sorted)

    # Recompute the leaky version for comparison
    leaky = df_sorted["sales"].rolling(7).mean().rename("leaky")

    merged = (
        featured[["date", "rolling_mean_7d"]]
        .merge(
            leaky.rename("leaky").reset_index().rename(columns={"index": "_idx"}),
            left_index=True,
            right_on="_idx",
            how="inner",
        )
        .dropna()
    )

    if merged.empty:
        pytest.skip("No overlapping rows to compare")

    diff_frac = ((merged["rolling_mean_7d"] - merged["leaky"]).abs() > 1e-9).mean()

    assert diff_frac > 0.5, (
        "rolling_mean_7d looks identical to a naive rolling(7).mean() on "
        "current sales values.  Consider whether your rolling window "
        "accidentally includes the value you are trying to predict."
    )


def test_rolling_std_is_positive():
    """rolling_std_7d must be strictly positive (non-constant window)."""
    df = pd.read_csv(DATA)
    featured = build_features(df)
    zero_rows = (featured["rolling_std_7d"] <= 0).sum()
    assert zero_rows == 0, f"{zero_rows} rows have rolling_std_7d <= 0"
