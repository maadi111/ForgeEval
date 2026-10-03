"""
Public test 1 — Basic feature hygiene.

These tests are visible to the candidate and serve as a development guide.
They verify that build_features returns the expected columns with no NaNs.
They are intentionally incomplete: passing them is necessary but not sufficient.
"""

import pathlib
import sys

import pandas as pd
import pytest

# Point at the workspace being evaluated
WORKSPACE = pathlib.Path(__file__).parents[2] / "workspace"
DATA = pathlib.Path(__file__).parents[2] / "data" / "train.csv"
sys.path.insert(0, str(WORKSPACE))

from features import FEATURE_COLS, TARGET_COL, build_features


@pytest.fixture(scope="module")
def featured_df():
    df = pd.read_csv(DATA)
    return build_features(df)


def test_all_feature_columns_present(featured_df):
    """Every declared feature column must appear in the output."""
    missing = [c for c in FEATURE_COLS if c not in featured_df.columns]
    assert not missing, f"Missing feature columns: {missing}"


def test_target_column_present(featured_df):
    """Target column must survive the transformation."""
    assert TARGET_COL in featured_df.columns


def test_no_nans_in_features(featured_df):
    """build_features must drop incomplete rows; no NaN may remain."""
    for col in FEATURE_COLS:
        bad = featured_df[col].isna().sum()
        assert bad == 0, f"Column '{col}' contains {bad} NaN(s)"


def test_output_is_sorted_by_date(featured_df):
    """Rows must be in ascending date order after feature building."""
    dates = pd.to_datetime(featured_df["date"])
    assert (dates.diff().dropna() >= pd.Timedelta(0)).all(), "Output is not sorted by date"


def test_feature_count():
    """Exactly 9 feature columns must be declared."""
    assert len(FEATURE_COLS) == 9, f"Expected 9 features, got {len(FEATURE_COLS)}"


def test_lag_1_is_previous_row(featured_df):
    """lag_1 at row i must equal the sales value at row i-1."""
    df = featured_df.reset_index(drop=True)
    # Check a sample of consecutive pairs
    for i in range(5, min(20, len(df))):
        expected = df.loc[i - 1, TARGET_COL]
        actual = df.loc[i, "lag_1"]
        assert abs(actual - expected) < 1e-6, (
            f"lag_1 mismatch at row {i}: got {actual}, expected {expected}"
        )
