"""
Hidden test — Temporal integrity.

Why these tests exist
---------------------
The core property of a point-in-time feature pipeline is that the feature
value at time T must be identical regardless of what data exists after T.
A leaky pipeline violates this: adding or perturbing future rows changes
past feature values, which is impossible in a real serving system.

Three independent checks are run:

1. Sort invariance  — shuffling rows must not change any feature value.
2. Future-row perturbation — adding ±9999 noise to rows after a cut-off
   must leave all features before the cut-off unchanged.
3. Incremental consistency — computing features on a prefix df[:T] must
   yield the same feature vector at T as computing on the full dataset.
"""

import pathlib
import sys

import pandas as pd
import pytest

WORKSPACE = pathlib.Path(__file__).parents[2] / "workspace"
DATA = pathlib.Path(__file__).parents[2] / "data" / "train.csv"
DATA_DIR = pathlib.Path(__file__).parents[2] / "data"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(DATA_DIR))

from features import FEATURE_COLS, build_features


@pytest.fixture(scope="module")
def base_df():
    return pd.read_csv(DATA).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Sort invariance
# ---------------------------------------------------------------------------


def test_sort_invariant(base_df):
    """
    Shuffling input rows must produce identical feature values.

    Rationale: a correct implementation sorts internally before computing
    features.  A buggy implementation that relies on row order will produce
    different feature values when given shuffled input.
    """
    featured_sorted = build_features(base_df)
    shuffled = base_df.sample(frac=1, random_state=99).reset_index(drop=True)
    featured_shuffled = build_features(shuffled)

    merged = featured_sorted.merge(featured_shuffled, on="date", suffixes=("_s", "_sh"))

    for col in FEATURE_COLS:
        col_s, col_sh = f"{col}_s", f"{col}_sh"
        if col_s not in merged.columns:
            continue
        max_diff = (merged[col_s] - merged[col_sh]).abs().max()
        assert max_diff < 1e-6, (
            f"Feature '{col}' is not sort-invariant (max diff={max_diff:.6f}). "
            "The pipeline must sort the input before computing features."
        )


# ---------------------------------------------------------------------------
# 2. Future-row perturbation
# ---------------------------------------------------------------------------


def test_future_row_perturbation_does_not_affect_past_features(base_df):
    """
    Perturbing rows after a cut-off must not change features before the cut-off.

    This directly tests point-in-time correctness: if rolling_mean_7d at date T
    uses sales[T], then artificially inflating sales[T] will alter features at
    T-1, T-2, …, T-6 in the next window.  A fixed pipeline prevents this.
    """
    n = len(base_df)
    pivot = n // 2

    featured_clean = build_features(base_df)

    df_perturbed = base_df.copy()
    df_perturbed.loc[pivot:, "sales"] += 9999  # enormous signal to reveal leakage
    featured_perturbed = build_features(df_perturbed)

    cutoff_date = base_df["date"].iloc[pivot]

    orig_sub = featured_clean[featured_clean["date"].astype(str) <= str(cutoff_date)]
    pert_sub = featured_perturbed[featured_perturbed["date"].astype(str) <= str(cutoff_date)]

    merged = orig_sub.merge(pert_sub, on="date", suffixes=("_o", "_p"))
    assert len(merged) > 0, "No rows available before cut-off for comparison"

    for col in FEATURE_COLS:
        co, cp = f"{col}_o", f"{col}_p"
        if co not in merged.columns:
            continue
        max_diff = (merged[co] - merged[cp]).abs().max()
        assert max_diff < 1e-6, (
            f"Feature '{col}' at dates before {cutoff_date} changed when "
            f"future rows were perturbed (max diff={max_diff:.4f}). "
            "This indicates that the rolling window is not point-in-time correct."
        )


# ---------------------------------------------------------------------------
# 3. Incremental consistency
# ---------------------------------------------------------------------------


def test_feature_is_identical_with_or_without_subsequent_rows(base_df):
    """
    The feature vector at date T must not depend on rows after T.

    Concretely: build_features(df[:T]) at row T must equal
    build_features(df[:T+10]) at row T for all feature columns.
    """
    pivot = len(base_df) // 2 + 30
    date_t = base_df["date"].iloc[pivot]

    featured_prefix = build_features(base_df.iloc[: pivot + 1])
    featured_extended = build_features(base_df.iloc[: pivot + 11])

    row_prefix = featured_prefix[featured_prefix["date"].astype(str) == str(date_t)]
    row_extended = featured_extended[featured_extended["date"].astype(str) == str(date_t)]

    if row_prefix.empty or row_extended.empty:
        pytest.skip(f"Date {date_t} not present in both feature sets")

    for col in FEATURE_COLS:
        v1 = row_prefix[col].values[0]
        v2 = row_extended[col].values[0]
        assert abs(v1 - v2) < 1e-6, (
            f"Feature '{col}' at {date_t} differs when computed on "
            f"prefix vs extended dataset ({v1:.4f} vs {v2:.4f}). "
            "The feature must not depend on future rows."
        )
