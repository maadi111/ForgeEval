"""
Hidden test — Negative controls (known-bad solutions must fail).

Why these tests exist
---------------------
The grading system must reject shortcuts that game visible tests without
solving the actual problem.  This file documents and verifies that three
known-bad implementations FAIL the evaluation criteria.

These are not testing the candidate's code directly — they test that the
grading harness is adversarially sound.  They run the buggy implementations
through the same checks as a real submission.

Controls verified:
  NC-1  Constant predictor      — always returns mean(train_sales)
  NC-2  Hardcoded output        — returns memorised training labels
  NC-3  Still-leaky pipeline    — the original unmodified features.py

All three must achieve MAE > 20 on the adversarial holdout
and/or fail the temporal integrity check.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).parents[2] / "data"
WORKSPACE = pathlib.Path(__file__).parents[2] / "workspace"
sys.path.insert(0, str(DATA_DIR))

from generate_data import generate_sales_data

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TRAIN_MEAN = 130.0  # approximate; constant predictor uses this value

MAE_FAIL_THRESHOLD = 20.0  # same threshold as test_unseen_timestamps


def _adversarial_data():
    return generate_sales_data(start="2025-01-01", end="2025-06-30", seed=99)


def _compute_mae(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.abs(preds - actuals).mean())


# ---------------------------------------------------------------------------
# NC-1: Constant predictor
# ---------------------------------------------------------------------------


def test_nc1_constant_predictor_fails_mae_threshold():
    """
    NC-1: A predictor that always returns TRAIN_MEAN must fail the MAE threshold.

    If this test fails, the MAE threshold is too lenient and should be lowered.
    """
    holdout = _adversarial_data()
    y_true = holdout["sales"].values
    preds = np.full(len(y_true), TRAIN_MEAN)
    mae = _compute_mae(preds, y_true)

    assert mae > MAE_FAIL_THRESHOLD, (
        f"NC-1: Constant predictor MAE={mae:.2f} is below threshold {MAE_FAIL_THRESHOLD}. "
        "The threshold is too lenient — tighten it."
    )


# ---------------------------------------------------------------------------
# NC-2: Hardcoded / memorised output
# ---------------------------------------------------------------------------


def test_nc2_memorised_output_fails_on_adversarial_seed():
    """
    NC-2: Returning memorised training labels must fail on adversarial data.

    The adversarial holdout uses seed=99 and dates 2025-01-01 – 2025-06-30,
    so any outputs memorised from the visible training set (seed=42, 2023–2024)
    will not match and must produce high MAE.
    """
    train_df = generate_sales_data(start="2023-01-01", end="2024-06-30", seed=42)
    memorised = train_df["sales"].values

    holdout = _adversarial_data()
    y_true = holdout["sales"].values

    # Repeat or truncate memorised values to match holdout length
    n = len(y_true)
    preds = np.tile(memorised, (n // len(memorised)) + 1)[:n]

    mae = _compute_mae(preds, y_true)
    assert mae > MAE_FAIL_THRESHOLD, (
        f"NC-2: Memorised-output predictor MAE={mae:.2f} is below threshold. "
        "Hidden data overlaps or is too similar to training data."
    )


# ---------------------------------------------------------------------------
# NC-3: Still-leaky pipeline fails temporal integrity
# ---------------------------------------------------------------------------


def test_nc3_leaky_pipeline_fails_temporal_integrity():
    """
    NC-3: The original (unmodified) features.py with rolling(7) must fail
    the future-row perturbation check.

    This verifies that the hidden integrity test actually catches the bug.
    """

    def _leaky_build_features(df: pd.DataFrame) -> pd.DataFrame:
        """Deliberately leaky: replicates the buggy workspace code."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["lag_1"] = df["sales"].shift(1)
        df["lag_7"] = df["sales"].shift(7)
        df["lag_14"] = df["sales"].shift(14)
        # THE BUG — no shift before rolling
        df["rolling_mean_7d"] = df["sales"].rolling(window=7).mean()
        df["rolling_std_7d"] = df["sales"].rolling(window=7).std()
        df["rolling_mean_14d"] = df["sales"].rolling(window=14).mean()
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_month"] = df["date"].dt.day
        return df.dropna().reset_index(drop=True)

    base = generate_sales_data(start="2023-01-01", end="2024-06-30", seed=42)
    base = base.sort_values("date").reset_index(drop=True)

    n = len(base)
    pivot = n // 2

    featured_clean = _leaky_build_features(base)

    perturbed = base.copy()
    perturbed.loc[pivot:, "sales"] += 9999
    featured_perturbed = _leaky_build_features(perturbed)

    cutoff = base["date"].iloc[pivot]
    orig_sub = featured_clean[featured_clean["date"].astype(str) <= str(cutoff)]
    pert_sub = featured_perturbed[featured_perturbed["date"].astype(str) <= str(cutoff)]

    merged = orig_sub.merge(pert_sub, on="date", suffixes=("_o", "_p"))

    leak_detected = False
    for col in ["rolling_mean_7d", "rolling_std_7d", "rolling_mean_14d"]:
        co, cp = f"{col}_o", f"{col}_p"
        if co in merged.columns:
            max_diff = (merged[co] - merged[cp]).abs().max()
            if max_diff > 1e-6:
                leak_detected = True
                break

    assert leak_detected, (
        "NC-3: The leaky pipeline was not detected by the perturbation test. "
        "The temporal integrity check is not strong enough."
    )
