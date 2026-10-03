"""
Hidden test — Generalization to unseen timestamps.

Why these tests exist
---------------------
Candidates might unknowingly hard-code assumptions about the visible date
range (2023-01-01 – 2024-06-30).  The pipeline must also work on the holdout
period (2024-07-01 – 2024-12-31) and on the adversarial period
(2025-01-01 – 2025-06-30) that uses a different random seed.

A model trained on visible data should achieve MAE < 20 on the holdout —
a threshold that the original leaky model fails on the adversarial set
because its inflated training accuracy was false.
"""

import pathlib
import pickle
import sys

import numpy as np
import pytest

WORKSPACE = pathlib.Path(__file__).parents[2] / "workspace"
DATA_DIR = pathlib.Path(__file__).parents[2] / "data"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(DATA_DIR))

from features import FEATURE_COLS, TARGET_COL, build_features
from generate_data import generate_sales_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def holdout_df():
    """Visible holdout: same seed as training data, next date window."""
    return generate_sales_data(start="2024-07-01", end="2024-12-31", seed=42)


@pytest.fixture(scope="module")
def adversarial_df():
    """Adversarial holdout: different seed, future dates, never seen by candidate."""
    return generate_sales_data(start="2025-01-01", end="2025-06-30", seed=99)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_features_work_on_holdout_timestamps(holdout_df):
    """Feature pipeline must not error on dates beyond the training range."""
    try:
        featured = build_features(holdout_df)
    except Exception as exc:
        pytest.fail(f"build_features raised on holdout timestamps: {exc}")

    assert len(featured) > 0, "No rows produced from holdout data"
    missing = [c for c in FEATURE_COLS if c not in featured.columns]
    assert not missing, f"Missing columns on holdout: {missing}"


def test_features_work_on_adversarial_timestamps(adversarial_df):
    """Feature pipeline must not error on adversarial (seed=99) timestamps."""
    try:
        featured = build_features(adversarial_df)
    except Exception as exc:
        pytest.fail(f"build_features raised on adversarial timestamps: {exc}")

    assert len(featured) > 0


def test_model_accuracy_on_holdout(holdout_df):
    """
    A model trained on 2023-01-01 – 2024-06-30 must achieve MAE < 20 on
    the 2024-07-01 – 2024-12-31 holdout.

    The leaky pipeline achieves artificially low validation error on training
    data but generalises poorly; a fixed pipeline should meet this threshold.
    """
    model_path = WORKSPACE / "model.pkl"
    if not model_path.exists():
        pytest.skip("model.pkl not found — run train.py first")

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    featured = build_features(holdout_df)
    if featured.empty:
        pytest.skip("Holdout feature set is empty")

    preds = model.predict(featured[FEATURE_COLS])
    mae = np.abs(preds - featured[TARGET_COL].values).mean()

    assert mae < 20.0, (
        f"MAE on holdout = {mae:.2f} (threshold = 20.0). "
        "The model trained on a leaky pipeline typically fails this check "
        "because its training accuracy was inflated by future data."
    )


def test_no_hardcoded_date_assumptions(adversarial_df):
    """
    build_features must accept dates it has never encountered.
    Raises if calendar features error on unseen months/years.
    """
    featured = build_features(adversarial_df)
    # Verify calendar columns are within valid ranges
    assert featured["day_of_week"].between(0, 6).all()
    assert featured["month"].between(1, 12).all()
    assert featured["day_of_month"].between(1, 31).all()
