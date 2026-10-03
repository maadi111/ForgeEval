"""Synthetic data and trained model generator for the Model Serving benchmark."""

import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = [
    "age",
    "income",
    "credit_score",
    "debt_to_income",
    "num_accounts",
    "recent_inquiries",
    "employment_years",
    "loan_amount",
]


def generate_tabular_data(n_samples: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, size=n_samples)
    income = np.round(rng.uniform(20000, 180000, size=n_samples), 2)
    credit_score = rng.integers(350, 850, size=n_samples)
    debt_to_income = np.round(rng.uniform(0.05, 0.85, size=n_samples), 4)
    num_accounts = rng.integers(1, 12, size=n_samples)
    recent_inquiries = rng.integers(0, 8, size=n_samples)
    employment_years = np.round(rng.uniform(0.5, 30.0, size=n_samples), 1)
    loan_amount = np.round(rng.uniform(2000, 45000, size=n_samples), 2)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "num_accounts": num_accounts,
            "recent_inquiries": recent_inquiries,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
        }
    )

    # Ground truth logistic signal
    z = (
        0.00003 * income
        + 0.008 * (credit_score - 600)
        - 2.5 * debt_to_income
        - 0.25 * recent_inquiries
        + 0.05 * employment_years
        - 0.00004 * loan_amount
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    labels = (rng.uniform(0, 1, size=n_samples) < prob).astype(int)

    return df, labels


def train_and_export_model(output_path: pathlib.Path) -> None:
    df, y = generate_tabular_data(n_samples=3000, seed=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(df[FEATURE_NAMES], y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": clf, "features": FEATURE_NAMES}, f)
    print(f"Exported trained model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="model.pkl")
    args = parser.parse_args()
    train_and_export_model(pathlib.Path(args.output))
