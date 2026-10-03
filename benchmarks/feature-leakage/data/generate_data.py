"""
Synthetic daily sales data generator for the Feature Leakage benchmark.

Generates a realistic demand time-series with:
- Linear trend
- Annual seasonality
- Weekly seasonality
- Gaussian noise

Used by both the workspace (visible data) and the grader (hidden holdout).
"""

import argparse
import pathlib

import numpy as np
import pandas as pd


def generate_sales_data(
    start: str = "2023-01-01",
    end: str = "2024-06-30",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with columns ['date', 'sales'].

    Parameters
    ----------
    start : str
        First date (inclusive), ISO format.
    end : str
        Last date (inclusive), ISO format.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    trend = np.linspace(100, 160, n)
    annual = 25 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    weekly = 12 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = rng.normal(0, 6, n)

    sales = (trend + annual + weekly + noise).clip(min=10).round(2)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": sales})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sales data")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="train.csv")
    args = parser.parse_args()

    df = generate_sales_data(args.start, args.end, args.seed)
    out = pathlib.Path(args.output)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows -> {out}")
