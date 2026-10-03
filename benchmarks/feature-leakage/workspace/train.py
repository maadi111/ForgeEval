"""Training script for the daily sales forecasting model."""

import argparse
import pathlib
import pickle
import sys

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Allow running from the workspace directory
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from features import FEATURE_COLS, TARGET_COL, build_features


def train(data_path: str, model_path: str) -> None:
    df = pd.read_csv(data_path)
    df = build_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X, y)

    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    print(f"Trained on {len(df)} rows | model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--output", default="model.pkl")
    args = parser.parse_args()
    train(args.data, args.output)
