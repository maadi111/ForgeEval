"""Inference script — loads a trained model and writes predictions to CSV."""

import argparse
import pathlib
import pickle
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from features import FEATURE_COLS, build_features


def predict(model_path: str, data_path: str, output_path: str) -> None:
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    df = pd.read_csv(data_path)
    featured = build_features(df)

    preds = model.predict(featured[FEATURE_COLS])
    out = pd.DataFrame({"date": featured["date"].astype(str), "prediction": preds.round(2)})
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--data", default="data/test.csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict(args.model, args.data, args.output)
