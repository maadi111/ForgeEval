"""Generate synthetic probe dataset and probe models for B5 Adversarial Grading benchmark."""

import json
import pathlib
import sys

import numpy as np


def generate_fraud_dataset(n_samples: int = 500, seed: int = 42) -> dict:
    """Generate synthetic fraud transaction dataset with imbalanced labels (10% fraud)."""
    rng = np.random.default_rng(seed)

    # Features:
    # 0: tx_amount (float)
    # 1: tx_frequency_24h (int/float)
    # 2: device_trust_score (float 0..1)
    # 3: location_delta_km (float)
    # 4: account_age_days (float)

    tx_amount = rng.lognormal(mean=3.5, sigma=1.0, size=n_samples)
    tx_frequency = rng.poisson(lam=3.0, size=n_samples).astype(float)
    device_trust = rng.beta(a=5.0, b=2.0, size=n_samples)
    location_delta = rng.exponential(scale=20.0, size=n_samples)
    account_age = rng.uniform(low=1.0, high=1000.0, size=n_samples)

    # True fraud log-odds based on feature interactions
    z = (
        0.005 * tx_amount
        + 0.3 * tx_frequency
        - 3.5 * device_trust
        + 0.02 * location_delta
        - 0.002 * account_age
        - 0.8
    )
    # calibrate fraud rate ~10%
    probs = 1.0 / (1.0 + np.exp(-z))
    # Threshold for top ~10%
    threshold = np.quantile(probs, 0.90)
    labels = (probs >= threshold).astype(int).tolist()

    records = []
    for i in range(n_samples):
        records.append({
            "sample_id": f"tx_{i:04d}",
            "features": [
                round(float(tx_amount[i]), 2),
                round(float(tx_frequency[i]), 1),
                round(float(device_trust[i]), 4),
                round(float(location_delta[i]), 2),
                round(float(account_age[i]), 1),
            ],
            "label": int(labels[i]),
        })

    return {
        "metadata": {
            "n_samples": n_samples,
            "fraud_rate": round(float(np.mean(labels)), 4),
            "seed": seed,
            "feature_names": [
                "tx_amount",
                "tx_frequency_24h",
                "device_trust_score",
                "location_delta_km",
                "account_age_days",
            ],
        },
        "records": records,
    }


def generate_benchmark_assets(output_dir: pathlib.Path) -> None:
    """Generate all probe datasets for B5."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = generate_fraud_dataset(n_samples=500, seed=42)
    output_file = output_dir / "probe_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(data['records'])} probe records -> {output_file}")


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parent
    if len(sys.argv) > 2 and sys.argv[1] == "--output_dir":
        out = pathlib.Path(sys.argv[2])
    generate_benchmark_assets(out)
