"""Pre-generate all synthetic benchmark datasets."""

import pathlib
import sys

# Add feature-leakage data directory to sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FL_DATA_DIR = REPO_ROOT / "benchmarks" / "feature-leakage" / "data"
sys.path.insert(0, str(FL_DATA_DIR))

from generate_data import generate_sales_data


def generate_feature_leakage_data() -> None:
    """Generate training and testing data for the feature-leakage benchmark."""
    train_path = FL_DATA_DIR / "train.csv"
    if not train_path.exists():
        print(f"Generating feature-leakage training data -> {train_path}")
        df = generate_sales_data(start="2023-01-01", end="2024-06-30", seed=42)
        df.to_csv(train_path, index=False)
        print(f"Wrote {len(df)} rows to {train_path}")
    else:
        print(f"Found existing training data: {train_path}")

    # Also place a copy in workspace/data if needed by workspace scripts
    ws_data_dir = REPO_ROOT / "benchmarks" / "feature-leakage" / "workspace" / "data"
    ws_data_dir.mkdir(parents=True, exist_ok=True)
    ws_train_path = ws_data_dir / "train.csv"
    if not ws_train_path.exists():
        df = generate_sales_data(start="2023-01-01", end="2024-06-30", seed=42)
        df.to_csv(ws_train_path, index=False)
        print(f"Wrote {len(df)} rows to workspace data: {ws_train_path}")


def generate_model_serving_data() -> None:
    """Generate trained model artifacts for the model-serving benchmark."""
    ms_dir = REPO_ROOT / "benchmarks" / "model-serving"
    ms_data_dir = ms_dir / "data"
    sys.path.insert(0, str(ms_data_dir))
    from generate_model import train_and_export_model

    model_path = ms_data_dir / "model.pkl"
    if not model_path.exists():
        print(f"Generating model-serving model -> {model_path}")
        train_and_export_model(model_path)
    else:
        print(f"Found existing model: {model_path}")

    # Copy to workspace and reference solutions
    for sub in ["workspace", "reference/solution-a", "reference/solution-b"]:
        target = ms_dir / sub / "model.pkl"
        if not target.exists():
            import shutil

            shutil.copyfile(model_path, target)
            print(f"Copied model to {target}")


def generate_retrieval_drift_data() -> None:
    """Generate corpus and query benchmarks for retrieval-drift."""
    rd_dir = REPO_ROOT / "benchmarks" / "retrieval-drift"
    rd_data_dir = rd_dir / "data"
    sys.path.insert(0, str(rd_data_dir))
    from generate_corpus import generate_benchmark_assets

    corpus_path = rd_data_dir / "corpus.json"
    if not corpus_path.exists():
        print(f"Generating retrieval-drift corpus -> {corpus_path}")
        generate_benchmark_assets(rd_data_dir)
    else:
        print(f"Found existing retrieval corpus: {corpus_path}")

    # Copy to workspace and reference solutions
    import shutil
    for sub in ["workspace/data", "reference/solution-a/data", "reference/solution-b/data"]:
        target_dir = rd_dir / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["corpus.json", "queries.json"]:
            src = rd_data_dir / fname
            dst = target_dir / fname
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)


def generate_adversarial_grading_data() -> None:
    """Generate probe dataset for adversarial-grading benchmark."""
    ag_dir = REPO_ROOT / "benchmarks" / "adversarial-grading"
    ag_data_dir = ag_dir / "data"
    sys.path.insert(0, str(ag_data_dir))
    from generate_probes import generate_benchmark_assets

    probe_path = ag_data_dir / "probe_dataset.json"
    if not probe_path.exists():
        print(f"Generating adversarial-grading probe dataset -> {probe_path}")
        generate_benchmark_assets(ag_data_dir)
    else:
        print(f"Found existing probe dataset: {probe_path}")

    # Copy to workspace and reference solutions
    import shutil
    for sub in ["workspace/data", "reference/solution-a/data", "reference/solution-b/data"]:
        target_dir = ag_dir / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        src = probe_path
        dst = target_dir / "probe_dataset.json"
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)


def generate_gpu_optimization_data() -> None:
    """Generate model weights and eval requests for gpu-optimization benchmark."""
    gpu_dir = REPO_ROOT / "benchmarks" / "gpu-optimization"
    gpu_data_dir = gpu_dir / "data"

    import importlib.util
    gen_script = gpu_data_dir / "generate_model.py"
    spec = importlib.util.spec_from_file_location("b4_generate_model", gen_script)
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        generate_benchmark_assets = mod.generate_benchmark_assets
    else:
        raise ImportError(f"Could not load generator from {gen_script}")

    weights_path = gpu_data_dir / "model_weights.pt"
    if not weights_path.exists():
        print(f"Generating gpu-optimization neural model & requests -> {weights_path}")
        generate_benchmark_assets(gpu_data_dir)
    else:
        print(f"Found existing gpu-optimization model weights: {weights_path}")

    # Copy to workspace and reference solutions
    import shutil
    for sub in ["workspace/data", "reference/solution-a/data", "reference/solution-b/data"]:
        target_dir = gpu_dir / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["model_weights.pt", "eval_requests.json", "ground_truth.json"]:
            src = gpu_data_dir / fname
            dst = target_dir / fname
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)


if __name__ == "__main__":
    generate_feature_leakage_data()
    generate_model_serving_data()
    generate_retrieval_drift_data()
    generate_adversarial_grading_data()
    generate_gpu_optimization_data()
