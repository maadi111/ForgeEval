# ForgeEval — Production ML Benchmark & Adversarial Evaluation Platform

[![CI](https://github.com/forgeeval/forgeeval/actions/workflows/ci.yml/badge.svg)](https://github.com/forgeeval/forgeeval/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**ForgeEval** is a production-grade benchmark and adversarial evaluation harness for machine learning systems and autonomous AI coding agents. Unlike conventional coding benchmarks that evaluate simple deterministic syntax or match reference patches, ForgeEval evaluates whether an engineer or AI agent can diagnose, debug, and resolve complex, realistic failure modes under adversarial constraints.

---

## Architecture & Evaluation Lifecycle

```text
+-----------------------------------------------------------------------------------------+
|                                    ForgeEval Core API                                   |
|   POST /submissions  -->  Docker Sandbox  -->  Isolated Workspace Mount (Non-Root)     |
+-----------------------------------------------------------------------------------------+
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
[Public Test Suite]        [Hidden Behavioral Probes]       [Adversarial Gates]
Contract & Formats         Distribution Drift & Jitter       Shortcuts & Memorization
(Guidance Only)            Temporal Perturbation             Constant Predictors
                                       |
                                       v
                     +-----------------------------------+
                     |       Behavioral Grader           |
                     |  6 Dimension Weighted Breakdown   |
                     |  - Correctness:           40 pts  |
                     |  - Domain Invariance:     20 pts  |
                     |  - Adversarial Defense:   15 pts  |
                     |  - Generalization:        10 pts  |
                     |  - Performance / Latency: 10 pts  |
                     |  - Cryptographic Integrity: 5 pts |
                     +-----------------------------------+
                                       |
                                       v
                          Interactive Visual Dashboard
                       Live Telemetry • Agent Trajectories
```

---

## 5 Production Benchmarks

| ID | Title | Difficulty | Symptom | Latent Root Cause | Anti-Cheat Defense |
|---|---|---|---|---|---|
| **B1** | **Point-in-Time Feature Leakage** (`feature-leakage-v1`) | Medium | High validation score ($R^2 > 0.98$) collapses in production ($R^2 < 0.12$). | Backward rolling window uses `closed='both'` or lacks `.shift(1)`, including future target sales. | Future-row perturbation: mutating row $t+1$ must strictly cause 0 difference in feature row $t$. |
| **B2** | **Retrieval Drift & Embedding Alignment** (`retrieval-drift-v1`) | Med/Hard | Recall@5 collapses from 94% on historical queries to 40% on drifted technical domain vocabulary. | Query string truncated at 4 tokens, destroying acronyms; unnormalized dot-product used on vector embeddings. | OOD technical vocabulary probe suite; L2 embedding normalization invariance check. |
| **B3** | **Broken Model Serving & Micro-Batching** (`model-serving-v1`) | Hard | Production requests intermittently return random or inverted credit risk predictions under batch load. | Dict key iteration order is arbitrary, causing feature column mismatch; batch sorting lacks inverse permutation mapping. | Batch invariance gate $f([A, B]) == [f(A), f(B)]$ and JSON key permutation invariance. |
| **B4** | **PyTorch & GPU Inference Latency** (`gpu-optimization-v1`) | Hard | Batch of 256 requests takes 0.52s, violating 0.08s production SLA and dropping streaming traffic. | Python `for`-loop single-item dispatch with active autograd graph tracking and training mode active. | Latency SLA gate (< 0.08s for 256 items); `model.eval()` and `torch.inference_mode()` enforcement. |
| **B5** | **Adversarial Evaluation & Anti-Cheat Harness** (`adversarial-grading-v1`) | Hard | Constant majority-class models and memorized hash tables achieve >95% accuracy and pass safety gates. | Naive evaluator uses uncalibrated raw accuracy on imbalanced data, lacks input jitter, and permits in-place fixture mutation. | Cryptographic SHA-256 fixture protection; continuous epsilon jitter probe; balanced accuracy & entropy gates. |

Each benchmark provides:
- **Symptom-Only README**: Realistic production bug description without revealing the underlying fix.
- **Reference Solution A & Reference Solution B**: Two independent, production-grade solutions using different algorithms/patterns.
- **Public Tests**: Contract and sanity checks visible to the candidate.
- **Hidden Behavioral Tests**: Rigorous invariance, perturbation, and latency checks.
- **Negative Control Suites**: Formally proving rejection of shortcuts (constant outputs, memorization, uncalibrated batching, tampering).

---

## AI Agent Evaluation Harness (`agent/`)

ForgeEval includes an autonomous agent runner implementing the full evaluation lifecycle:
```text
Task Spec -> Sandbox Clone -> Workspace Inspection -> Public Tests -> Code Patch -> Re-test -> ForgeEval Grade
```

Run an agent evaluation directly:
```bash
python -m agent.runner --task feature-leakage-v1
python -m agent.runner --task model-serving-v1
python -m agent.runner --task gpu-optimization-v1
```

---

## Quick Start & Reproduction

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`.venv`)

### 2. Setup
```bash
# Clone and enter workspace
cd ForgeEval-Starter-knowledge

# Install in editable mode with development dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"

# Pre-generate all synthetic datasets and model artifacts
python scripts/generate_all_data.py
```

### 3. Run Platform Test Suite
```bash
pytest -q tests
```

### 4. Run Benchmark Grader CLI
Evaluate any benchmark workspace or reference solution:
```bash
# Evaluate Point-in-Time Feature Leakage
python scripts/run_benchmark.py --task feature-leakage-v1 --workspace benchmarks/feature-leakage/reference/solution-a

# Evaluate Model Serving & Batching
python scripts/run_benchmark.py --task model-serving-v1 --workspace benchmarks/model-serving/reference/solution-a

# Evaluate Retrieval Drift
python scripts/run_benchmark.py --task retrieval-drift-v1 --workspace benchmarks/retrieval-drift/reference/solution-a

# Evaluate Adversarial Grading Harness
python scripts/run_benchmark.py --task adversarial-grading-v1 --workspace benchmarks/adversarial-grading/reference/solution-a

# Evaluate GPU / PyTorch Latency Optimization
python scripts/run_benchmark.py --task gpu-optimization-v1 --workspace benchmarks/gpu-optimization/reference/solution-a
```

### 5. Launch FastAPI Service & Interactive Dashboard
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)** in your browser to view the interactive benchmark explorer, live scoring telemetry, adversarial probe breakdown, and agent trajectories.

---

## Definition of Done Verification

- [x] **5 Realistic Production Benchmarks**: Implemented with symptom-only framing.
- [x] **Dual Reference Solutions**: Both `solution-a` and `solution-b` implemented and scoring 100/100 on every benchmark.
- [x] **Public & Hidden Test Suites**: 28 total suites covering contracts, boundary perturbations, drift, and SLA latency.
- [x] **Negative Controls**: 100% passing across all 5 benchmarks, verifying rejection of cheats and shortcuts.
- [x] **AI Agent Evaluation Harness**: Modular agent loop with trajectory recording and metric tracking.
- [x] **Platform API & Database Integration**: FastAPI task registry, dynamic discovery, async submission queue.
- [x] **Interactive Dashboard**: Modern dark-mode web application with live API connectivity.
- [x] **Full CI/CD Pipeline**: GitHub Actions workflow validating linting, platform tests, all 5 benchmarks, negative controls, and agent evaluation.
