/**
 * ForgeEval — Production ML Benchmark & Adversarial Evaluation Platform
 * Complete telemetry, benchmarks, runs, agents, artifacts, environments, and health datasets.
 */
window.FORGEEVAL_DATA = {
  platform: {
    name: "ForgeEval",
    version: "1.4.2",
    release: "production",
    cluster: "prod-us-central1 (Docker-GPU)",
    uptime: "99.98%",
    active_benchmarks: 5,
    evaluation_runs: 148,
    pass_rate: "68.4%",
    grader_reliability: "99.8%",
    avg_runtime: "42s",
    failed_runs: 47,
  },

  benchmarks: [
    {
      id: "FE-PTL-001",
      slug: "feature-leakage-v1",
      name: "Point-in-Time Feature Leakage",
      category: "Feature Engineering",
      difficulty: "Hard",
      version: "v1.4.2",
      status: "Active",
      framework: "PyTorch / Pandas",
      runtime_sla: "15s",
      avg_runtime: "6.8s",
      pass_rate: "41.2%",
      owner: "m.vasquez@forgeeval.internal",
      created_at: "2026-08-12",
      last_run: "14m ago",
      tags: ["Temporal Leakage", "Time Series", "Lookahead Bias", "Financial Risk"],
      description: "Repair a high-frequency financial risk forecasting feature pipeline whose validation metrics are artificially inflated (R² > 0.98) because rolling statistics calculate aggregations inclusive of prediction-time future timestamps.",
      expected_behavior: "All rolling window aggregations must be strictly computed using data available at or before cutoff timestamp t - 1. Predictions must be invariant to perturbation of future time steps.",
      known_symptoms: "Validation loss collapses near zero in backtests, but live paper trading experiences severe drawdown. Adding random noise to t+1 alters feature vector at time t.",
      estimated_solve_time: "45 mins",
      required_skills: ["Temporal Window Partitioning", "Pandas Rolling Semantics", "Lookahead Offset", "Data Leakage Diagnostics"],
      environment_reqs: { cpu: "4 vCPU", memory: "8 GB", gpu: "None", docker: "forgeeval/runtime-pandas:3.12" },
      dataset_info: { name: "Synthetic Market Events 2024-2025", size: "128 MB", rows: "2,400,000", format: "Parquet", split: "Chronological 70/15/15" },
      version_history: [
        { version: "v1.4.2", date: "2026-09-18", notes: "Tightened adversarial future-row perturbation probe threshold." },
        { version: "v1.3.0", date: "2026-08-30", notes: "Added Alternative Solution B (closed='left') verification." }
      ],
      weights: { correctness: 40, generalization: 20, regression: 15, adversarial: 15, performance: 10 },
      test_summary: { public: 2, hidden: 18, negative_controls: 3, regression: 4, performance: 1 },
      buggy_score: 85.0,
      buggy_status: "FAILED [X]",
      buggy_failures: ["[adversarial] Future event perturbation altered feature vector at timestamp t (diff=1428.43)"],
      sol_a: {
        name: "Canonical Shift Offset (Solution A)",
        approach: "Applied explicit .shift(1) to all backward rolling windows prior to computing rolling aggregates.",
        commit: "7d31ac2",
        author: "alex.chen@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "7.15s",
        memory: "384 MB"
      },
      sol_b: {
        name: "Strict Left-Closed Window (Solution B)",
        approach: "Configured rolling window with closed='left' parameter, eliminating prediction-time timestamp inclusion.",
        commit: "9f82d11",
        author: "sophia.lee@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "6.55s",
        memory: "380 MB"
      },
      files: [
        { name: "features.py", path: "src/features.py", type: "code", active: true },
        { name: "train.py", path: "src/train.py", type: "code" },
        { name: "pipeline.py", path: "src/pipeline.py", type: "code" },
        { name: "test_contract.py", path: "tests/public/test_contract.py", type: "test" },
        { name: "test_perturbation.py", path: "tests/hidden/test_perturbation.py", type: "test" },
        { name: "events.parquet", path: "data/events.parquet", type: "data" },
        { name: "Dockerfile", path: "Dockerfile", type: "config" },
        { name: "requirements.txt", path: "requirements.txt", type: "config" },
        { name: "README.md", path: "README.md", type: "doc" }
      ],
      tests: [
        { id: "PTL-P-001", name: "Feature Output Schema Contract", suite: "Public", type: "Contract", status: "PASS", runtime: "42ms" },
        { id: "PTL-P-002", name: "Non-Null Prediction Verification", suite: "Public", type: "Contract", status: "PASS", runtime: "38ms" },
        { id: "PTL-H-014", name: "Future Row Perturbation Invariance", suite: "Hidden", type: "Adversarial", status: "FAIL", runtime: "128ms", note: "Tripped on buggy workspace" },
        { id: "PTL-H-015", name: "Temporal Split Validation Boundary Isolation", suite: "Hidden", type: "Behavioral", status: "PASS", runtime: "94ms" },
        { id: "PTL-H-016", name: "Epsilon Jitter Timestamp Invariance", suite: "Hidden", type: "Robustness", status: "PASS", runtime: "112ms" },
        { id: "PTL-NC-001", name: "Negative Control: Cumulative Aggregation Leak", suite: "Negative Control", type: "Safety", status: "PASS", runtime: "88ms" },
        { id: "PTL-NC-002", name: "Negative Control: Direct Target Shifting", suite: "Negative Control", type: "Safety", status: "PASS", runtime: "82ms" },
        { id: "PTL-NC-003", name: "Negative Control: Centered Rolling Window", suite: "Negative Control", type: "Safety", status: "PASS", runtime: "91ms" },
        { id: "PTL-R-001", name: "Historical Sales Baseline Accuracy", suite: "Regression", type: "Regression", status: "PASS", runtime: "210ms" },
        { id: "PTL-SLA-001", name: "Feature Generation Latency SLA (<15s)", suite: "Performance", type: "Latency", status: "PASS", runtime: "6.8s" }
      ]
    },

    {
      id: "FE-SRV-003",
      slug: "model-serving-v1",
      name: "Broken Model Serving & Micro-Batching",
      category: "Model Serving",
      difficulty: "Hard",
      version: "v1.2.0",
      status: "Active",
      framework: "FastAPI / scikit-learn",
      runtime_sla: "20s",
      avg_runtime: "8.2s",
      pass_rate: "33.8%",
      owner: "k.tanaka@forgeeval.internal",
      created_at: "2026-08-20",
      last_run: "32m ago",
      tags: ["Micro-Batching", "Thread Safety", "JSON Serialization", "Permutation Leakage"],
      description: "Debug and repair a high-throughput micro-batching inference service where concurrent requests intermittently receive inverted or transposed credit risk scores.",
      expected_behavior: "Batch prediction f([A, B]) must strictly equal [f(A), f(B)]. Dict keys in JSON payloads must be deterministically mapped to model feature indices regardless of transmission ordering.",
      known_symptoms: "Under low concurrency single-item tests pass, but load testing with batched payloads yields a 14% inversion rate.",
      estimated_solve_time: "50 mins",
      required_skills: ["Micro-batch Dispatching", "Canonical Column Mapping", "Inverse Permutation Tracking", "FastAPI Concurrency"],
      environment_reqs: { cpu: "4 vCPU", memory: "8 GB", gpu: "None", docker: "forgeeval/runtime-serving:3.12" },
      dataset_info: { name: "Credit Applicant Evaluation Requests", size: "48 MB", rows: "500,000", format: "JSONL / Parquet", split: "Concurrent Streaming" },
      version_history: [
        { version: "v1.2.0", date: "2026-09-02", notes: "Added JSON key permutation adversarial probe." }
      ],
      weights: { correctness: 40, batch_invariance: 20, schema_robustness: 15, concurrency_safety: 15, performance: 10 },
      test_summary: { public: 3, hidden: 14, negative_controls: 3, regression: 3, performance: 2 },
      buggy_score: 65.0,
      buggy_status: "FAILED [X]",
      buggy_failures: ["[batch_invariance] Batch output diverged from single request evaluations (MAE=0.48)"],
      sol_a: {
        name: "Canonical Schema Alignment & Batch Preserving (Solution A)",
        approach: "Explicitly projected incoming dictionary keys against model.feature_names_in_ and maintained arrival index queue.",
        commit: "4e91b03",
        author: "alex.chen@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "8.12s",
        memory: "412 MB"
      },
      sol_b: {
        name: "Indexed Sequence Sorter with Permutation Inversion (Solution B)",
        approach: "Sorted batch by feature length with explicit permutation index tracking, inverting sort before response dispatch.",
        commit: "8c22d71",
        author: "d.kovacs@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "7.94s",
        memory: "410 MB"
      },
      files: [
        { name: "service.py", path: "src/service.py", type: "code", active: true },
        { name: "models.py", path: "src/models.py", type: "code" },
        { name: "batcher.py", path: "src/batcher.py", type: "code" },
        { name: "test_serving_contract.py", path: "tests/public/test_serving_contract.py", type: "test" },
        { name: "test_batch_invariance.py", path: "tests/hidden/test_batch_invariance.py", type: "test" },
        { name: "Dockerfile", path: "Dockerfile", type: "config" }
      ],
      tests: [
        { id: "SRV-P-001", name: "Single Request Contract Evaluation", suite: "Public", type: "Contract", status: "PASS", runtime: "28ms" },
        { id: "SRV-P-002", name: "Health & Readiness Endpoints", suite: "Public", type: "Contract", status: "PASS", runtime: "12ms" },
        { id: "SRV-H-007", name: "Batch Invariance f([A, B]) == [f(A), f(B)]", suite: "Hidden", type: "Behavioral", status: "FAIL", runtime: "145ms", note: "Tripped on buggy workspace" },
        { id: "SRV-H-008", name: "JSON Key Permutation Invariance", suite: "Hidden", type: "Robustness", status: "FAIL", runtime: "110ms" },
        { id: "SRV-H-009", name: "Concurrent Worker Thread Isolation", suite: "Hidden", type: "Concurrency", status: "PASS", runtime: "450ms" },
        { id: "SRV-NC-001", name: "Negative Control: Random Permutation Inverter", suite: "Negative Control", type: "Safety", status: "PASS", runtime: "95ms" }
      ]
    },

    {
      id: "FE-RET-002",
      slug: "retrieval-drift-v1",
      name: "Retrieval Drift & Embedding Alignment",
      category: "Vector Retrieval",
      difficulty: "Medium",
      version: "v1.1.0",
      status: "Active",
      framework: "Faiss / PyTorch",
      runtime_sla: "25s",
      avg_runtime: "11.4s",
      pass_rate: "54.2%",
      owner: "j.morales@forgeeval.internal",
      created_at: "2026-08-25",
      last_run: "1h ago",
      tags: ["Vector Search", "Distribution Drift", "L2 Normalization", "Semantic Search"],
      description: "Diagnose and fix a semantic retrieval system whose Recall@5 collapses from 94% on historical queries to 40% when exposed to domain vocabulary drift.",
      expected_behavior: "Vectors must be L2-normalized prior to dot product indexing to maintain cosine equivalence; query tokenization must preserve multi-word terminology without arbitrary truncations.",
      known_symptoms: "Retrieval accuracy degrades dramatically on long technical acronyms. Vector norm distributions diverge between query encoder and document store.",
      estimated_solve_time: "40 mins",
      required_skills: ["Vector Normalization", "Cosine Distance Math", "Tokenizer Truncation Limits", "Embedding Alignment"],
      environment_reqs: { cpu: "4 vCPU", memory: "8 GB", gpu: "None", docker: "forgeeval/runtime-nlp:3.11" },
      dataset_info: { name: "Technical Documentation Corpus v2", size: "64 MB", rows: "150,000 docs", format: "JSON / Vector Index", split: "Temporal Drift Benchmark" },
      version_history: [
        { version: "v1.1.0", date: "2026-09-08", notes: "Calibrated recall degradation threshold on domain technical queries." }
      ],
      weights: { correctness: 40, drift_robustness: 25, embedding_alignment: 15, performance: 15, integrity: 5 },
      test_summary: { public: 2, hidden: 12, negative_controls: 3, regression: 2, performance: 1 },
      buggy_score: 80.0,
      buggy_status: "FAILED [X]",
      buggy_failures: ["[drift_robustness] Recall@5 on drifted queries (0.40) fell below 0.70 threshold"],
      sol_a: {
        name: "L2 Normalization & Full Token Sequence (Solution A)",
        approach: "Applied F.normalize(dim=1) to both query and doc embeddings and raised tokenizer max_length to 128.",
        commit: "3b18c99",
        author: "j.morales@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "11.2s",
        memory: "512 MB"
      },
      sol_b: {
        name: "Cosine Distance Projection Layer (Solution B)",
        approach: "Inserted explicit unit-sphere projection layer in Faiss index constructor, enforcing cosine invariant distances.",
        commit: "6f51a24",
        author: "alex.chen@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "10.8s",
        memory: "508 MB"
      },
      files: [
        { name: "retriever.py", path: "src/retriever.py", type: "code", active: true },
        { name: "index.py", path: "src/index.py", type: "code" },
        { name: "test_retrieval_contract.py", path: "tests/public/test_retrieval_contract.py", type: "test" },
        { name: "test_drift_robustness.py", path: "tests/hidden/test_drift_robustness.py", type: "test" }
      ],
      tests: [
        { id: "RET-P-001", name: "In-Distribution Historical Recall@5", suite: "Public", type: "Accuracy", status: "PASS", runtime: "65ms" },
        { id: "RET-H-003", name: "Drifted Acronym Vocabulary Recall@5", suite: "Hidden", type: "Robustness", status: "FAIL", runtime: "210ms" },
        { id: "RET-H-004", name: "Vector Unit Norm Validation", suite: "Hidden", type: "Integrity", status: "FAIL", runtime: "45ms" },
        { id: "RET-NC-001", name: "Negative Control: Truncated Substring Matcher", suite: "Negative Control", type: "Safety", status: "PASS", runtime: "78ms" }
      ]
    },

    {
      id: "FE-GPU-004",
      slug: "gpu-optimization-v1",
      name: "PyTorch & GPU Inference Latency Optimization",
      category: "GPU Optimization",
      difficulty: "Hard",
      version: "v1.0.0",
      status: "Active",
      framework: "PyTorch 2.6 / CUDA",
      runtime_sla: "10s",
      avg_runtime: "3.2s",
      pass_rate: "48.5%",
      owner: "s.nair@forgeeval.internal",
      created_at: "2026-09-01",
      last_run: "2h ago",
      tags: ["PyTorch", "CUDA", "Tensor Vectorization", "Latency SLA", "autograd"],
      description: "Optimize a production PyTorch neural inference pipeline suffering from severe latency degradation (0.52s per 256 items vs 0.08s SLA) causing dropped streaming transactions.",
      expected_behavior: "Inference must be fully vectorized into contiguous tensor batches under torch.inference_mode() with model.eval(), executing in under 0.08s for 256 samples with identical predictions.",
      known_symptoms: "GPU memory profile shows continuous growth during inference. Per-sample Python for-loop creates autograd computation graphs for evaluation requests.",
      estimated_solve_time: "35 mins",
      required_skills: ["PyTorch Tensor Batching", "torch.inference_mode()", "Deterministic Eval Mode", "TorchScript JIT"],
      environment_reqs: { cpu: "8 vCPU", memory: "16 GB", gpu: "1x NVIDIA T4 (16GB)", docker: "forgeeval/runtime-pytorch:2.6-cuda12" },
      dataset_info: { name: "Synthetic Residual Risk Evaluation Tensors", size: "32 MB", rows: "100,000 vectors", format: "PyTorch .pt / HDF5", split: "Batch Latency Suite" },
      version_history: [
        { version: "v1.0.0", date: "2026-09-15", notes: "Initial release with 0.08s SLA threshold." }
      ],
      weights: { correctness: 40, throughput: 25, batch_invariance: 15, memory_efficiency: 15, integrity: 5 },
      test_summary: { public: 2, hidden: 10, negative_controls: 4, regression: 2, performance: 2 },
      buggy_score: 60.0,
      buggy_status: "FAILED [X]",
      buggy_failures: ["[throughput] Latency SLA breached: 0.520s for 256 items (threshold < 0.08s)"],
      sol_a: {
        name: "Vectorized Batch Tensor with inference_mode (Solution A)",
        approach: "Converted input items to stacked torch.tensor, placed model in eval mode, and ran inside torch.inference_mode(). Latency dropped to 0.005s (96x faster).",
        commit: "1a89c02",
        author: "s.nair@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "0.005s",
        memory: "180 MB"
      },
      sol_b: {
        name: "TorchScript JIT Tracing & Micro-Chunking (Solution B)",
        approach: "Compiled neural architecture via torch.jit.trace with fused batch kernels and static memory allocation.",
        commit: "5e42b10",
        author: "alex.chen@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "0.006s",
        memory: "172 MB"
      },
      files: [
        { name: "pipeline.py", path: "src/pipeline.py", type: "code", active: true },
        { name: "model.py", path: "src/model.py", type: "code" },
        { name: "test_gpu_contract.py", path: "tests/public/test_gpu_contract.py", type: "test" },
        { name: "test_latency_sla.py", path: "tests/hidden/test_latency_sla.py", type: "test" }
      ],
      tests: [
        { id: "GPU-P-001", name: "Contract Input-Output Dimensions", suite: "Public", type: "Contract", status: "PASS", runtime: "18ms" },
        { id: "GPU-H-001", name: "Throughput Latency SLA (<0.08s / 256 items)", suite: "Hidden", type: "Performance", status: "FAIL", runtime: "520ms" },
        { id: "GPU-H-002", name: "Deterministic Eval Mode (Dropout Zeroed)", suite: "Hidden", type: "Integrity", status: "FAIL", runtime: "42ms" },
        { id: "GPU-H-003", name: "Numerical Prediction Precision (atol=1e-4)", suite: "Hidden", type: "Accuracy", status: "PASS", runtime: "36ms" },
        { id: "GPU-NC-001", name: "Negative Control: Fixed Prediction Cache Exploit", suite: "Negative Control", type: "Safety", status: "PASS", runtime: "52ms" }
      ]
    },

    {
      id: "FE-ADV-005",
      slug: "adversarial-grading-v1",
      name: "Adversarial Evaluation & Anti-Cheat Grading Harness",
      category: "Security & Evaluation",
      difficulty: "Hard",
      version: "v1.3.0",
      status: "Active",
      framework: "Custom Sandbox / Pytest",
      runtime_sla: "30s",
      avg_runtime: "12.8s",
      pass_rate: "28.4%",
      owner: "alex.chen@forgeeval.internal",
      created_at: "2026-09-10",
      last_run: "3h ago",
      tags: ["Anti-Cheat", "Adversarial Probes", "Fixture Tampering", "Balanced Accuracy", "Epsilon Jitter"],
      description: "Harden an ML evaluation pipeline against adversarial shortcuts, in-place memory fixture tampering, memorization lookup tables, and uncalibrated metric exploitation.",
      expected_behavior: "Grader must defensively deepcopy fixtures, verify SHA-256 fixture checksums, compute balanced accuracy on imbalanced distributions, and perturb continuous features with epsilon jitter to reject lookup tables.",
      known_symptoms: "A trivial dummy model predicting constant 0 scores 90% uncalibrated accuracy. A malicious model modifying dataset in-memory passes all safety assertions.",
      estimated_solve_time: "60 mins",
      required_skills: ["Defensive Copying", "Cryptographic Fixture Hashing", "Balanced Accuracy Calculation", "Continuous Jitter Invariance"],
      environment_reqs: { cpu: "4 vCPU", memory: "8 GB", gpu: "None", docker: "forgeeval/runtime-security:3.12" },
      dataset_info: { name: "Adversarial Model Verification Fixtures", size: "24 MB", rows: "50,000 cases", format: "Pickle / Encrypted Binary", split: "Adversarial Probe Suite" },
      version_history: [
        { version: "v1.3.0", date: "2026-09-22", notes: "Upgraded continuous jitter magnitude to defeat rounded floating-point hash lookups." }
      ],
      weights: { correctness: 40, anti_tamper: 20, shortcut_rejection: 20, performance: 10, integrity: 10 },
      test_summary: { public: 2, hidden: 16, negative_controls: 8, regression: 2, performance: 1 },
      buggy_score: 60.0,
      buggy_status: "FAILED [X]",
      buggy_failures: [
        "[adversarial] Evaluator passed a memorized lookup shortcut (must reject)",
        "[adversarial] Evaluator passed a constant majority-class predictor (must reject)",
        "[adversarial] In-place fixture tampering altered ground-truth labels"
      ],
      sol_a: {
        name: "Cryptographic Fixture Guard & Balanced Metric (Solution A)",
        approach: "Implemented defensive deepcopy, pre/post evaluation SHA-256 state hashing, balanced accuracy metric, and 0.02 continuous epsilon jitter.",
        commit: "2c77a90",
        author: "alex.chen@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "12.6s",
        memory: "295 MB"
      },
      sol_b: {
        name: "Immutable Tuple Projections & Distribution Entropy (Solution B)",
        approach: "Transformed fixtures into immutable frozen tuples, calculated prediction distribution entropy to catch degenerate constant models, and enforced cryptographic digest logging.",
        commit: "7e91044",
        author: "elena.rostova@forgeeval.internal",
        score: 100.0,
        status: "PASSED [OK]",
        runtime: "12.2s",
        memory: "290 MB"
      },
      files: [
        { name: "evaluator.py", path: "src/evaluator.py", type: "code", active: true },
        { name: "security.py", path: "src/security.py", type: "code" },
        { name: "models.py", path: "src/models.py", type: "code" },
        { name: "test_evaluator_contract.py", path: "tests/public/test_evaluator_contract.py", type: "test" },
        { name: "test_anti_cheat.py", path: "tests/hidden/test_anti_cheat.py", type: "test" }
      ],
      tests: [
        { id: "ADV-P-001", name: "Evaluator Contract Protocol", suite: "Public", type: "Contract", status: "PASS", runtime: "22ms" },
        { id: "ADV-H-001", name: "In-Place Dataset Tamper Detection", suite: "Hidden", type: "Security", status: "FAIL", runtime: "140ms" },
        { id: "ADV-H-002", name: "Memorization Lookup Table Rejection", suite: "Hidden", type: "Adversarial", status: "FAIL", runtime: "98ms" },
        { id: "ADV-H-003", name: "Constant Predictor Exploit Rejection", suite: "Hidden", type: "Adversarial", status: "FAIL", runtime: "82ms" },
        { id: "ADV-H-004", name: "Permutation Equivalence Check", suite: "Hidden", type: "Robustness", status: "PASS", runtime: "115ms" },
        { id: "ADV-NC-001", name: "Negative Control: Tamper Bypass Probe", suite: "Negative Control", type: "Safety", status: "PASS", runtime: "65ms" }
      ]
    }
  ],

  runs: [
    {
      id: "run_8f2a91",
      benchmark_id: "FE-PTL-001",
      benchmark_name: "Point-in-Time Feature Leakage",
      agent: "agent-gpt-06",
      agent_type: "LLM Agent",
      commit: "7d31ac2",
      started_at: "2026-10-03 23:42:09",
      duration: "3m 42s",
      score: 92.0,
      status: "PASSED",
      environment: "Docker (4 vCPU, 8GB)",
      trigger: "Web API",
      stages: [
        { name: "Environment Initialized", time: "14:32:09", duration: "4s", status: "OK", detail: "Container spun up from forgeeval/runtime-pandas:3.12" },
        { name: "Dependencies Installed", time: "14:32:13", duration: "28s", status: "OK", detail: "Installed pinned dependencies from requirements.txt" },
        { name: "Dataset Mounted", time: "14:32:41", duration: "2s", status: "OK", detail: "Mounted /data/events.parquet (128 MB read-only)" },
        { name: "Public Tests", time: "14:32:43", duration: "19s", status: "OK", detail: "14/14 tests passed in pytest" },
        { name: "Hidden Tests", time: "14:33:02", duration: "43s", status: "OK", detail: "18/18 hidden behavioral tests passed" },
        { name: "Negative Controls", time: "14:33:45", duration: "38s", status: "OK", detail: "6/6 negative control broken mutants rejected" },
        { name: "Performance Tests", time: "14:34:23", duration: "29s", status: "OK", detail: "Latency SLA satisfied (6.8s < 15s limit)" },
        { name: "Grading & Cryptographic Integrity", time: "14:34:52", duration: "17s", status: "OK", detail: "Score: 92.0/100. SHA-256 fixture checksum verified" },
        { name: "Completed", time: "14:35:09", duration: "0s", status: "OK", detail: "Execution completed cleanly with exit code 0" }
      ],
      score_breakdown: {
        correctness: 40.0,
        generalization: 18.0,
        regression_safety: 14.0,
        performance: 9.0,
        integrity: 6.0,
        reproducibility: 5.0,
        total: 92.0
      },
      failure_analysis: null,
      diff: {
        files_modified: 1,
        additions: 4,
        deletions: 2,
        patch: `--- a/src/features.py
+++ b/src/features.py
@@ -34,7 +34,9 @@ def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
     df = df.sort_values("timestamp")
     # FIX: Apply lag shift to prevent lookahead bias
-    df["rolling_mean_7d"] = df["sales"].rolling(window="7D").mean()
-    df["rolling_std_7d"]  = df["sales"].rolling(window="7D").std()
+    shifted_sales = df["sales"].shift(1)
+    df["rolling_mean_7d"] = shifted_sales.rolling(window="7D").mean()
+    df["rolling_std_7d"]  = shifted_sales.rolling(window="7D").std()
     return df`
      },
      logs: [
        "[14:32:09] INFO [sandbox] Initializing isolated Docker container (ID: c8f912b4e7)",
        "[14:32:11] INFO [sandbox] Security profile applied: seccomp=default, cap-drop=ALL, no-new-privileges",
        "[14:32:13] INFO [deps] Verifying pinned dependencies against lockfile checksum sha256:4d8a1c...",
        "[14:32:41] INFO [storage] Ephemeral tmpfs mounted at /tmp (size: 2048 MB)",
        "[14:32:43] INFO [pytest] Executing public test suite: tests/public/test_contract.py",
        "[14:33:02] INFO [pytest] Public suite passed: 14 passed in 1.42s",
        "[14:33:05] INFO [probes] Executing behavioral hidden probes across 2,400,000 events...",
        "[14:33:41] INFO [probes] Future row perturbation probe PASSED (invariance preserved, diff=0.00)",
        "[14:33:45] INFO [negative-controls] Running 6 negative control mutant pipelines...",
        "[14:34:21] INFO [negative-controls] 6/6 mutants correctly rejected (zero false positive leakage)",
        "[14:34:25] INFO [perf] Measured feature aggregation runtime: 6.82s (SLA target < 15.0s) [PASS]",
        "[14:34:52] INFO [grader] Calculating behavioral weight matrix (Correctness: 40/40, Generalization: 18/20, Integrity: 6/10)...",
        "[14:35:09] INFO [system] Execution completed successfully. Final score: 92.0/100. Artifacts signed."
      ]
    },

    {
      id: "run_4c1e09",
      benchmark_id: "FE-PTL-001",
      benchmark_name: "Point-in-Time Feature Leakage",
      agent: "forge-heuristic-agent-v1",
      agent_type: "Local Agent",
      commit: "5a21ef4",
      started_at: "2026-10-03 22:15:30",
      duration: "2m 14s",
      score: 100.0,
      status: "PASSED",
      environment: "Docker (4 vCPU, 8GB)",
      trigger: "CLI Runner",
      stages: [
        { name: "Environment Initialized", time: "22:15:30", duration: "3s", status: "OK" },
        { name: "Public Tests", time: "22:15:33", duration: "12s", status: "OK" },
        { name: "Hidden Tests", time: "22:15:45", duration: "35s", status: "OK" },
        { name: "Negative Controls", time: "22:16:20", duration: "25s", status: "OK" },
        { name: "Completed", time: "22:17:44", duration: "0s", status: "OK" }
      ],
      score_breakdown: { correctness: 40.0, generalization: 20.0, regression_safety: 15.0, performance: 15.0, integrity: 10.0, total: 100.0 },
      failure_analysis: null,
      diff: {
        files_modified: 1,
        additions: 3,
        deletions: 1,
        patch: `--- a/src/features.py
+++ b/src/features.py
@@ -34,6 +34,8 @@ def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
-    df["rolling_mean_7d"] = df["sales"].rolling(window="7D").mean()
+    df["rolling_mean_7d"] = df["sales"].rolling(window="7D", closed="left").mean()`
      },
      logs: [
        "[22:15:30] INFO [agent] Autonomous loop started: Inspect -> Test -> Patch -> Re-test -> Grade",
        "[22:16:10] INFO [patch] Applied closed='left' rolling window parameter",
        "[22:17:44] INFO [grader] Full score awarded: 100.0/100.0"
      ]
    },

    {
      id: "run_3b19e2",
      benchmark_id: "FE-PTL-001",
      benchmark_name: "Point-in-Time Feature Leakage",
      agent: "agent-claude-35",
      agent_type: "LLM Agent",
      commit: "3a91bb8",
      started_at: "2026-10-03 21:05:11",
      duration: "3m 18s",
      score: 85.0,
      status: "FAILED",
      environment: "Docker (4 vCPU, 8GB)",
      trigger: "Scheduled Suite",
      stages: [
        { name: "Environment Initialized", time: "21:05:11", duration: "4s", status: "OK" },
        { name: "Public Tests", time: "21:05:15", duration: "16s", status: "OK" },
        { name: "Hidden Tests", time: "21:05:31", duration: "42s", status: "FAIL" },
        { name: "Grading", time: "21:06:13", duration: "10s", status: "FAIL" }
      ],
      score_breakdown: { correctness: 40.0, generalization: 15.0, regression_safety: 15.0, performance: 10.0, integrity: 5.0, total: 85.0 },
      failure_analysis: {
        summary: "Temporal leakage detected",
        failed_test: "PTL-H-014 (Future Row Perturbation Invariance)",
        expected: "Historical prediction must only use features available at prediction time (cutoff t <= T). Perturbing sales at t+1 must yield zero diff at t.",
        observed: "Altering sales value at t+1 altered rolling_mean_7d at t by 1428.43. Future information leaked across decision boundary.",
        evidence: "features.py:L35 computes rolling window without .shift(1) or closed='left', including day-of sales in predictor.",
        suggested_investigation: "Inspect window boundary alignment in src/features.py. Introduce an explicit lag offset or verify closed window parameter semantics."
      },
      diff: {
        files_modified: 1,
        additions: 2,
        deletions: 1,
        patch: `--- a/src/features.py
+++ b/src/features.py
@@ -34,5 +34,6 @@ def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
-    df["rolling_mean_7d"] = df["sales"].rolling(window="7D").mean()
+    # Candidate only clipped negative values, did not fix lookahead
+    df["rolling_mean_7d"] = df["sales"].clip(lower=0).rolling(window="7D").mean()`
      },
      logs: [
        "[21:05:11] INFO [sandbox] Initialized sandbox",
        "[21:05:31] WARN [probes] Tripped probe PTL-H-014: Future perturbation changed feature vector",
        "[21:06:13] ERROR [grader] Submission failed adversarial lookahead gate"
      ]
    },

    {
      id: "run_9d2f11",
      benchmark_id: "FE-GPU-004",
      benchmark_name: "PyTorch & GPU Inference Latency Optimization",
      agent: "agent-gpt-06",
      agent_type: "LLM Agent",
      commit: "9c81a20",
      started_at: "2026-10-03 20:30:00",
      duration: "1m 45s",
      score: 100.0,
      status: "PASSED",
      environment: "Docker-GPU (1x T4, 16GB)",
      trigger: "Web API",
      stages: [
        { name: "Environment Initialized", time: "20:30:00", duration: "5s", status: "OK" },
        { name: "GPU Diagnostics", time: "20:30:05", duration: "3s", status: "OK", detail: "CUDA 12.4, Device 0: NVIDIA T4 16GB verified" },
        { name: "Public Tests", time: "20:30:08", duration: "10s", status: "OK" },
        { name: "Latency SLA Benchmark", time: "20:30:18", duration: "25s", status: "OK", detail: "Measured: 0.005s for 256 items (target < 0.08s)" },
        { name: "Completed", time: "20:31:45", duration: "0s", status: "OK" }
      ],
      score_breakdown: { correctness: 40.0, throughput: 25.0, batch_invariance: 15.0, memory_efficiency: 15.0, integrity: 5.0, total: 100.0 },
      failure_analysis: null,
      diff: {
        files_modified: 1,
        additions: 8,
        deletions: 4,
        patch: `--- a/src/pipeline.py
+++ b/src/pipeline.py
@@ -18,4 +18,8 @@ def predict_batch(model, items):
-    preds = []
-    for item in items:
-        preds.append(model(torch.tensor(item).unsqueeze(0)))
-    return preds
+    model.eval()
+    tensor_batch = torch.as_tensor(items, dtype=torch.float32)
+    with torch.inference_mode():
+        outputs = model(tensor_batch)
+    return outputs.cpu().numpy()`
      },
      logs: [
        "[20:30:05] INFO [gpu] Device initialized: NVIDIA T4 (Compute Capability 7.5)",
        "[20:30:20] INFO [sla] Baseline: 0.520s -> Candidate: 0.005s (104x speedup) [PASS]",
        "[20:31:45] INFO [grader] Full score: 100.0/100.0"
      ]
    },

    {
      id: "run_1a4b88",
      benchmark_id: "FE-SRV-003",
      benchmark_name: "Broken Model Serving & Micro-Batching",
      agent: "eng-baseline-senior",
      agent_type: "Human Baseline",
      commit: "6d11f90",
      started_at: "2026-10-03 19:10:45",
      duration: "4m 12s",
      score: 100.0,
      status: "PASSED",
      environment: "Docker (4 vCPU, 8GB)",
      trigger: "Manual Submit",
      stages: [
        { name: "Environment Initialized", time: "19:10:45", duration: "3s", status: "OK" },
        { name: "Public Tests", time: "19:10:48", duration: "14s", status: "OK" },
        { name: "Hidden Batch Tests", time: "19:11:02", duration: "48s", status: "OK" },
        { name: "Completed", time: "19:14:57", duration: "0s", status: "OK" }
      ],
      score_breakdown: { correctness: 40.0, batch_invariance: 20.0, schema_robustness: 15.0, concurrency_safety: 15.0, performance: 10.0, total: 100.0 },
      failure_analysis: null,
      diff: null,
      logs: ["[19:10:45] INFO [human] Human engineer verified canonical schema ordering"]
    },

    {
      id: "run_5e71c0",
      benchmark_id: "FE-ADV-005",
      benchmark_name: "Adversarial Evaluation & Anti-Cheat Grading Harness",
      agent: "agent-gpt-06",
      agent_type: "LLM Agent",
      commit: "2b99a11",
      started_at: "2026-10-03 18:02:14",
      duration: "2m 50s",
      score: 60.0,
      status: "FAILED",
      environment: "Docker (4 vCPU, 8GB)",
      trigger: "API",
      stages: [
        { name: "Environment Initialized", time: "18:02:14", duration: "3s", status: "OK" },
        { name: "Adversarial Probe Suite", time: "18:02:17", duration: "55s", status: "FAIL" }
      ],
      score_breakdown: { correctness: 40.0, anti_tamper: 0.0, shortcut_rejection: 10.0, performance: 10.0, integrity: 0.0, total: 60.0 },
      failure_analysis: {
        summary: "In-Place Dataset Tampering & Memorization Exploit Accepted",
        failed_test: "ADV-H-001 (Fixture Mutability Detection)",
        expected: "Grader must defensively copy fixtures or calculate SHA-256 pre/post hashes to prevent candidate code modifying labels.",
        observed: "Candidate injected modified labels array into memory, causing grading assertions to evaluate against manipulated ground truth.",
        evidence: "evaluator.py did not implement copy.deepcopy() or cryptographic state validation.",
        suggested_investigation: "Enforce fixture immutability using cryptographic hashing or deepcopy prior to handing data to candidate model."
      },
      diff: null,
      logs: ["[18:02:17] ERROR [security] In-place fixture mutation was not caught by candidate evaluator"]
    }
  ],

  agents: [
    {
      id: "agent-gpt-06",
      name: "GPT Agent (gpt-4o-eval-06)",
      provider: "OpenAI",
      type: "LLM Agent",
      runs: 64,
      pass_rate: "78.1%",
      avg_score: "88.4",
      median_runtime: "3m 12s",
      failure_rate: "21.9%",
      benchmark_coverage: "5 / 5",
      status: "Active",
      scores_over_time: [74, 82, 85, 91, 88, 92, 95, 92],
      failures_by_category: { "Feature Leakage": 3, "Timeout": 2, "Schema Invariance": 5, "Adversarial Shortcuts": 4 }
    },
    {
      id: "agent-claude-35",
      name: "Claude Agent (claude-3.5-sonnet)",
      provider: "Anthropic",
      type: "LLM Agent",
      runs: 52,
      pass_rate: "82.7%",
      avg_score: "91.2",
      median_runtime: "2m 54s",
      failure_rate: "17.3%",
      benchmark_coverage: "5 / 5",
      status: "Active",
      scores_over_time: [80, 85, 89, 90, 94, 92, 96, 95],
      failures_by_category: { "Feature Leakage": 2, "Timeout": 1, "Schema Invariance": 3, "Adversarial Shortcuts": 3 }
    },
    {
      id: "forge-heuristic-agent-v1",
      name: "Forge Local Agent (v1.0)",
      provider: "ForgeEval Internal",
      type: "Rule/AST Agent",
      runs: 20,
      pass_rate: "90.0%",
      avg_score: "95.5",
      median_runtime: "1m 45s",
      failure_rate: "10.0%",
      benchmark_coverage: "5 / 5",
      status: "Active",
      scores_over_time: [90, 92, 95, 98, 95, 100, 100, 100],
      failures_by_category: { "Feature Leakage": 0, "Timeout": 0, "Schema Invariance": 1, "Adversarial Shortcuts": 1 }
    },
    {
      id: "eng-baseline-senior",
      name: "Human ML Engineer Baseline",
      provider: "Internal Team",
      type: "Human Engineer",
      runs: 12,
      pass_rate: "100.0%",
      avg_score: "99.2",
      median_runtime: "28m 30s",
      failure_rate: "0.0%",
      benchmark_coverage: "5 / 5",
      status: "Verified",
      scores_over_time: [98, 100, 100, 98, 100, 100],
      failures_by_category: { "Feature Leakage": 0, "Timeout": 0, "Schema Invariance": 0, "Adversarial Shortcuts": 0 }
    }
  ],

  artifacts: [
    { name: "docker-runtime-pandas-3.12.tar.gz", type: "Docker Image", version: "v1.4.2", hash: "sha256:4d8a1c90ef22b7a3", size: "482 MB", created: "2026-09-18", used_by: "FE-PTL-001" },
    { name: "dataset-market-events-2025.parquet", type: "Dataset Snapshot", version: "v2.1", hash: "sha256:7f4a0198bb4e112d", size: "128 MB", created: "2026-08-12", used_by: "FE-PTL-001" },
    { name: "docker-runtime-pytorch-cuda12.tar.gz", type: "Docker Image", version: "v2.6.0", hash: "sha256:88e1a409cb319d00", size: "1.84 GB", created: "2026-09-01", used_by: "FE-GPU-004" },
    { name: "fixture-adversarial-models.bin", type: "Test Bundle", version: "v1.3", hash: "sha256:3a77d0182ecbf811", size: "24 MB", created: "2026-09-10", used_by: "FE-ADV-005" },
    { name: "requirements-pinned-locked.txt", type: "Dependency Lock", version: "v1.0", hash: "sha256:119bc0429f98aa31", size: "4.2 KB", created: "2026-09-20", used_by: "All Tasks" },
    { name: "grader-behavioral-engine-v1.4.py", type: "Grader Executable", version: "v1.4.2", hash: "sha256:99f3028c1ab41029", size: "64 KB", created: "2026-09-15", used_by: "Global Platform" },
    { name: "reference-solution-ptl-a.bundle", type: "Reference Solution", version: "v1.4.2", hash: "sha256:ee8129038ba41829", size: "18 KB", created: "2026-08-14", used_by: "FE-PTL-001" },
    { name: "reference-solution-ptl-b.bundle", type: "Reference Solution", version: "v1.4.2", hash: "sha256:55ab901842cca910", size: "19 KB", created: "2026-08-30", used_by: "FE-PTL-001" }
  ],

  environments: [
    {
      id: "env-py312-cuda12",
      name: "Python 3.12 • PyTorch 2.6 • CUDA 12.4",
      os: "Ubuntu 24.04 LTS (Docker)",
      runtime: "Docker Container (Isolated non-root)",
      cpu: "8 vCPU",
      memory: "16 GB RAM",
      gpu: "1x NVIDIA T4 (16GB VRAM)",
      storage: "20 GB Ephemeral tmpfs",
      network: "Disabled (Isolated Airgap)",
      timeout: "180 seconds",
      reproducibility: "VERIFIED",
      used_by_tasks: ["FE-GPU-004", "FE-RET-002"],
      active_instances: 2
    },
    {
      id: "env-py312-ml-cpu",
      name: "Python 3.12 • Pandas • scikit-learn • LightGBM",
      os: "Ubuntu 24.04 LTS (Docker)",
      runtime: "Docker Container (Isolated non-root)",
      cpu: "4 vCPU",
      memory: "8 GB RAM",
      gpu: "None",
      storage: "10 GB Ephemeral tmpfs",
      network: "Disabled (Isolated Airgap)",
      timeout: "120 seconds",
      reproducibility: "VERIFIED",
      used_by_tasks: ["FE-PTL-001", "FE-SRV-003", "FE-ADV-005"],
      active_instances: 2
    },
    {
      id: "env-sandbox-hardened",
      name: "Hardened Minimal Sandbox (Seccomp Strict)",
      os: "Alpine Linux 3.20 (Minimal)",
      runtime: "gVisor / runsc",
      cpu: "2 vCPU",
      memory: "4 GB RAM",
      gpu: "None",
      storage: "2 GB tmpfs (read-only rootfs)",
      network: "Disabled (Zero Sockets)",
      timeout: "60 seconds",
      reproducibility: "VERIFIED",
      used_by_tasks: ["Adversarial Untrusted Code"],
      active_instances: 0
    }
  ],

  grader_health: {
    false_positive_rate: "0.00%",
    false_negative_rate: "0.00%",
    negative_control_pass_rate: "100.0%",
    reference_solution_pass_rate: "100.0%",
    alternative_solution_pass_rate: "100.0%",
    test_stability_index: "99.98%",
    flaky_tests_count: 0,
    total_evaluations: 148,
    active_negative_controls: [
      { id: "NC-PTL-001", target: "FE-PTL-001", name: "Cumulative Shift Leak Mutant", status: "REJECTED (Correct)", last_verified: "1h ago" },
      { id: "NC-SRV-001", target: "FE-SRV-003", name: "Arbitrary Key Order Dependency", status: "REJECTED (Correct)", last_verified: "1h ago" },
      { id: "NC-RET-001", target: "FE-RET-002", name: "Unnormalized Dot Product Projection", status: "REJECTED (Correct)", last_verified: "2h ago" },
      { id: "NC-GPU-001", target: "FE-GPU-004", name: "Fixed Lookup Array Bypass", status: "REJECTED (Correct)", last_verified: "2h ago" },
      { id: "NC-ADV-001", target: "FE-ADV-005", name: "In-Place Dataset Mutation Bypass", status: "REJECTED (Correct)", last_verified: "3h ago" }
    ]
  },

  system_health: [
    { service: "FastAPI Platform API", status: "Operational", latency: "18ms", uptime: "99.99%", details: "Uvicorn 0.54.0 (Python 3.12.10)" },
    { service: "Docker Execution Runner", status: "Operational", latency: "45ms", uptime: "99.98%", details: "Docker Engine 27.1 / Seccomp sandboxing active" },
    { service: "PostgreSQL Telemetry DB", status: "Operational", latency: "4.2ms", uptime: "99.99%", details: "Connection pool: 12/50 active" },
    { service: "Redis Task Queue", status: "Operational", latency: "1.2ms", uptime: "100.0%", details: "0 backlog / 4 worker threads listening" },
    { service: "Artifact Storage (S3 / MinIO)", status: "Operational", latency: "24ms", uptime: "99.99%", details: "Bucket size: 12.4 GB / 87 artifacts stored" },
    { service: "Benchmark Discovery Engine", status: "Operational", latency: "2ms", uptime: "100.0%", details: "5/5 benchmarks loaded with zero schema warnings" }
  ],

  analytics: {
    failure_modes: [
      { name: "Temporal Lookahead Leakage", count: 18, pct: "38.3%" },
      { name: "Batch / Permutation Inversion", count: 11, pct: "23.4%" },
      { name: "In-Place Fixture Modification", count: 7, pct: "14.9%" },
      { name: "Throughput / SLA Breach", count: 5, pct: "10.6%" },
      { name: "Vector Normalization Collapse", count: 4, pct: "8.5%" },
      { name: "Execution Timeout (>180s)", count: 2, pct: "4.3%" }
    ],
    difficulty_calibration: [
      { task: "FE-PTL-001", estimated_mins: 45, actual_agent_mins: 3.4, actual_human_mins: 38 },
      { task: "FE-SRV-003", estimated_mins: 50, actual_agent_mins: 4.1, actual_human_mins: 42 },
      { task: "FE-RET-002", estimated_mins: 40, actual_agent_mins: 2.8, actual_human_mins: 31 },
      { task: "FE-GPU-004", estimated_mins: 35, actual_agent_mins: 1.8, actual_human_mins: 24 },
      { task: "FE-ADV-005", estimated_mins: 60, actual_agent_mins: 4.9, actual_human_mins: 55 }
    ]
  },

  settings: {
    general: { org_name: "ML Engineering Platform Team", environment: "Production", default_region: "us-central1" },
    execution_limits: { default_cpu: "4 vCPU", max_cpu: "8 vCPU", default_memory: "8 GB", max_memory: "16 GB", default_timeout: "180s", allow_network: false },
    api_keys: [
      { id: "fe_live_98a1", name: "CI/CD GitHub Actions Key", prefix: "fe_live_98a1...", created: "2026-08-10", last_used: "14m ago", status: "Active" },
      { id: "fe_live_33b4", name: "Agent Runner Service Account", prefix: "fe_live_33b4...", created: "2026-08-22", last_used: "2h ago", status: "Active" }
    ],
    audit_logs: [
      { time: "2026-10-03 23:42:09", actor: "agent-gpt-06", action: "evaluation_run_created", target: "FE-PTL-001 (run_8f2a91)" },
      { time: "2026-10-03 22:15:30", actor: "alex.chen", action: "benchmark_updated", target: "FE-PTL-001 (v1.4.2)" },
      { time: "2026-10-03 20:30:00", actor: "ci-pipeline", action: "negative_control_verified", target: "All 5 Benchmarks" }
    ]
  }
};
