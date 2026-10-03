"""Synthetic document corpus and queries generator for B2 Retrieval Drift benchmark."""

import argparse
import json
import pathlib

HISTORICAL_DOCS = [
    {
        "id": "doc_cloud_01",
        "domain": "cloud",
        "title": "Virtual Machine Auto-scaling",
        "text": "Configure horizontal auto-scaling for virtual machine instances based on CPU utilization metrics and memory thresholds.",
    },
    {
        "id": "doc_cloud_02",
        "domain": "cloud",
        "title": "Block Storage Snapshots",
        "text": "Automate snapshot backups of persistent disk volumes with lifecycle policies and cross-region replication for disaster recovery.",
    },
    {
        "id": "doc_cloud_03",
        "domain": "cloud",
        "title": "VPC Network Peering",
        "text": "Connect multiple private cloud networks with internal IP routing, subnets, and security group firewall rules.",
    },
    {
        "id": "doc_cloud_04",
        "domain": "cloud",
        "title": "Object Storage Access Control",
        "text": "Configure fine-grained IAM roles, uniform bucket-level access, and public read prevention on cloud storage buckets.",
    },
    {
        "id": "doc_cloud_05",
        "domain": "cloud",
        "title": "Cloud Load Balancing Architecture",
        "text": "Deploy global HTTP and TCP load balancers with health checks, SSL termination, and CDN edge caching.",
    },
]

DRIFTED_DOCS = [
    {
        "id": "doc_mlops_01",
        "domain": "mlops",
        "title": "Kubernetes GPU Operator Deployment",
        "text": "Install nvidia gpu-operator helm chart on kubernetes clusters to enable automated cuda driver injection and dynamic gpu resource scheduling for inference pods.",
    },
    {
        "id": "doc_mlops_02",
        "domain": "mlops",
        "title": "Model Registry Versioning and Lineage",
        "text": "Track machine learning artifacts, model lineage, hyperparameter metadata, and production staging transitions using an enterprise model-registry.",
    },
    {
        "id": "doc_mlops_03",
        "domain": "mlops",
        "title": "vLLM Inference Engine Optimization",
        "text": "Deploy high-throughput vllm-engine with paged-attention, tensor parallelism, and continuous batching on multi-gpu nodes.",
    },
    {
        "id": "doc_mlops_04",
        "domain": "mlops",
        "title": "Feature Store Online Serving",
        "text": "Low-latency key-value retrieval of point-in-time features using redis and feast feature-store for real-time recommendation scoring.",
    },
    {
        "id": "doc_mlops_05",
        "domain": "mlops",
        "title": "Horizontal Pod Autoscaler with Prometheus Metrics",
        "text": "Configure hpa-autoscaling for model serving deployments based on custom prometheus inference request latency and queue depth metrics.",
    },
]

# Additional background docs to create realistic noise in retrieval corpus
BACKGROUND_DOCS = [
    {
        "id": f"doc_bg_{i:02d}",
        "domain": "general",
        "title": f"General IT Policy {i}",
        "text": f"Internal company standard operating procedure {i} regarding hardware maintenance, access provisioning, and system logging.",
    }
    for i in range(1, 31)
]

HISTORICAL_QUERIES = [
    {
        "query": "how to configure auto-scaling for virtual machine CPU",
        "relevant_ids": ["doc_cloud_01"],
    },
    {"query": "persistent disk snapshot backup policies", "relevant_ids": ["doc_cloud_02"]},
    {"query": "private VPC peering subnets firewall", "relevant_ids": ["doc_cloud_03"]},
    {"query": "cloud storage bucket IAM permissions and access", "relevant_ids": ["doc_cloud_04"]},
    {"query": "global HTTP load balancer with SSL health checks", "relevant_ids": ["doc_cloud_05"]},
]

DRIFTED_QUERIES = [
    {
        "query": "How do I configure the cluster for cuda driver using nvidia gpu-operator?",
        "relevant_ids": ["doc_mlops_01"],
    },
    {
        "query": "Where can we track lineage versions and model metadata in the model-registry?",
        "relevant_ids": ["doc_mlops_02"],
    },
    {
        "query": "What setup enables continuous batching on multi-gpu nodes in vllm-engine?",
        "relevant_ids": ["doc_mlops_03"],
    },
    {
        "query": "How to perform real-time recommendation scoring with redis feature-store?",
        "relevant_ids": ["doc_mlops_04"],
    },
    {
        "query": "How do we scale model serving pods using prometheus queue depth in hpa-autoscaling?",
        "relevant_ids": ["doc_mlops_05"],
    },
]


def generate_benchmark_assets(data_dir: pathlib.Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    all_docs = HISTORICAL_DOCS + DRIFTED_DOCS + BACKGROUND_DOCS

    corpus_path = data_dir / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2)

    queries_path = data_dir / "queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "historical": HISTORICAL_QUERIES,
                "drifted": DRIFTED_QUERIES,
            },
            f,
            indent=2,
        )
    print(f"Generated {len(all_docs)} documents -> {corpus_path}")
    print(
        f"Generated {len(HISTORICAL_QUERIES)} historical + {len(DRIFTED_QUERIES)} drifted queries -> {queries_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()
    generate_benchmark_assets(pathlib.Path(args.output_dir))
