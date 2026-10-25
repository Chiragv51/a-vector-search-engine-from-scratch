"""
Generates publication-quality plots from benchmark results:
1. exact_scaling.png: Latency scaling of brute-force exact search
2. recall_latency.png: Pareto frontier comparing Scratch HNSW vs FAISS
3. filter_collapse.png: Recall collapse curve of filtering strategies
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_exact_scaling():
    path = Path("benchmarks/results/exact.json")
    if not path.exists():
        print(f"Skipping exact scaling plot: {path} not found")
        return

    data = json.loads(path.read_text())["results"]
    df = pd.DataFrame(data)

    plt.figure(figsize=(7, 4.5), dpi=200)
    plt.plot(df["n"], df["p99_ms"], marker="o", color="#e63946", linewidth=2, label="P99 Latency (ms)")
    plt.plot(df["n"], df["p50_ms"], marker="s", color="#457b9d", linewidth=2, linestyle="--", label="P50 Latency (ms)")
    plt.axhline(20, color="#1d3557", linestyle=":", alpha=0.7, label="20ms Interactive Limit")

    plt.xscale("log")
    plt.xlabel("Indexed Vectors (N)", fontsize=11, fontweight="bold")
    plt.ylabel("Query Latency (ms)", fontsize=11, fontweight="bold")
    plt.title("Exact Brute-Force Vector Search Scaling (D=128)", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_file = Path("plots/exact_scaling.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")


def plot_faiss_comparison():
    path = Path("benchmarks/results/comparison.json")
    if not path.exists():
        print(f"Skipping FAISS comparison plot: {path} not found")
        return

    data = json.loads(path.read_text())["results"]
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5), dpi=200)

    colors = {
        "Scratch HNSW (Pure Python)": "#e63946",
        "FAISS IndexHNSWFlat (C++)": "#2a9d8f",
    }
    markers = {
        "Scratch HNSW (Pure Python)": "o",
        "FAISS IndexHNSWFlat (C++)": "s",
    }

    for engine, group in df.groupby("engine"):
        group = group.sort_values("recall_at_10")
        c = colors.get(engine, "#1d3557")
        m = markers.get(engine, "o")
        plt.plot(
            group["recall_at_10"],
            group["p99_ms"],
            marker=m,
            color=c,
            linewidth=2.2,
            label=f"{engine}",
        )
        for _, row in group.iterrows():
            plt.annotate(
                f"ef={int(row['ef_search'])}",
                (row["recall_at_10"], row["p99_ms"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=c,
            )

    plt.xlabel("Recall@10 vs Exact Ground Truth", fontsize=11, fontweight="bold")
    plt.ylabel("P99 Query Latency (ms)", fontsize=11, fontweight="bold")
    plt.title("HNSW Recall-Latency Frontier: Scratch vs FAISS (N=5,000, D=128)", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_file = Path("plots/recall_latency.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")


def plot_filtering_collapse():
    path = Path("benchmarks/results/filtering.json")
    if not path.exists():
        print(f"Skipping filtering plot: {path} not found")
        return

    data = json.loads(path.read_text())["results"]
    df = pd.DataFrame(data)

    plt.figure(figsize=(7.5, 4.8), dpi=200)
    plt.plot(df["actual_pct"], df["filter_aware_recall"], marker="o", color="#2a9d8f", linewidth=2.2, label="Filter-Aware Traversal")
    plt.plot(df["actual_pct"], df["overfetch_recall"], marker="^", color="#e76f51", linewidth=2.0, linestyle="--", label="Over-Fetch (10x)")
    plt.plot(df["actual_pct"], df["post_filter_recall"], marker="x", color="#e63946", linewidth=2.2, linestyle="-.", label="Naive Post-Filter")

    plt.xscale("log")
    plt.xlabel("Filter Selectivity (% of dataset allowed)", fontsize=11, fontweight="bold")
    plt.ylabel("Recall@5 vs Ground Truth", fontsize=11, fontweight="bold")
    plt.title("Metadata Filtering: The Post-Filter Collapse Curve", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_file = Path("plots/filter_collapse.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Generated {out_file}")


def main():
    set_plot_style()
    Path("plots").mkdir(parents=True, exist_ok=True)
    plot_exact_scaling()
    plot_faiss_comparison()
    plot_filtering_collapse()


if __name__ == "__main__":
    main()
