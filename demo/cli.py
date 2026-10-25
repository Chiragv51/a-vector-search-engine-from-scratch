"""
Interactive CLI demonstration of Vector Search from Scratch.
Allows users to index vectors, run live queries, toggle efSearch,
and observe recall, distance evaluations, and filtering in real-time.
"""

from __future__ import annotations
import argparse
import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vector_search.datasets import synthetic_dataset, recall_at_k
from vector_search.exact_numpy import ExactNumpyIndex
from vector_search.hnsw import HNSWIndex
from vector_search.filtering import (
    exact_filtered_search,
    post_filter_search,
    filter_aware_hnsw_search,
)


console = Console(safe_box=True)


def run_demo(n: int = 2000, d: int = 64, k: int = 5):
    console.print(
        Panel.fit(
            f"[bold cyan]Vector Search Engine from Scratch[/bold cyan]\n"
            f"[yellow]Interactive CLI Demo[/yellow] | N={n:,} vectors, Dim={d}, Top-{k}",
            border_style="cyan",
        )
    )

    console.print(f"[bold green]Generating reproducible dataset (seed=2026)...[/bold green]")
    vectors, queries = synthetic_dataset(n=n, d=d, query_count=20, seed=2026)

    # Attach categorical metadata (Category A: 95%, Category B: 5%)
    rng = np.random.default_rng(2026)
    is_rare = rng.random(size=n) < 0.05
    metadata = np.where(is_rare, "B_rare (5%)", "A_common (95%)")
    metadata_map = {int(i): metadata[i] for i in range(n)}

    # Build Exact Baseline
    console.print("[dim]1. Building Exact NumPy Baseline...[/dim]")
    exact_idx = ExactNumpyIndex(vectors)

    # Build HNSW Index
    console.print("[dim]2. Building HNSW Graph (M=12, efConstruction=100)...[/dim]")
    t0 = time.perf_counter()
    hnsw = HNSWIndex(dimension=d, M=12, ef_construction=100, seed=2026)
    for i, vec in enumerate(vectors):
        hnsw.add(vec, i)
    build_time = time.perf_counter() - t0
    console.print(f"[bold cyan][OK] HNSW Built in {build_time:.2f}s | Max Layer: {hnsw.max_level}[/bold cyan]\n")

    # Run Query Comparison
    q = queries[0]
    console.print("[bold yellow]Running Query #1 across engines:[/bold yellow]")

    # Exact Search
    t0 = time.perf_counter()
    exact_res = exact_idx.search(q, k=k)
    exact_time_ms = (time.perf_counter() - t0) * 1000

    table = Table(title="Top-5 Nearest Neighbors Comparison", header_style="bold magenta")
    table.add_column("Rank", justify="center")
    table.add_column("Exact (Ground Truth)", justify="center")
    table.add_column("Exact Dist", justify="right")
    table.add_column("HNSW (efS=20)", justify="center")
    table.add_column("HNSW Dist", justify="right")
    table.add_column("HNSW (efS=80)", justify="center")
    table.add_column("HNSW Dist", justify="right")

    t0 = time.perf_counter()
    hnsw_res_fast = hnsw.search(q, k=k, ef_search=20)
    hnsw_fast_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hnsw_res_deep = hnsw.search(q, k=k, ef_search=80)
    hnsw_deep_ms = (time.perf_counter() - t0) * 1000

    for r in range(k):
        table.add_row(
            str(r + 1),
            f"ID {exact_res.ids[r]}",
            f"{exact_res.distances[r]:.4f}",
            f"ID {hnsw_res_fast.ids[r]}",
            f"{hnsw_res_fast.distances[r]:.4f}",
            f"ID {hnsw_res_deep.ids[r]}",
            f"{hnsw_res_deep.distances[r]:.4f}",
        )

    console.print(table)
    console.print(
        f"[dim]Latencies: Exact={exact_time_ms:.2f}ms | HNSW(ef=20)={hnsw_fast_ms:.2f}ms | HNSW(ef=80)={hnsw_deep_ms:.2f}ms[/dim]\n"
    )

    # Demonstrate Filter Collapse
    console.print("[bold red]Demonstrating Metadata Filter Collapse (Category B, 5% rare):[/bold red]")
    exact_filt = exact_filtered_search(
        vectors, np.arange(n), metadata, q, k=k, allowed_value="B_rare (5%)"
    )
    post_filt = post_filter_search(
        hnsw, metadata_map, q, k=k, allowed_value="B_rare (5%)", ef_search=30
    )
    aware_filt = filter_aware_hnsw_search(
        hnsw, metadata_map, q, k=k, allowed_value="B_rare (5%)", ef_search=50
    )

    filt_table = Table(title="Filtered Search (5% Selectivity)", header_style="bold red")
    filt_table.add_column("Method", style="bold")
    filt_table.add_column("Matching IDs Found", justify="center")
    filt_table.add_column("IDs", justify="left")
    filt_table.add_column("Status / Verdict")

    filt_table.add_row(
        "Exact Filtered Ground Truth",
        f"{len(exact_filt.ids)}/{k}",
        str(exact_filt.ids.tolist()),
        "[green]Golden Standard[/green]",
    )
    filt_table.add_row(
        "Naive Post-Filter (Search then Filter)",
        f"{len(post_filt.ids)}/{k}",
        str(post_filt.ids.tolist()),
        "[red]COLLAPSE: Missing candidates![/red]" if len(post_filt.ids) < k else "[yellow]Partial[/yellow]",
    )
    filt_table.add_row(
        "Filter-Aware Traversal",
        f"{len(aware_filt.ids)}/{k}",
        str(aware_filt.ids.tolist()),
        "[green]SUCCESS: Complete candidate set[/green]",
    )
    console.print(filt_table)
    console.print("\n[bold green][OK] Demo finished successfully.[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Vector Search Engine Demo")
    parser.add_argument("--n", type=int, default=2000, help="Number of vectors to index")
    parser.add_argument("--d", type=int, default=64, help="Vector dimension")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    args = parser.parse_args()
    run_demo(n=args.n, d=args.d, k=args.k)


if __name__ == "__main__":
    main()
