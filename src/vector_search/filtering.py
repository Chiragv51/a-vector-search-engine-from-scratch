"""
Metadata filtering strategies for vector search:
1. Post-filtering (search top-k, discard non-matching) - collapses at high selectivity!
2. Over-fetching (search top-M*k, then filter)
3. Filter-aware graph traversal (non-matching route navigation, only matching enter result heap)
4. Exact filtered search (ground truth)
"""

from __future__ import annotations
from typing import Any
import heapq
import numpy as np
from .exact_loop import SearchResult
from .exact_numpy import ExactNumpyIndex
from .hnsw import HNSWIndex


def exact_filtered_search(
    vectors: np.ndarray,
    ids: np.ndarray,
    metadata: np.ndarray,
    query: np.ndarray,
    k: int,
    allowed_value: Any,
) -> SearchResult:
    """Exact ground truth for filtered search: mask out non-matching rows before KNN."""
    mask = metadata == allowed_value
    if mask.sum() < k:
        raise ValueError(
            f"fewer than k allowed items matching filter: found {mask.sum()}, requested {k}"
        )
    filtered = ExactNumpyIndex(vectors[mask], ids[mask])
    return filtered.search(query, k)


def post_filter_search(
    index: HNSWIndex,
    metadata_map: dict[int, Any],
    query: np.ndarray,
    k: int,
    allowed_value: Any,
    ef_search: int = 50,
) -> SearchResult:
    """Naive Post-Filtering: Runs standard ANN search, then discards items that don't match.

    Fails severely when the predicate is highly selective.
    """
    res = index.search(query, k=k, ef_search=ef_search)
    valid_ids: list[int] = []
    valid_dist: list[float] = []

    for item_id, dist in zip(res.ids, res.distances):
        if metadata_map.get(int(item_id)) == allowed_value:
            valid_ids.append(int(item_id))
            valid_dist.append(float(dist))

    return SearchResult(
        np.asarray(valid_ids, dtype=np.int64),
        np.asarray(valid_dist, dtype=np.float32),
    )


def overfetch_filter_search(
    index: HNSWIndex,
    metadata_map: dict[int, Any],
    query: np.ndarray,
    k: int,
    allowed_value: Any,
    multiplier: int = 10,
    ef_search: int = 100,
) -> SearchResult:
    """Over-fetch Filtering: Request more candidates than k, then filter down."""
    fetch_count = min(len(index), k * multiplier)
    ef = max(ef_search, fetch_count)
    res = index.search(query, k=fetch_count, ef_search=ef)

    valid_ids: list[int] = []
    valid_dist: list[float] = []

    for item_id, dist in zip(res.ids, res.distances):
        if metadata_map.get(int(item_id)) == allowed_value:
            valid_ids.append(int(item_id))
            valid_dist.append(float(dist))
            if len(valid_ids) == k:
                break

    return SearchResult(
        np.asarray(valid_ids, dtype=np.int64),
        np.asarray(valid_dist, dtype=np.float32),
    )


def filter_aware_hnsw_search(
    index: HNSWIndex,
    metadata_map: dict[int, Any],
    query: np.ndarray,
    k: int,
    allowed_value: Any,
    ef_search: int = 50,
) -> SearchResult:
    """Filter-aware graph traversal.

    Traverse the graph freely through non-matching nodes for routing,
    but only collect matching nodes into the final result set.
    """
    q = np.asarray(query, dtype=np.float32)
    if index.entry is None:
        raise ValueError("index is empty")

    entry = index.entry
    # Upper-layer greedy descent
    for layer in range(index.max_level, 0, -1):
        entry = index._greedy(q, entry, layer)

    # Layer 0 exploration with filtering logic
    visited = {entry}
    candidates: list[tuple[float, int]] = []
    best_matching: list[tuple[float, int]] = []  # max-heap: (-distance, node)

    entry_dist = index._distance(q, entry)
    heapq.heappush(candidates, (entry_dist, entry))

    entry_item_id = index.external_ids[entry]
    if metadata_map.get(entry_item_id) == allowed_value:
        heapq.heappush(best_matching, (-entry_dist, entry))

    while candidates:
        cand_dist, candidate = heapq.heappop(candidates)

        if len(best_matching) >= ef_search and cand_dist > -best_matching[0][0]:
            break

        for neighbor in index.links[0][candidate]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            dist = index._distance(q, neighbor)
            heapq.heappush(candidates, (dist, neighbor))

            neighbor_item_id = index.external_ids[neighbor]
            if metadata_map.get(neighbor_item_id) == allowed_value:
                if len(best_matching) < ef_search or dist < -best_matching[0][0]:
                    heapq.heappush(best_matching, (-dist, neighbor))
                    if len(best_matching) > ef_search:
                        heapq.heappop(best_matching)

    top = sorted(
        [(node, -neg_d) for neg_d, node in best_matching],
        key=lambda x: (x[1], index.external_ids[x[0]]),
    )[:k]

    ids = np.asarray([index.external_ids[node] for node, _ in top], dtype=np.int64)
    distances = np.asarray([d for _, d in top], dtype=np.float32)
    return SearchResult(ids, distances)
