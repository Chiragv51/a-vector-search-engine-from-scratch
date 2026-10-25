"""
Hierarchical Navigable Small World (HNSW) Index from scratch.
Implements multi-layer proximity graphs, greedy upper-layer descent,
bounded 2-heap search at layer 0, and neighbor pruning heuristics.
"""

from __future__ import annotations
from collections import defaultdict
import heapq
import numpy as np
from .exact_loop import SearchResult


class HNSWIndex:
    """Educational Hierarchical Navigable Small World (HNSW) Index."""

    def __init__(
        self,
        dimension: int,
        M: int = 12,
        ef_construction: int = 100,
        seed: int = 2026,
        heuristic: bool = True,
    ):
        if dimension <= 0 or M < 2 or ef_construction < M:
            raise ValueError(
                f"invalid HNSW configuration: dimension={dimension}, M={M}, ef_construction={ef_construction}"
            )
        self.dimension = dimension
        self.M = M
        self.M0 = 2 * M  # Max degree on layer 0 (standard HNSW convention)
        self.ef_construction = ef_construction
        self.rng = np.random.default_rng(seed)
        self.level_multiplier = 1.0 / np.log(M)
        self.heuristic = heuristic

        self.vectors: list[np.ndarray] = []
        self.external_ids: list[int] = []
        self.levels: list[int] = []
        # links[layer][node] -> set(neighbor_nodes)
        self.links: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
        self.entry: int | None = None
        self.max_level: int = -1

    def __len__(self) -> int:
        return len(self.vectors)

    def _distance(self, q: np.ndarray, node: int) -> float:
        """Calculate squared L2 distance between query q and indexed node."""
        delta = q - self.vectors[node]
        return float(delta @ delta)

    def _random_level(self) -> int:
        """Assign max level to node using exponential decay distribution."""
        u = max(float(self.rng.random()), np.finfo(float).tiny)
        return int(np.floor(-np.log(u) * self.level_multiplier))

    def _greedy(self, query: np.ndarray, entry: int, layer: int) -> int:
        """Greedy walk on a single upper layer to find the nearest local minimum."""
        current = entry
        current_distance = self._distance(query, current)
        changed = True
        while changed:
            changed = False
            for neighbor in self.links[layer][current]:
                distance = self._distance(query, neighbor)
                if distance < current_distance:
                    current, current_distance = neighbor, distance
                    changed = True
        return current

    def _search_layer(
        self, query: np.ndarray, entry_nodes: list[int], ef: int, layer: int
    ) -> list[tuple[int, float]]:
        """Bounded candidate expansion using two priority queues.

        - candidates: min-heap of (distance, node) to explore promising frontier
        - best: max-heap of (-distance, node) tracking the closest ef nodes discovered so far
        """
        visited = set(entry_nodes)
        candidates: list[tuple[float, int]] = []
        best: list[tuple[float, int]] = []

        for node in entry_nodes:
            dist = self._distance(query, node)
            heapq.heappush(candidates, (dist, node))
            heapq.heappush(best, (-dist, node))

        while candidates:
            cand_dist, candidate = heapq.heappop(candidates)
            worst_best_dist = -best[0][0]

            if cand_dist > worst_best_dist and len(best) >= ef:
                break

            for neighbor in self.links[layer][candidate]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                dist = self._distance(query, neighbor)

                if len(best) < ef or dist < -best[0][0]:
                    heapq.heappush(candidates, (dist, neighbor))
                    heapq.heappush(best, (-dist, neighbor))
                    if len(best) > ef:
                        heapq.heappop(best)

        return [(node, -neg_dist) for neg_dist, node in best]

    def _select_neighbors_simple(
        self, candidates: list[tuple[int, float]], limit: int
    ) -> list[int]:
        """Simple nearest-M neighbor selection."""
        sorted_cand = sorted(candidates, key=lambda x: x[1])
        return [node for node, _ in sorted_cand[:limit]]

    def _select_neighbors_heuristic(
        self, q: np.ndarray, candidates: list[tuple[int, float]], limit: int
    ) -> list[int]:
        """Paper's diversity heuristic: accept candidate only if closer to base node

        than to any previously selected neighbor. Reduces redundant clustered links.
        """
        # Sort candidates ascending by distance to q
        sorted_cands = sorted(candidates, key=lambda x: x[1])
        selected: list[int] = []

        for cand_node, cand_dist in sorted_cands:
            if len(selected) >= limit:
                break
            cand_vec = self.vectors[cand_node]
            # Check if closer to q than to any already chosen neighbor
            is_diverse = True
            for sel_node in selected:
                dist_to_sel = float(np.sum((cand_vec - self.vectors[sel_node]) ** 2))
                if dist_to_sel < cand_dist:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(cand_node)

        # Fallback: if heuristic is too aggressive, backfill with nearest
        if len(selected) < limit:
            for cand_node, _ in sorted_cands:
                if cand_node not in selected:
                    selected.append(cand_node)
                    if len(selected) >= limit:
                        break

        return selected

    def _connect(self, a: int, b: int, layer: int) -> None:
        """Add bidirectional edge between nodes a and b at given layer and enforce degree limits."""
        self.links[layer][a].add(b)
        self.links[layer][b].add(a)

        max_deg = self.M0 if layer == 0 else self.M
        for node in (a, b):
            if len(self.links[layer][node]) > max_deg:
                q = self.vectors[node]
                neighbors = list(self.links[layer][node])
                cand_list = [(n, self._distance(q, n)) for n in neighbors]

                if self.heuristic:
                    keep = set(self._select_neighbors_heuristic(q, cand_list, max_deg))
                else:
                    keep = set(self._select_neighbors_simple(cand_list, max_deg))

                removed = self.links[layer][node] - keep
                self.links[layer][node] = keep
                for other in removed:
                    self.links[layer][other].discard(node)

    def add(self, vector, item_id: int) -> None:
        """Insert a vector with stable external item_id into the HNSW graph."""
        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dimension,) or not np.all(np.isfinite(q)):
            raise ValueError(f"invalid vector: expected shape ({self.dimension},) with finite values")
        if item_id in self.external_ids:
            raise ValueError(f"duplicate item_id: {item_id}")

        node = len(self.vectors)
        level = self._random_level()
        self.vectors.append(q.copy())
        self.external_ids.append(item_id)
        self.levels.append(level)

        if self.entry is None:
            self.entry = node
            self.max_level = level
            for layer in range(level + 1):
                self.links[layer][node]
            return

        entry = self.entry
        # 1. Greedy descent down to level + 1
        for layer in range(self.max_level, level, -1):
            entry = self._greedy(q, entry, layer)

        # 2. Bounded candidate search and edge creation from min(level, max_level) down to 0
        curr_entry = [entry]
        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(q, curr_entry, self.ef_construction, layer)
            max_deg = self.M0 if layer == 0 else self.M

            if self.heuristic:
                selected = self._select_neighbors_heuristic(q, candidates, max_deg)
            else:
                selected = self._select_neighbors_simple(candidates, max_deg)

            for neighbor in selected:
                self._connect(node, neighbor, layer)

            if candidates:
                curr_entry = [min(candidates, key=lambda x: x[1])[0]]

        if level > self.max_level:
            self.entry = node
            self.max_level = level

    def search(self, query, k: int = 10, ef_search: int = 50) -> SearchResult:
        """Query top-k nearest neighbors using HNSW graph traversal."""
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dimension,) or not np.all(np.isfinite(q)):
            raise ValueError(f"invalid query: expected shape ({self.dimension},) with finite values")
        if self.entry is None:
            raise ValueError("index is empty")
        if not 1 <= k <= len(self.vectors):
            raise ValueError(f"k must be between 1 and {len(self.vectors)}, got {k}")
        if ef_search < k:
            raise ValueError(f"ef_search ({ef_search}) must be >= k ({k})")

        entry = self.entry
        # Coarse greedy descent on upper layers
        for layer in range(self.max_level, 0, -1):
            entry = self._greedy(q, entry, layer)

        # Fine-grained 2-heap search on layer 0
        candidates = self._search_layer(q, [entry], ef_search, 0)
        # Sort candidates by distance with external ID tie-breaker
        top = sorted(candidates, key=lambda x: (x[1], self.external_ids[x[0]]))[:k]

        ids = np.asarray([self.external_ids[node] for node, _ in top], dtype=np.int64)
        distances = np.asarray([dist for _, dist in top], dtype=np.float32)
        return SearchResult(ids, distances)
