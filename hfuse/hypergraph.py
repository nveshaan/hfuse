"""Unified hypergraph container.

Stores the edge list plus a flattened incidence structure and the degree /
volume bookkeeping shared by every adjacency operator. `d` is the unweighted
vertex degree (number of incident hyperedges) and `m2 = sum_e |e|`, so the
Banerjee identity vol(V) = sum_i d_i = m2 holds.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Hypergraph:
    n: int
    edges: list                      # list[np.ndarray[int]]
    labels: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.edges = [np.asarray(e, dtype=np.int64) for e in self.edges
                      if len(np.unique(e)) >= 2]
        self.n_edges = len(self.edges)
        sizes = np.array([len(e) for e in self.edges], dtype=np.int64)
        self.edge_sizes = sizes
        self.flat_v = (np.concatenate(self.edges) if self.n_edges else
                       np.array([], dtype=np.int64))
        self.edge_id = np.repeat(np.arange(self.n_edges), sizes)
        self.d = np.bincount(self.flat_v, minlength=self.n).astype(float)
        self.m2 = float(sizes.sum())

    @property
    def avg_edge_size(self):
        return float(self.edge_sizes.mean()) if self.n_edges else 0.0

    def edges_by_size(self):
        groups: dict[int, list] = {}
        for e in self.edges:
            groups.setdefault(len(e), []).append(e)
        return {s: np.asarray(v, dtype=np.int64) for s, v in groups.items()}

    def __repr__(self):
        return (f"Hypergraph(n={self.n}, m={self.n_edges}, "
                f"avg|e|={self.avg_edge_size:.2f})")
