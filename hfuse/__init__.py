"""hfuse: unified adjacency-operator FUSE benchmark on real + synthetic hypergraphs."""
from . import datasets, operators, fuse, ssl, baselines, metrics
from .hypergraph import Hypergraph
__all__ = ["datasets", "operators", "fuse", "ssl", "baselines", "metrics", "Hypergraph"]
