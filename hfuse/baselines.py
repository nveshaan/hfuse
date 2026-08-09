"""Two authentic non-FUSE baselines for grounding.

  * HG-Spectral(Zhou07) -- leading eigenvectors of the normalized hypergraph
    operator Theta = Dv^-1/2 H De^-1 H^T Dv^-1/2 (Zhou-Huang-Scholkopf 2007).
  * HNX-Kumar          -- HyperNetX hypergraph_modularity.kumar: Kumar et al.
    reweighted-Louvain on the Kaminski-Pralat-Theberge modularity. Returns a
    partition directly.
"""
from __future__ import annotations
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
warnings.filterwarnings("ignore")


def _incidence(H):
    rows, cols = [], []
    for ei, e in enumerate(H.edges):
        for v in e:
            rows.append(int(v)); cols.append(ei)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                             shape=(H.n, H.n_edges))


def zhou_spectral_embedding(H, dim):
    Hi = _incidence(H)
    dv = np.asarray(Hi.sum(axis=1)).ravel()
    de = np.asarray(Hi.sum(axis=0)).ravel()
    Dv = sparse.diags(1.0 / np.sqrt(np.where(dv > 0, dv, 1.0)))
    De = sparse.diags(1.0 / np.where(de > 0, de, 1.0))
    Theta = (Dv @ Hi @ De @ Hi.T @ Dv).tocsr()
    kk = min(dim + 1, H.n - 1)
    vals, vecs = eigsh(Theta, k=max(kk, 1), which="LA")
    order = np.argsort(vals)[::-1]
    return vecs[:, order][:, 1:dim + 1]


def zhou_spectral(H, K, seed=0):
    t0 = time.perf_counter()
    emb = zhou_spectral_embedding(H, dim=max(K, 2))
    from .metrics import kmeans_labels
    lab = kmeans_labels(emb, K, seed)
    return lab, {"time": time.perf_counter() - t0}


def hnx_kumar(H, K, seed=0):
    t0 = time.perf_counter()
    import hypernetx as hnx
    import hypernetx.algorithms.hypergraph_modularity as hmod
    HG = hnx.Hypergraph({f"e{i}": [int(v) for v in e] for i, e in enumerate(H.edges)})
    part = hmod.kumar(HG)
    labels = -np.ones(H.n, dtype=int)
    for ci, s in enumerate(part):
        for v in s:
            labels[int(v)] = ci
    nxt = len(part)
    for i in range(H.n):
        if labels[i] < 0:
            labels[i] = nxt; nxt += 1
    return labels, {"time": time.perf_counter() - t0, "n_found": len(part)}
