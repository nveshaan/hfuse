"""Hypergraph adjacency operators for FUSE.

Every operator induces a symmetric n x n adjacency A and a modularity matrix
    B = A - d d^T / 2m,   d = A 1,   2m = sum_i d_i,
and FUSE maximizes  Q(S) = (1/2m) Tr(S^T B S)  by projected gradient ascent.
The exact modularity gradient (matrix-free via the cached sparse A) is
    grad Q = (1/m) ( A S - (1/2m) d (d^T S) ).

Three operator VARIANTS are tested (this is the explicit answer to "how many
adjacency versions of FUSE"):

  1. banerjee    -- A_ij = sum_{e supseteq {i,j}} 1/(|e|-1)   [PRIMARY: the
                    definition provided; d_i = # incident edges, 2m = sum_e |e|]
  2. clique      -- A_ij = sum_{e supseteq {i,j}} 1           [unnormalized
                    clique expansion]
  3. normalized  -- D^-1/2 A_banerjee D^-1/2                  [symmetric-normalized,
                    Zhou-style]

Each operator exposes: sparse A, degree d, m2; a matrix-free apply(S)=A@S; the
modularity grad(S); B_matvec(v) for the Lipschitz power iteration; and an
edge-dropout matrix-free apply_subset(S, groups) used by the SSL regularizer,
so SSL is consistent with whichever operator FUSE is using.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


def _weight(kind, s):
    if kind == "banerjee":
        return 1.0 / (s - 1)
    if kind == "clique":
        return 1.0
    if kind == "normalized":          # base weights are Banerjee; normalized after
        return 1.0 / (s - 1)
    raise ValueError(kind)


def _build_sparse(H, kind):
    """Vectorized sparse clique expansion with per-size weight w(|e|)."""
    sizes = H.edge_sizes
    offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    rows, cols, data = [], [], []
    for s in np.unique(sizes):
        if s < 2:
            continue
        idx = np.where(sizes == s)[0]
        starts = offsets[idx]
        E = H.flat_v[starts[:, None] + np.arange(s)[None, :]]     # (n_s, s)
        w = _weight(kind, s)
        A = np.broadcast_to(E[:, :, None], (len(idx), s, s))
        B = np.broadcast_to(E[:, None, :], (len(idx), s, s))
        mask = A != B
        rows.append(A[mask]); cols.append(B[mask])
        data.append(np.full(int(mask.sum()), w))
    A = sparse.coo_matrix((np.concatenate(data),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(H.n, H.n)).tocsr()
    return A


class Operator:
    def __init__(self, H, kind="banerjee"):
        self.kind = kind
        self.H = H
        self.groups = H.edges_by_size()
        A = _build_sparse(H, kind)
        if kind == "normalized":
            d = np.asarray(A.sum(axis=1)).ravel()
            dinv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
            Dm = sparse.diags(dinv)
            A = (Dm @ A @ Dm).tocsr()
        self.A = A
        self.d = np.asarray(A.sum(axis=1)).ravel()
        self.m2 = float(self.d.sum())
        if self.m2 <= 0:
            self.m2 = 1.0

    # --- core ops ---
    def apply(self, S):
        return self.A @ S

    def grad(self, S):
        AS = self.A @ S
        dS = self.d @ S
        m = self.m2 / 2.0
        return (1.0 / m) * (AS - (1.0 / self.m2) * np.outer(self.d, dS))

    def modularity(self, S):
        AS = self.A @ S
        dS = self.d @ S
        return (np.sum(S * AS) - (1.0 / self.m2) * np.sum(dS * dS)) / self.m2

    def B_matvec(self, v):
        Av = self.A @ v
        return Av - (self.d * (self.d @ v)) / self.m2

    # --- matrix-free edge-dropout apply, for SSL consensus views ---
    def apply_subset(self, S, groups_subset, dinv_sqrt=None):
        n, k = S.shape
        Sn = S if dinv_sqrt is None else dinv_sqrt[:, None] * S
        out = np.zeros((n, k))
        for s, Em in groups_subset.items():
            if Em.shape[0] == 0:
                continue
            w = _weight(self.kind, s)
            Ssum = Sn[Em].sum(axis=1)
            contrib = w * (Ssum[:, None, :] - Sn[Em])
            np.add.at(out, Em, contrib)
        return out if dinv_sqrt is None else dinv_sqrt[:, None] * out


OPERATOR_KINDS = ["banerjee", "clique", "normalized"]
