"""The single FUSE engine: projected gradient ascent on modularity with a
backtracking (Armijo-style) line search and a power-iteration Lipschitz seed.

One function drives every FUSE variant -- the only differences are (a) which
Operator is passed in, and (b) whether SSL is on. This is the unified core.
"""
from __future__ import annotations
import time
import numpy as np
from . import ssl as SSL


def power_iteration_spectral_norm(operator, n_iters=40, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=operator.H.n)
    v /= np.linalg.norm(v)
    for _ in range(n_iters):
        Bv = operator.B_matvec(v)
        nrm = np.linalg.norm(Bv)
        if nrm < 1e-12:
            break
        v = Bv / nrm
    Bv = operator.B_matvec(v)
    return abs(float(v @ Bv))


def fuse_embedding(operator, k=16, n_iters=200, seed=1,
                   ssl=False, ssl_views=2, ssl_drop=0.2, ssl_lambda=0.0,
                   ssl_seed=0, growth=1.2, shrink=0.5, max_backtracks=30,
                   patience=25, tol=1e-6):
    """Return (S, info). Projected ascent on operator's modularity; optional SSL."""
    H = operator.H
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(H.n, k))
    S /= np.linalg.norm(S, axis=1, keepdims=True)

    L = 2.0 * power_iteration_spectral_norm(operator, seed=seed)
    eta = 2.0 / max(L, 1e-12)
    eta_floor = eta * 1e-8

    use_ssl = bool(ssl) and ssl_lambda > 0.0
    dinv_sqrt = SSL.make_dinv_sqrt(operator) if use_ssl else None

    Q = operator.modularity(S)
    Q0 = Q
    no_improve = 0
    t0 = time.perf_counter()
    stop = n_iters
    for t in range(1, n_iters + 1):
        G = operator.grad(S)
        if use_ssl:
            rng_v = np.random.default_rng(ssl_seed * 100003 + t)
            g_ssl = SSL.spectral_contrastive_grad(S, operator, dinv_sqrt,
                                                  ssl_views, ssl_drop, rng_v)
            G = SSL.mix_scale_free(G, g_ssl, ssl_lambda)
        # backtracking line search preserving monotone Q
        S_new, Q_new = S, Q
        for _ in range(max_backtracks):
            cand = S + eta * G
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            Qc = operator.modularity(cand)
            if Qc >= Q:
                S_new, Q_new = cand, Qc
                break
            eta = max(eta * shrink, eta_floor)
        improve = Q_new - Q
        S, Q = S_new, Q_new
        eta *= growth
        no_improve = no_improve + 1 if improve < tol else 0
        if no_improve >= patience:
            stop = t
            break
    elapsed = time.perf_counter() - t0
    return S, {"time": elapsed, "Q": float(Q), "Q0": float(Q0),
               "iters": stop, "L": L, "ssl": use_ssl}
