"""Self-supervised spectral-contrastive regularizer (HaoChen et al., NeurIPS 2021),
operator-agnostic, run INSIDE the FUSE line-search loop.

    grad_ssl = W_bar S - S (S^T S)

W_bar is the consensus over `views` edge-dropout augmentations of the SAME
adjacency operator FUSE is using (via operator.apply_subset), symmetric-
normalized for stable scale. -S(S^T S) is the uniformity term. Mixing is
scale-free so one lambda transfers across datasets:

    grad = grad_mod + lambda * (||grad_mod|| / ||grad_ssl||) * grad_ssl

lambda == 0 returns grad_mod unchanged (exact baseline).
"""
from __future__ import annotations
import numpy as np


def make_dinv_sqrt(operator):
    d = operator.d
    return np.where(d > 0, 1.0 / np.sqrt(d + 1e-12), 0.0)


def _dropout_groups(groups, drop, rng):
    out = {}
    for s, Em in groups.items():
        if Em.shape[0] == 0:
            continue
        keep = rng.random(Em.shape[0]) >= drop
        out[s] = Em[keep]
    return out


def spectral_contrastive_grad(S, operator, dinv_sqrt, views, drop, rng):
    n, k = S.shape
    acc = np.zeros((n, k))
    for _ in range(views):
        gsub = _dropout_groups(operator.groups, drop, rng)
        acc += operator.apply_subset(S, gsub, dinv_sqrt=dinv_sqrt)
    Wbar_S = acc / max(views, 1)
    return Wbar_S - S @ (S.T @ S)


def mix_scale_free(grad_mod, grad_ssl, lam, eps=1e-12):
    if lam == 0.0:
        return grad_mod
    nm = float(np.linalg.norm(grad_mod))
    ns = float(np.linalg.norm(grad_ssl))
    return grad_mod + lam * (nm / (ns + eps)) * grad_ssl
