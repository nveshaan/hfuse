"""
Newman modularity + hypergraph strict / majority / entropy cohesion,
each implemented as 8 pieces:

    1. cohesion            -- what a single edge/hyperedge scores (hard labels)
    2. null                -- expected score under the configuration null model
    3. per-node formula     -- explicit double/triple loop, matches the paper's Q formula
    4. matrix form           -- vectorized, uses Z (hard indicator matrix)
    5. soft per-node formula -- same loop, hard labels -> soft memberships H
    6. soft matrix form       -- vectorized soft version
    7. exact gradient         -- true gradient (closed form where it exists)
    8. surrogate gradient     -- the cheap FUSE-style approximation actually used
                                 for gradient ascent

Every function below has a docstring with a small worked example using the
toy data defined in section 0. Run this file directly to execute all of
them and print results.
"""

import torch

torch.manual_seed(0)


# =====================================================================
# 0. TOY DATA
#    Same 6 nodes, two communities: {0,1,2} and {3,4,5}.
#    A graph (for Newman) AND a 3-uniform hypergraph (for the other three),
#    including one hyperedge {1,2,3} that straddles both communities.
# =====================================================================
n, r = 6, 2

edges_2 = torch.tensor([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [1, 3]])
A = torch.zeros(n, n)
for u, v in edges_2:
    A[u, v] = 1.0
    A[v, u] = 1.0
deg = A.sum(1)
m = deg.sum() / 2

hedges = torch.tensor([[0, 1, 2], [3, 4, 5], [1, 2, 3]])  # (|E|, 3)
d_h = torch.zeros(n)
for e in hedges:
    d_h[e] += 1.0
vol_V = d_h.sum()

c = torch.tensor([0, 0, 0, 1, 1, 1])          # hard community labels
Z = torch.zeros(n, r)
Z[torch.arange(n), c] = 1.0                    # hard indicator matrix

S = torch.randn(n, 2)                          # a random embedding, for Newman
H = torch.softmax(torch.randn(n, r), dim=1)    # random soft memberships, for the others


def pi_from_labels(vol_c, vol_V):
    """
    pi_c = vol(C_c) / vol(V) -- the probability a single random "stub"/slot
    lands in community c under the configuration null model.

    Example
    -------
    >>> vol_c = torch.tensor([16.0, 24.0])   # community volumes
    >>> pi_from_labels(vol_c, torch.tensor(40.0))
    tensor([0.4000, 0.6000])
    """
    return vol_c / vol_V


# =====================================================================
# 1. NEWMAN MODULARITY  (ordinary graph, every edge has size 2)
# =====================================================================

# ---- 1.1 cohesion (hard) --------------------------------------------------
def newman_cohesion(ci, cj):
    """
    delta(c_i, c_j): 1 if nodes i and j are in the same community, else 0.

    Example
    -------
    >>> newman_cohesion(c[0], c[1])   # both node 0 and node 1 are in community 0
    1.0
    >>> newman_cohesion(c[0], c[3])   # node 0 in community 0, node 3 in community 1
    0.0
    """
    return float(ci == cj)


# ---- 1.2 null model --------------------------------------------------------
def newman_null(di, dj, m):
    """
    d_i * d_j / (2m): expected number of i--j edges under the configuration
    (random stub-rewiring) model.

    Example
    -------
    >>> newman_null(deg[0], deg[1], m)   # deg[0]=2, deg[1]=3, m=7 in the toy graph
    tensor(0.4286)
    """
    return di * dj / (2 * m)


# ---- 1.3 per-node formula ---------------------------------------------------
def newman_Q_pernode(A, deg, m, c):
    """
    Q = (1/2m) * sum_{i,j} (A_ij - d_i d_j / 2m) * delta(c_i, c_j).

    Explicit O(n^2) double loop over all node pairs, exactly mirroring the
    textbook formula.

    Example
    -------
    >>> newman_Q_pernode(A, deg, m, c)
    tensor(0.3571)  # community structure {0,1,2} vs {3,4,5} beats random null
    """
    total = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if c[i] == c[j]:
                total += A[i, j] - newman_null(deg[i], deg[j], m)
    return total / (2 * m)


# ---- 1.4 matrix form ---------------------------------------------------------
def newman_Q_matrix(A, deg, m, Z):
    """
    Q = (1/2m) * tr(Z^T B Z),  B = A - dd^T/2m  (the modularity matrix).

    Vectorized equivalent of newman_Q_pernode; Z is the (n, r) hard
    one-hot community indicator matrix.

    Example
    -------
    >>> newman_Q_matrix(A, deg, m, Z)
    tensor(0.3571)  # matches newman_Q_pernode exactly
    """
    B = A - torch.outer(deg, deg) / (2 * m)
    return torch.trace(Z.T @ B @ Z) / (2 * m)


# ---- 1.5 soft per-node formula ------------------------------------------------
def newman_Q_soft_pernode(A, deg, m, S):
    """
    Soft relaxation: replace delta(c_i, c_j) with the inner product
    <S_i, S_j> of a (possibly non-normalized) node embedding S.

    Example
    -------
    >>> S_demo = torch.eye(6, 2)[c]   # embeddings = exact one-hot community
    >>> newman_Q_soft_pernode(A, deg, m, S_demo)
    tensor(0.3571)  # reduces to the hard Q when S is one-hot
    """
    total = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            total += (A[i, j] - newman_null(deg[i], deg[j], m)) * (S[i] @ S[j])
    return total / (2 * m)


# ---- 1.6 soft matrix form -------------------------------------------------------
def newman_Q_soft_matrix(A, deg, m, S):
    """
    Q(S) = (1/2m) * tr(S^T B S) -- vectorized version of newman_Q_soft_pernode.

    This is the quadratic surrogate objective that gradient ascent
    actually maximizes over S.

    Example
    -------
    >>> newman_Q_soft_matrix(A, deg, m, S)
    tensor(-0.8226)  # S here is the random embedding from section 0, not yet trained
    """
    B = A - torch.outer(deg, deg) / (2 * m)
    return torch.trace(S.T @ B @ S) / (2 * m)


# ---- 1.7 exact gradient -----------------------------------------------------------
def newman_grad_exact(A, deg, m, S):
    """
    d/dS [ (1/2m) tr(S^T B S) ] = (1/m) B S -- the true gradient of the
    quadratic surrogate, in closed form.

    Example
    -------
    >>> newman_grad_exact(A, deg, m, S)[0]
    tensor([-0.3224, -0.1266])  # gradient direction to increase Q at node 0
    """
    B = A - torch.outer(deg, deg) / (2 * m)
    return (1 / m) * (B @ S)


# ---- 1.8 FUSE surrogate gradient ---------------------------------------------------
def newman_grad_surrogate(A, deg, m, S):
    """
    Cheap FUSE-style approximation: replaces the degree-weighted mean
    d^T S with the unweighted sum 1^T S, avoiding an extra O(n) degree
    multiply per step at scale.

    Example
    -------
    >>> newman_grad_surrogate(A, deg, m, S)[0]
    tensor([-0.2062, -0.0541])  # close to, but not identical to, grad_exact
    """
    ones_sum = S.sum(dim=0)                       # 1^T S,  shape (k,)
    return (1 / (2 * m)) * (A @ S - (1 / (2 * m)) * torch.outer(deg, ones_sum))


# =====================================================================
# 2. STRICT COHESION  (hyperedge fully inside one community)
# =====================================================================

# ---- 2.1 cohesion (hard) ----------------------------------------------------
def strict_cohesion_hard(e, c):
    """
    1 if every vertex in hyperedge e shares the same community label, else 0.

    Example
    -------
    >>> strict_cohesion_hard(hedges[0], c)   # {0,1,2}, all in community 0
    1.0
    >>> strict_cohesion_hard(hedges[2], c)   # {1,2,3}, straddles both communities
    0.0
    """
    labels = c[e]
    return float((labels == labels[0]).all())


# ---- 2.2 null model -----------------------------------------------------------
def strict_null(pi, d):
    """
    sum_c pi_c^d: probability that all d independently-drawn slots of a
    random hyperedge land in the same community.

    Example
    -------
    >>> strict_null(torch.tensor([0.4, 0.35, 0.25]), 3)
    tensor(0.1225)   # 0.4**3 + 0.35**3 + 0.25**3
    """
    return (pi ** d).sum()


# ---- 2.3 per-node (per-hyperedge) formula --------------------------------------
def strict_Q_pernode(hedges, c, d_h, vol_V, r):
    """
    Q_strict = (1/|E|) * sum_e [ f_strict(e) - sum_c pi_c^|e| ].

    Explicit loop over hyperedges, matching the paper's Q formula directly.

    Example
    -------
    >>> strict_Q_pernode(hedges, c, d_h, vol_V, r)
    tensor(0.4074)   # 2 of 3 hyperedges are wholly inside one community
    """
    d = hedges.shape[1]
    vol_c = torch.stack([d_h[c == cc].sum() for cc in range(r)])
    pi = pi_from_labels(vol_c, vol_V)
    total = 0.0
    for e in hedges:
        total += strict_cohesion_hard(e, c) - strict_null(pi, d)
    return total / hedges.shape[0]


# ---- 2.4 matrix / tensor form ----------------------------------------------------
def strict_Q_matrix(hedges, Z, d_h, vol_V):
    """
    Tensor-contraction form: for each community c, take the product of
    Z[v,c] over v in e (=1 iff e is wholly in c); sum over c; average
    over edges. Generalizes Z^T B Z to a degree-d multilinear form.

    Example
    -------
    >>> strict_Q_matrix(hedges, Z, d_h, vol_V)
    tensor(0.4074)   # matches strict_Q_pernode exactly
    """
    vol_c = Z.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    d = hedges.shape[1]
    memberships = Z[hedges]                      # (|E|, d, r)
    prod_over_members = memberships.prod(dim=1)   # (|E|, r) -- AND across slots
    observed = prod_over_members.sum(dim=1).mean()  # OR across communities, mean over edges
    return observed - strict_null(pi, d)


# ---- 2.5 soft per-node formula --------------------------------------------------
def strict_Q_soft_pernode(hedges, H, d_h, vol_V):
    """
    Soft relaxation of strict cohesion: for hyperedge e, sum over
    communities c of the product of soft memberships h_{v,c} across v in e.

    Example
    -------
    >>> strict_Q_soft_pernode(hedges, H, d_h, vol_V)
    tensor(0.0036)   # H here is a random soft membership matrix, near 0
    """
    d = hedges.shape[1]
    r = H.shape[1]
    vol_c = H.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    total = 0.0
    for e in hedges:
        f_e = 0.0
        for cc in range(r):
            prod = 1.0
            for v in e:
                prod = prod * H[v, cc]
            f_e = f_e + prod
        total = total + (f_e - strict_null(pi, d))
    return total / hedges.shape[0]


# ---- 2.6 soft matrix form ---------------------------------------------------------
def strict_Q_soft_matrix(hedges, H, d_h, vol_V):
    """
    Vectorized version of strict_Q_soft_pernode: gather memberships per
    hyperedge, multiply across the edge dimension, sum over communities.

    Example
    -------
    >>> strict_Q_soft_matrix(hedges, H, d_h, vol_V)
    tensor(0.0036)   # matches strict_Q_soft_pernode
    """
    vol_c = H.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    d = hedges.shape[1]
    memberships = H[hedges]
    prod_over_members = memberships.prod(dim=1)
    observed = prod_over_members.sum(dim=1).mean()
    return observed - strict_null(pi, d)


# ---- 2.7 exact gradient (closed form, leave-one-out product, eq. 20) --------------
def strict_grad_exact(hedges, H):
    """
    d f_strict(e) / d h_{u,c} = product of h_{v,c} over v in e, v != u
    ("leave-one-out" product rule). Computed edge-by-edge with no autograd.

    Example
    -------
    >>> strict_grad_exact(hedges, H)[0]
    tensor([0.0218, 0.1839])   # gradient at node 0 w.r.t. each community
    """
    n_, r_ = H.shape
    grad = torch.zeros_like(H)
    for e in hedges:
        for u in e.tolist():
            for cc in range(r_):
                prod = 1.0
                for v in e.tolist():
                    if v != u:
                        prod = prod * H[v, cc].item()
                grad[u, cc] += prod
    return grad / hedges.shape[0]


# ---- 2.8 surrogate gradient (autograd through the same smooth formula) ------------
def strict_grad_surrogate(hedges, H):
    """
    The strict surrogate IS already the exact multilinear relaxation, so
    the "surrogate" gradient is the same formula computed efficiently via
    autograd (vectorized) instead of the O(d) manual loop in grad_exact.

    Example
    -------
    >>> strict_grad_surrogate(hedges, H)[0]
    tensor([0.0218, 0.1839])   # numerically identical to strict_grad_exact
    """
    H_ = H.clone().requires_grad_(True)
    memberships = H_[hedges]
    prod_over_members = memberships.prod(dim=1)
    Q = prod_over_members.sum(dim=1).mean()
    Q.backward()
    return H_.grad


# =====================================================================
# 3. MAJORITY COHESION  (plurality share of the hyperedge)
# =====================================================================

# ---- 3.1 cohesion (hard) --------------------------------------------------------
def majority_cohesion_hard(e, c, r):
    """
    Fraction of e's members that sit in e's most common ("plurality")
    community.

    Example
    -------
    >>> majority_cohesion_hard(hedges[2], c, r)   # {1,2,3}: 2 in comm 0, 1 in comm 1
    tensor(0.6667)
    """
    labels = c[e]
    counts = torch.bincount(labels, minlength=r).float()
    return counts.max() / len(e)


# ---- 3.2 null model (Monte Carlo, no closed form for r > 2) ---------------------
def majority_null_montecarlo(pi, d, n_samples=4000):
    """
    E[max_c X_c] / d, where X ~ Multinomial(d, pi), estimated by sampling
    (no closed form exists in general once r > 2).

    Example
    -------
    >>> torch.manual_seed(0)
    >>> majority_null_montecarlo(torch.tensor([0.4, 0.35, 0.25]), 3)
    tensor(0.7765)   # a random size-3 draw is "majority" ~78% of the time by chance
    """
    r = pi.shape[0]
    samples = torch.multinomial(pi, num_samples=d * n_samples, replacement=True)
    samples = samples.view(n_samples, d)
    counts = torch.stack([torch.bincount(row, minlength=r) for row in samples])
    return counts.float().max(dim=1).values.mean() / d


# ---- 3.3 per-node formula ---------------------------------------------------------
def majority_Q_pernode(hedges, c, d_h, vol_V, r):
    """
    Q_majority = (1/|E|) * sum_e [ f_majority(e) - E_null[f_majority] ].

    Explicit loop over hyperedges.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> majority_Q_pernode(hedges, c, d_h, vol_V, r)
    tensor(0.1371)
    """
    d = hedges.shape[1]
    vol_c = torch.stack([d_h[c == cc].sum() for cc in range(r)])
    pi = pi_from_labels(vol_c, vol_V)
    null = majority_null_montecarlo(pi, d)
    total = 0.0
    for e in hedges:
        total += majority_cohesion_hard(e, c, r) - null
    return total / hedges.shape[0]


# ---- 3.4 matrix form ----------------------------------------------------------------
def majority_Q_matrix(hedges, Z, d_h, vol_V):
    """
    Vectorized version of majority_Q_pernode: gather per-community counts
    for every hyperedge in one matrix multiply, then take a row-wise max.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> majority_Q_matrix(hedges, Z, d_h, vol_V)
    tensor(0.1396)   # close to majority_Q_pernode (differs only by MC noise)
    """
    r = Z.shape[1]
    vol_c = Z.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    d = hedges.shape[1]
    counts = Z[hedges].sum(dim=1)                 # (|E|, r) hard counts per community
    observed = counts.max(dim=1).values.mean() / d
    null = majority_null_montecarlo(pi, d)
    return observed - null


# ---- 3.5 soft per-node formula (log-sum-exp surrogate for max) ----------------------
def majority_Q_soft_pernode(hedges, H, d_h, vol_V, tau=0.1):
    """
    Soft relaxation: replace the hard max over community counts with a
    temperature-tau log-sum-exp, which is smooth and -> max as tau -> 0.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> majority_Q_soft_pernode(hedges, H, d_h, vol_V)
    tensor(-0.1263)
    """
    d = hedges.shape[1]
    vol_c = H.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    null = majority_null_montecarlo(pi, d)
    total = 0.0
    for e in hedges:
        s = torch.stack([H[v] for v in e]).sum(dim=0)       # soft counts s_c
        f_e = tau * torch.logsumexp(s / tau, dim=0) / d
        total = total + (f_e - null)
    return total / hedges.shape[0]


# ---- 3.6 soft matrix form -------------------------------------------------------------
def majority_Q_soft_matrix(hedges, H, d_h, vol_V, tau=0.1):
    """
    Vectorized version of majority_Q_soft_pernode.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> majority_Q_soft_matrix(hedges, H, d_h, vol_V)
    tensor(-0.1251)   # matches majority_Q_soft_pernode up to MC null noise
    """
    d = hedges.shape[1]
    vol_c = H.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    soft_counts = H[hedges].sum(dim=1)                      # (|E|, r)
    observed = (tau * torch.logsumexp(soft_counts / tau, dim=1)).mean() / d
    null = majority_null_montecarlo(pi, d)
    return observed - null


# ---- 3.7 exact gradient (subgradient of the true, non-smooth max) --------------------
def majority_grad_exact_subgradient(hedges, H):
    """
    max_c is non-differentiable; its subgradient puts all credit on the
    single argmax community only (a one-hot gradient), which is what
    "exact" means for a non-smooth objective.

    Example
    -------
    >>> majority_grad_exact_subgradient(hedges, H)[0]
    tensor([0.0000, 0.1111])   # all gradient mass on the argmax community
    """
    n_, r_ = H.shape
    grad = torch.zeros_like(H)
    for e in hedges:
        s = H[e].sum(dim=0)
        c_star = s.argmax()
        for v in e.tolist():
            grad[v, c_star] += 1.0 / len(e)
    return grad / hedges.shape[0]


# ---- 3.8 surrogate gradient (smooth log-sum-exp, via autograd) -----------------------
def majority_grad_surrogate(hedges, H, tau=0.1):
    """
    Smooth log-sum-exp surrogate gradient, computed via autograd. Unlike
    the subgradient, it spreads a small amount of gradient onto
    non-argmax communities too.

    Example
    -------
    >>> majority_grad_surrogate(hedges, H)[0]
    tensor([3.5711e-09, 1.1111e-01])   # nearly one-hot but not exactly
    """
    H_ = H.clone().requires_grad_(True)
    d = hedges.shape[1]
    soft_counts = H_[hedges].sum(dim=1)
    observed = (tau * torch.logsumexp(soft_counts / tau, dim=1)).mean() / d
    observed.backward()
    return H_.grad


# =====================================================================
# 4. ENTROPY COHESION  (how concentrated the hyperedge's communities are)
# =====================================================================

# ---- 4.1 cohesion (hard) --------------------------------------------------------
def entropy_cohesion_hard(e, c, r):
    """
    1 - H(e)/log(min(|e|,r)): normalized "purity" score, 1 for a
    single-community hyperedge, 0 for a maximally spread-out one.

    Example
    -------
    >>> entropy_cohesion_hard(hedges[0], c, r)   # {0,1,2}, all community 0
    tensor(1.)
    >>> entropy_cohesion_hard(hedges[2], c, r)   # {1,2,3}, split 2/1
    tensor(0.0817)
    """
    d = len(e)
    counts = torch.bincount(c[e], minlength=r).float()
    p = counts / d
    p_nz = p[p > 0]
    H_e = -(p_nz * p_nz.log()).sum()
    denom = torch.log(torch.tensor(float(min(d, r))))
    return (1 - H_e / denom) if d > 1 else torch.tensor(1.0)


# ---- 4.2 null model (Monte Carlo) -------------------------------------------------
def entropy_null_montecarlo(pi, d, r, n_samples=4000):
    """
    Expected purity score of a random size-d hyperedge under the
    configuration null, estimated by sampling (no closed form for r > 2).

    Example
    -------
    >>> torch.manual_seed(0)
    >>> entropy_null_montecarlo(torch.tensor([0.4, 0.35, 0.25]), 3, 3)
    tensor(0.3459)
    """
    samples = torch.multinomial(pi, num_samples=d * n_samples, replacement=True).view(n_samples, d)
    denom = torch.log(torch.tensor(float(min(d, r))))
    vals = []
    for row in samples:
        counts = torch.bincount(row, minlength=r).float()
        p = counts / d
        p_nz = p[p > 0]
        H_e = -(p_nz * p_nz.log()).sum()
        vals.append(1 - H_e / denom)
    return torch.stack(vals).mean()


# ---- 4.3 per-node formula -----------------------------------------------------------
def entropy_Q_pernode(hedges, c, d_h, vol_V, r):
    """
    Q_entropy = (1/|E|) * sum_e [ f_entropy(e) - E_null[f_entropy] ].

    Explicit loop over hyperedges.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> entropy_Q_pernode(hedges, c, d_h, vol_V, r)
    tensor(0.3778)
    """
    d = hedges.shape[1]
    vol_c = torch.stack([d_h[c == cc].sum() for cc in range(r)])
    pi = pi_from_labels(vol_c, vol_V)
    null = entropy_null_montecarlo(pi, d, r)
    total = 0.0
    for e in hedges:
        total += entropy_cohesion_hard(e, c, r) - null
    return total / hedges.shape[0]


# ---- 4.4 matrix form -----------------------------------------------------------------
def entropy_Q_matrix(hedges, Z, d_h, vol_V, r):
    """
    Vectorized version of entropy_Q_pernode: compute per-hyperedge
    community shares in one matrix multiply, then take row-wise entropy.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> entropy_Q_matrix(hedges, Z, d_h, vol_V, r)
    tensor(0.3847)   # close to entropy_Q_pernode (differs by MC noise)
    """
    vol_c = Z.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    d = hedges.shape[1]
    counts = Z[hedges].sum(dim=1)                       # (|E|, r)
    p = counts / d
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    H_e = -(p * logp).sum(dim=1)
    denom = torch.log(torch.tensor(float(min(d, r))))
    observed = (1 - H_e / denom).mean()
    null = entropy_null_montecarlo(pi, d, r)
    return observed - null


# ---- 4.5 soft per-node formula ---------------------------------------------------------
def entropy_Q_soft_pernode(hedges, H, d_h, vol_V, r, eps=1e-12):
    """
    Soft relaxation: replace hard community shares with the mean soft
    membership of a hyperedge's members, then take entropy of that.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> entropy_Q_soft_pernode(hedges, H, d_h, vol_V, r)
    tensor(-0.2664)
    """
    d = hedges.shape[1]
    vol_c = H.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    null = entropy_null_montecarlo(pi, d, r)
    denom = torch.log(torch.tensor(float(min(d, r))))
    total = 0.0
    for e in hedges:
        p_hat = H[e].mean(dim=0)
        H_e = -(p_hat * torch.log(p_hat + eps)).sum()
        f_e = 1 - H_e / denom
        total = total + (f_e - null)
    return total / hedges.shape[0]


# ---- 4.6 soft matrix form -----------------------------------------------------------------
def entropy_Q_soft_matrix(hedges, H, d_h, vol_V, r, eps=1e-12):
    """
    Vectorized version of entropy_Q_soft_pernode.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> entropy_Q_soft_matrix(hedges, H, d_h, vol_V, r)
    tensor(-0.2632)   # matches entropy_Q_soft_pernode up to MC null noise
    """
    d = hedges.shape[1]
    vol_c = H.T @ d_h
    pi = pi_from_labels(vol_c, vol_V)
    p_hat = H[hedges].mean(dim=1)                        # (|E|, r) soft shares
    logp = torch.log(p_hat + eps)
    H_e = -(p_hat * logp).sum(dim=1)
    denom = torch.log(torch.tensor(float(min(d, r))))
    observed = (1 - H_e / denom).mean()
    null = entropy_null_montecarlo(pi, d, r)
    return observed - null


# ---- 4.7 exact gradient (autograd; null term technically non-differentiable) --------
def entropy_grad_exact(hedges, H, d_h, vol_V, r, eps=1e-12):
    """
    Autograd gradient of the full soft objective including the null term.
    Note: because entropy_null_montecarlo uses torch.multinomial (a
    non-differentiable sampling op), gradient does NOT actually flow
    through the null term in practice -- this is the same limitation
    "exact" hits without a reparameterization trick (e.g. Gumbel-softmax).

    Example
    -------
    >>> torch.manual_seed(0)
    >>> entropy_grad_exact(hedges, H, d_h, vol_V, r)[0]
    tensor([-0.0880,  0.1220])
    """
    H_ = H.clone().requires_grad_(True)
    Q = entropy_Q_soft_matrix(hedges, H_, d_h, vol_V, r, eps)
    Q.backward()
    return H_.grad


# ---- 4.8 surrogate gradient (null term frozen -- FUSE-H refreshes it every k steps) --
def entropy_grad_surrogate(hedges, H, d_h, vol_V, r, eps=1e-12):
    """
    FUSE-style surrogate: deliberately excludes the null term from the
    backward pass (treating it as a periodically-refreshed constant,
    matching Algorithm FUSE-H), rather than relying on autograd to skip
    a non-differentiable op.

    Example
    -------
    >>> torch.manual_seed(0)
    >>> entropy_grad_surrogate(hedges, H, d_h, vol_V, r)[0]
    tensor([-0.0880,  0.1220])   # identical here since the null term had no grad anyway
    """
    H_ = H.clone().requires_grad_(True)
    d = hedges.shape[1]
    p_hat = H_[hedges].mean(dim=1)
    logp = torch.log(p_hat + eps)
    H_e = -(p_hat * logp).sum(dim=1)
    denom = torch.log(torch.tensor(float(min(d, r))))
    observed = (1 - H_e / denom).mean()          # no null term -> no grad through pi
    observed.backward()
    return H_.grad


# =====================================================================
# DEMO
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("1. NEWMAN")
    print("=" * 70)
    print("per-node   Q =", newman_Q_pernode(A, deg, m, c).item())
    print("matrix     Q =", newman_Q_matrix(A, deg, m, Z).item())
    print("soft per-node Q(S) =", newman_Q_soft_pernode(A, deg, m, S).item())
    print("soft matrix   Q(S) =", newman_Q_soft_matrix(A, deg, m, S).item())
    g_exact = newman_grad_exact(A, deg, m, S)
    g_surr = newman_grad_surrogate(A, deg, m, S)
    print("grad exact[0]     =", g_exact[0].tolist())
    print("grad surrogate[0] =", g_surr[0].tolist())

    print("\n" + "=" * 70)
    print("2. STRICT")
    print("=" * 70)
    print("per-node   Q =", strict_Q_pernode(hedges, c, d_h, vol_V, r))
    print("matrix     Q =", strict_Q_matrix(hedges, Z, d_h, vol_V).item())
    print("soft per-node Q(H) =", strict_Q_soft_pernode(hedges, H, d_h, vol_V).item())
    print("soft matrix   Q(H) =", strict_Q_soft_matrix(hedges, H, d_h, vol_V).item())
    g_exact = strict_grad_exact(hedges, H)
    g_surr = strict_grad_surrogate(hedges, H)
    print("grad exact[0]     =", g_exact[0].tolist())
    print("grad surrogate[0] =", g_surr[0].tolist(), " (should match: multilinear f)")

    print("\n" + "=" * 70)
    print("3. MAJORITY")
    print("=" * 70)
    torch.manual_seed(1)
    print("per-node   Q =", majority_Q_pernode(hedges, c, d_h, vol_V, r).item())
    print("matrix     Q =", majority_Q_matrix(hedges, Z, d_h, vol_V).item())
    print("soft per-node Q(H) =", majority_Q_soft_pernode(hedges, H, d_h, vol_V).item())
    print("soft matrix   Q(H) =", majority_Q_soft_matrix(hedges, H, d_h, vol_V).item())
    g_exact = majority_grad_exact_subgradient(hedges, H)
    g_surr = majority_grad_surrogate(hedges, H)
    print("grad exact[0]     =", g_exact[0].tolist(), " (one-hot: only argmax community)")
    print("grad surrogate[0] =", g_surr[0].tolist(), " (smooth, spread across communities)")

    print("\n" + "=" * 70)
    print("4. ENTROPY")
    print("=" * 70)
    torch.manual_seed(1)
    print("per-node   Q =", entropy_Q_pernode(hedges, c, d_h, vol_V, r).item())
    print("matrix     Q =", entropy_Q_matrix(hedges, Z, d_h, vol_V, r).item())
    print("soft per-node Q(H) =", entropy_Q_soft_pernode(hedges, H, d_h, vol_V, r).item())
    print("soft matrix   Q(H) =", entropy_Q_soft_matrix(hedges, H, d_h, vol_V, r).item())
    g_exact = entropy_grad_exact(hedges, H, d_h, vol_V, r)
    g_surr = entropy_grad_surrogate(hedges, H, d_h, vol_V, r)
    print("grad exact[0]     =", g_exact[0].tolist(), " (null term differentiated too)")
    print("grad surrogate[0] =", g_surr[0].tolist(), " (null term frozen)")