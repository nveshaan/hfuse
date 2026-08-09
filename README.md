# hfuse — Unified adjacency-operator FUSE benchmark for hypergraph modularity

One clean experiment: **FUSE** (projected-ascent modularity maximization on a
hypergraph adjacency operator) tested under **three adjacency operators**, each
**with and without** a self-supervised regularizer, against two authentic
baselines, on **real and synthetic** hypergraphs — a single runner, a single
results table.

```bash
uv sync
python run.py            # full run  (or: python run.py --quick)
```

## The one experiment

**FUSE engine** (`hfuse/fuse.py`): maximize `Q(S) = (1/2m)·Tr(SᵀB S)`,
`B = A − ddᵀ/2m`, by projected gradient ascent on the unit sphere with a
backtracking (Armijo) line search seeded by a power-iteration Lipschitz bound.
One function drives every variant; the only differences are the operator and the
SSL toggle.

**Adjacency operators** (`hfuse/operators.py`) — *this is how many "versions of
adjacency-matrix FUSE" are tested:* **3**.
1. `banerjee` — **primary**; `A_ij = Σ_{e⊇{i,j}} 1/(|e|−1)` (the provided
   definition: `d_i` = #incident edges, `2m = Σ_e|e|`).
2. `clique` — unnormalized clique expansion, weight 1 per within-edge pair.
3. `normalized` — `D^{-1/2} A_banerjee D^{-1/2}` (symmetric / Zhou-style).

**SSL** (`hfuse/ssl.py`) — a **uniform toggle** applied to *every* operator (so
6 FUSE variants = 3 operators × {off, on}). Multi-view spectral-contrastive
regularizer (HaoChen et al., NeurIPS 2021) run inside the same ascent loop:
`grad_ssl = W̄S − S(SᵀS)`, where `W̄` is the consensus over edge-dropout views of
the *same* operator; mixed scale-free so one `λ` transfers across datasets;
`λ=0` reproduces the baseline exactly.

**Baselines** (`hfuse/baselines.py`): `HG-Spectral(Zhou07)` (eigenvectors of the
normalized hypergraph operator) and `HNX-Kumar(KPT)` (HyperNetX reweighted-
Louvain on Kamiński–Prałat–Théberge modularity).

**Methods per dataset (8):** `FUSE[banerjee|clique|normalized]` × {—, +SSL},
plus the 2 baselines. All embeddings → k-means(K); recovery by ARI/AMI/NMI, plus
the Banerjee modularity of each partition as a common yardstick.

## Datasets (`hfuse/datasets.py`)

**Real (bundled under `hfuse/data/`, no network needed** — reconstructed from the
Chodrow–Veldt–Benson / Austin-Benson ARB release):

| dataset | n | hyperedges | K | ground truth |
|---|--:|--:|--:|---|
| `primary_school` | 242 | 12,704 | 11 | classroom (SocioPatterns contact) |
| `high_school` | 327 | 7,818 | 9 | class |
| `senate_committees` | 282 | 315 | 2 | party (committee co-membership) |
| `house_committees` | 1,290 | 340 | 2 | party |

**Synthetic (control + scalability):** `planted_partition`, `hierarchical_nested`.

## Results (`outputs/results.csv`, AMI)

| method | primary | high | senate | house | planted | hier |
|---|--:|--:|--:|--:|--:|--:|
| FUSE[banerjee] | 0.651 | 0.729 | ~0 | ~0 | 1.00 | 1.00 |
| **FUSE[banerjee]+SSL** | **0.923** | **0.993** | ~0 | ~0 | 1.00 | 1.00 |
| FUSE[clique] | 0.711 | 0.738 | ~0 | ~0 | 1.00 | 1.00 |
| FUSE[clique]+SSL | 0.898 | 0.960 | ~0 | ~0 | 1.00 | 1.00 |
| FUSE[normalized] | 0.688 | 0.713 | ~0 | ~0 | 1.00 | 1.00 |
| FUSE[normalized]+SSL | 0.902 | 1.000 | ~0 | ~0 | 1.00 | 1.00 |
| HG-Spectral(Zhou07) | 0.915 | 0.993 | ~0 | ~0 | 1.00 | 1.00 |
| HNX-Kumar(KPT) | 0.803 | 0.919 | ~0 | ~0 | 1.00 | 1.00 |

**Findings.**
1. **SSL is a large, consistent win on real contact hypergraphs** (+0.19…+0.29
   AMI across all operators), lifting FUSE to parity with / above the strongest
   baseline (primary 0.65→0.92; high 0.73→0.99). On saturated synthetic and on
   the committee data it neither helps nor hurts — it behaves as a
   noise-robustness regularizer, which matters on real data. See
   `fig_ssl_effect.png`.
2. **Operator choice matters little once SSL is on** — banerjee/clique/normalized
   all land ~0.90–1.00 on the contact data with SSL.
3. **Committee → party is unrecoverable for every method** (AMI ≈ 0): committee
   co-membership modularity does not encode party (committees are bipartisan).
   An honest negative result, not a bug.
4. **Scalability:** Banerjee FUSE is **near-linear** to 50k vertices
   (0.9 → 48.6 ms/iter, AMI = 1.0 throughout); `fig_scaling.png` tracks the O(n)
   reference. HNX-Kumar is the slow baseline (up to ~120 s on K=12).

## Layout
```
hfuse/
  hypergraph.py    unified container (flattened incidence, d, m2)
  operators.py     3 adjacency operators (+ matrix-free apply for SSL views)
  fuse.py          the ONE FUSE engine (line search + Lipschitz + SSL toggle)
  ssl.py           spectral-contrastive regularizer (operator-agnostic)
  baselines.py     Zhou-spectral, HNX-Kumar
  metrics.py       k-means + ARI/AMI/NMI
  datasets.py      real (ARB) loaders + synthetic generators
  experiment.py    dataset × method loop (checkpointed)
  data/            bundled real datasets
run.py             single entry point (experiment + scalability + figures)
outputs/           results.csv, summary.csv, scaling.csv, fig_*.png, run.log
```

Real data provenance: reconstructed from `github.com/nveldt/HyperModularity`
(Chodrow, Veldt & Benson, *Generative hypergraph clustering*, Science Advances
2021), originally from Austin Benson's ARB collection and SocioPatterns.
