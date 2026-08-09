"""Unified dataset registry -- real hypergraphs (primary) + synthetic (control).

REAL (bundled under hfuse/data/, no network needed; reconstructed from the
Chodrow-Veldt-Benson / Austin-Benson ARB releases):
  * primary_school   242 nodes, 11 classes  (SocioPatterns face-to-face contact;
                     hyperedges = groups in proximity; label = classroom)
  * high_school      327 nodes,  9 classes  (same, high school)
  * senate_committees ~282 nodes, 2 parties (US Senate; hyperedge = committee;
                     label = party)
  * house_committees  ~1290 nodes, 2 parties (US House; committee memberships)

SYNTHETIC (controlled signal + scalability):
  * planted_partition(n, K)       balanced hypergraph-SBM
  * hierarchical_nested(n, ...)   two-level community hierarchy

Every loader returns a Hypergraph with labels['community'] as an integer array
of ground-truth community ids, and meta['K'] = number of communities.
"""
from __future__ import annotations
import os
import numpy as np
from .hypergraph import Hypergraph

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# real: contact hypergraphs (ARB temporal-simplex format)
# --------------------------------------------------------------------------- #
def _load_contact(prefix, meta_name):
    nverts = np.loadtxt(os.path.join(DATA, f"{prefix}-nverts.txt"), dtype=int)
    simp = np.loadtxt(os.path.join(DATA, f"{prefix}-simplices.txt"), dtype=int)
    # reconstruct simplices from the flattened stream
    offs = np.concatenate([[0], np.cumsum(nverts)])
    raw = [simp[offs[i]:offs[i + 1]] for i in range(len(nverts))]

    # nodemap: new_id (1..n) <-> original_id  (header line present)
    nodemap = {}
    with open(os.path.join(DATA, f"{prefix}-nodemap.txt")) as f:
        for line in f:
            parts = line.split()
            if parts[0] == "new_id":
                continue
            nodemap[int(parts[1])] = int(parts[0])   # original -> new (1-based)

    # metadata: original_id \t class \t gender  -> class label per original id
    orig_class = {}
    with open(os.path.join(DATA, meta_name)) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            orig_class[int(parts[0])] = parts[1]

    # class label per new_id (1..n)
    classes = sorted(set(orig_class.values()))
    cls2int = {c: i for i, c in enumerate(classes)}
    n = len(nodemap)
    labels = np.full(n, -1, dtype=int)
    for orig, new in nodemap.items():
        if orig in orig_class:
            labels[new - 1] = cls2int[orig_class[orig]]
    # any unlabeled node -> its own singleton class
    nxt = len(classes)
    for i in range(n):
        if labels[i] < 0:
            labels[i] = nxt
            nxt += 1

    # unique hyperedges of size >= 2 (0-based node ids)
    seen = set()
    edges = []
    for s in raw:
        if len(s) < 2:
            continue
        key = tuple(sorted(int(x) - 1 for x in s))     # simplices are 1-based
        if key not in seen:
            seen.add(key)
            edges.append(np.array(key, dtype=np.int64))

    K = len(set(labels.tolist()))
    return Hypergraph(n, edges, labels={"community": labels},
                      meta={"name": prefix, "K": K, "source": "ARB/SocioPatterns"})


def primary_school():
    return _load_contact("primary", "primary-metadata.txt")


def high_school():
    return _load_contact("high", "high-metadata.txt")


# --------------------------------------------------------------------------- #
# real: congress committee hypergraphs (CSV: party,id,committee,session,...,new_id)
# --------------------------------------------------------------------------- #
def _load_committees(csv_name):
    import csv
    rows = []
    with open(os.path.join(DATA, csv_name)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # node = new_id (1-based); label = party; edge = (committee, session) group
    party = {}
    groups = {}
    maxid = 0
    for r in rows:
        try:
            nid = int(r["new_id"])
        except (KeyError, ValueError):
            continue
        maxid = max(maxid, nid)
        party[nid] = int(r["party"])
        key = (r["committee"], r["session"])
        groups.setdefault(key, set()).add(nid)
    n = maxid
    labels = np.zeros(n, dtype=int)
    parties = sorted(set(party.values()))
    p2i = {p: i for i, p in enumerate(parties)}
    for nid, p in party.items():
        labels[nid - 1] = p2i[p]
    edges = [np.array(sorted(v), dtype=np.int64) - 1 for v in groups.values()
             if len(v) >= 2]
    K = len(parties)
    return Hypergraph(n, edges, labels={"community": labels},
                      meta={"name": csv_name.replace(".csv", ""), "K": K,
                            "source": "Chodrow-Veldt-Benson / Congress"})


def senate_committees():
    return _load_committees("senate_committees.csv")


def house_committees():
    return _load_committees("house_committees.csv")


# --------------------------------------------------------------------------- #
# synthetic (controlled + scalable)
# --------------------------------------------------------------------------- #
def _mixed_size_edges(rng, verts, target, fracs=((2, 0.4), (3, 0.35), (4, 0.25))):
    edges = []
    verts = np.asarray(verts)
    for size, frac in fracs:
        per = len(verts) // size
        if per == 0:
            continue
        rounds = int(np.ceil(target * frac / per))
        for _ in range(rounds):
            perm = rng.permutation(verts)[:per * size].reshape(per, size)
            edges.extend(list(perm))
    return edges


def planted_partition(n=1000, K=6, seed=0, density=3.0, bridge_frac=0.03):
    rng = np.random.default_rng(seed)
    comm = np.repeat(np.arange(K), int(np.ceil(n / K)))[:n]
    verts = [np.where(comm == c)[0] for c in range(K)]
    edges = []
    for c in range(K):
        edges += _mixed_size_edges(rng, verts[c], int(density * len(verts[c])))
    n_bridge = int(bridge_frac * len(edges))
    for _ in range(n_bridge):
        cs = rng.choice(K, size=2, replace=False)
        e = np.array([rng.choice(verts[cs[0]]), rng.choice(verts[cs[1]])])
        edges.append(e)
    return Hypergraph(n, edges, labels={"community": comm},
                      meta={"name": "planted_partition", "K": K})


def hierarchical_nested(n=2400, n_macro=4, sub_per_macro=3, seed=3,
                        leaf_density=3.0, within_macro_frac=0.15):
    rng = np.random.default_rng(seed)
    n_leaf = n_macro * sub_per_macro
    leaf = np.repeat(np.arange(n_leaf), int(np.ceil(n / n_leaf)))[:n]
    macro = leaf // sub_per_macro
    lv = [np.where(leaf == l)[0] for l in range(n_leaf)]
    edges = []
    for l in range(n_leaf):
        edges += _mixed_size_edges(rng, lv[l], int(leaf_density * len(lv[l])))
    n_wm = int(within_macro_frac * len(edges))
    for _ in range(n_wm):
        m = rng.integers(n_macro)
        subs = rng.choice(sub_per_macro, size=2, replace=False)
        l1, l2 = m * sub_per_macro + subs[0], m * sub_per_macro + subs[1]
        e = np.array([rng.choice(lv[l1]), rng.choice(lv[l2])])
        edges.append(e)
    return Hypergraph(n, edges, labels={"community": leaf, "macro": macro},
                      meta={"name": "hierarchical_nested", "K": n_leaf, "macro_K": n_macro})


REAL = {
    "primary_school": primary_school,
    "high_school": high_school,
    "senate_committees": senate_committees,
    "house_committees": house_committees,
}
SYNTH = {
    "planted_partition": lambda: planted_partition(n=1000, K=6, seed=0),
    "hierarchical_nested": lambda: hierarchical_nested(n=2400, seed=3),
}
ALL = {**REAL, **SYNTH}
