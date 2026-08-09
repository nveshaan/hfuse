#!/usr/bin/env python3
"""Unified FUSE hypergraph-modularity benchmark -- single entry point.

Runs ONE experiment: every dataset (real + synthetic) x every method
(FUSE under 3 adjacency operators, each with/without SSL, + 2 baselines),
then a scalability sweep of the primary (Banerjee) FUSE on a synthetic
generator, and writes a clean results table + figures + log.

    python run.py                 # full run
    python run.py --quick         # smaller synthetic sizes / fewer scaling points

Outputs (./outputs):
    results.csv          one row per (dataset, method)
    summary.csv          AMI pivot (dataset x method)
    scaling.csv          Banerjee-FUSE fit time / AMI vs n
    fig_recovery.png     AMI per dataset x method (real datasets highlighted)
    fig_ssl_effect.png   AMI gain from SSL, per operator per dataset
    fig_scaling.png      fit time vs n (log-log)
    run.log
"""
from __future__ import annotations
import argparse, os, time, json, logging, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfuse import datasets as D
from hfuse import operators as OP, fuse as F, metrics as MET
from hfuse.experiment import run_all

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def get_logger():
    logger = logging.getLogger("hfuse"); logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(OUT, "run.log")); fh.setFormatter(fmt)
    ch = logging.StreamHandler(); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def scalability(logger, quick=False):
    logger.info("=" * 70)
    logger.info("SCALABILITY: Banerjee FUSE on planted_partition, n sweep")
    grid = [1000, 5000, 20000] if quick else [50000, 100000, 500000]
    rows = []
    for n in grid:
        H = D.planted_partition(n=n, K=6, seed=0)
        op = OP.Operator(H, "banerjee")
        t0 = time.perf_counter()
        S, info = F.fuse_embedding(op, k=16, n_iters=150, seed=1)
        fit = time.perf_counter() - t0
        lab = MET.kmeans_labels(S, 6, 1)
        ami = MET.recovery(H.labels["community"], lab)["AMI"]
        rows.append(dict(n=n, n_edges=H.n_edges, fit_s=fit,
                         ms_per_iter=fit / max(info["iters"], 1) * 1000,
                         iters=info["iters"], Q=info["Q"], AMI=ami))
        logger.info(f"    n={n:6d} |E|={H.n_edges:7d} fit={fit:6.2f}s "
                    f"{rows[-1]['ms_per_iter']:5.1f}ms/it AMI={ami:.3f}")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "scaling.csv"), index=False)
    return df


def make_figures(df, sdf, logger):
    real = list(D.REAL.keys())
    methods = ([f"FUSE[{k}]" for k in OP.OPERATOR_KINDS] +
               [f"FUSE[{k}]+SSL" for k in OP.OPERATOR_KINDS] +
               ["HG-Spectral(Zhou07)", "HNX-Kumar(KPT)"])
    datasets = [d for d in df["dataset"].unique()]

    # 1) recovery grouped bars per dataset
    piv = df.pivot_table(index="dataset", columns="method", values="AMI").reindex(datasets)[methods]
    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 4, 5))
    x = np.arange(len(datasets)); w = 0.1
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(methods):
        ax.bar(x + (i - len(methods) / 2) * w, piv[m].values, w,
               label=m, color=cmap(i % 10))
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AMI"); ax.set_ylim(0, 1.0)
    ax.set_title("Community recovery (AMI): FUSE operators ± SSL vs baselines", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_recovery.png"), dpi=130); plt.close(fig)

    # 2) SSL effect per operator per dataset
    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 3, 4.6))
    for i, kind in enumerate(OP.OPERATOR_KINDS):
        base = df[df.method == f"FUSE[{kind}]"].set_index("dataset")["AMI"]
        withssl = df[df.method == f"FUSE[{kind}]+SSL"].set_index("dataset")["AMI"]
        delta = (withssl - base).reindex(datasets)
        ax.bar(np.arange(len(datasets)) + (i - 1) * 0.25, delta.values, 0.25,
               label=f"{kind}", color=cmap(i))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(datasets))); ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AMI gain from SSL"); ax.set_title("Effect of SSL (Δ AMI), per operator", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ssl_effect.png"), dpi=130); plt.close(fig)

    # 3) scaling
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(sdf["n"], sdf["fit_s"], "o-", lw=2, label="Banerjee FUSE (fit time)")
    n0, t0 = sdf["n"].iloc[0], sdf["fit_s"].iloc[0]
    ax.plot(sdf["n"], t0 * sdf["n"] / n0, "k--", alpha=0.5, label="O(n) reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (vertices)"); ax.set_ylabel("fit time (s)")
    ax.set_title("Scalability: Banerjee FUSE (planted_partition)", fontweight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    ax2 = ax.twinx(); ax2.plot(sdf["n"], sdf["AMI"], "s--", color="tab:green", alpha=0.7)
    ax2.set_ylabel("AMI", color="tab:green"); ax2.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scaling.png"), dpi=130); plt.close(fig)
    logger.info("Saved figures: fig_recovery.png, fig_ssl_effect.png, fig_scaling.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    logger = get_logger()
    logger.info("#" * 70)
    logger.info("UNIFIED FUSE HYPERGRAPH-MODULARITY BENCHMARK")
    logger.info("FUSE operators: banerjee (primary), clique, normalized | SSL toggle uniform")
    logger.info("Datasets: 4 REAL (primary/high school, senate/house committees) + 2 synthetic")
    logger.info("#" * 70)

    datasets = dict(D.ALL)
    df = run_all(datasets, logger, n_iters=(120 if args.quick else 200))
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    df.pivot_table(index="dataset", columns="method", values="AMI").to_csv(
        os.path.join(OUT, "summary.csv"))
    logger.info(f"Saved results.csv ({len(df)} rows) and summary.csv")

    sdf = scalability(logger, quick=args.quick)
    make_figures(df, sdf, logger)

    # headline: best method per REAL dataset
    logger.info("=" * 70)
    logger.info("HEADLINE (best method by AMI, real datasets):")
    for d in D.REAL:
        sub = df[df.dataset == d].sort_values("AMI", ascending=False)
        if len(sub):
            top = sub.iloc[0]
            logger.info(f"  {d:20s} -> {top['method']:24s} AMI={top['AMI']:.3f}")
    logger.info("DONE.")
    return df


if __name__ == "__main__":
    main()
