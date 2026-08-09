"""The ONE unified experiment.

For every dataset (real + synthetic) we run every method and record recovery
(ARI/AMI/NMI), the Banerjee modularity of the resulting partition (a common
yardstick across all methods), and fit time.

Methods (8):
  FUSE[banerjee|clique|normalized]  x  SSL {off, on}     -> 6 FUSE variants
  HG-Spectral(Zhou07),  HNX-Kumar(KPT)                   -> 2 baselines

So FUSE is tested under 3 adjacency operators, each with and without the SAME
spectral-contrastive SSL regularizer (SSL is a uniform toggle, not special-cased).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from . import operators as OP
from . import fuse as F
from . import baselines as B
from . import metrics as MET

SSL_CFG = dict(ssl_views=3, ssl_drop=0.2, ssl_lambda=1.0, ssl_seed=7)


def _q_banerjee(op_ban, labels):
    """Banerjee modularity of a hard partition (common yardstick)."""
    labels = np.asarray(labels)
    K = labels.max() + 1
    Z = np.zeros((len(labels), K))
    Z[np.arange(len(labels)), labels] = 1.0
    return float(op_ban.modularity(Z))


def run_all(datasets, logger, k=16, n_iters=200, seed=1, checkpoint_csv=None):
    rows = []
    for dname, loader in datasets.items():
        H = loader()
        y = H.labels["community"]
        K = H.meta["K"]
        op_ban = OP.Operator(H, "banerjee")            # shared yardstick + reused
        ops = {"banerjee": op_ban,
               "clique": OP.Operator(H, "clique"),
               "normalized": OP.Operator(H, "normalized")}
        logger.info(f"=== {dname}: {H} | K={K} | source={H.meta.get('source','synthetic')} ===")

        def record(method, labels, fit_s):
            r = MET.recovery(y, labels)
            q = _q_banerjee(op_ban, labels)
            rows.append(dict(dataset=dname, n=H.n, n_edges=H.n_edges,
                             avg_edge=H.avg_edge_size, K=K, method=method,
                             is_fuse=method.startswith("FUSE"),
                             ssl=("+SSL" in method), fit_s=fit_s,
                             Q_banerjee=q, **r))
            logger.info(f"    {method:26s} AMI={r['AMI']:.3f} ARI={r['ARI']:.3f} "
                        f"Q={q:.3f} t={fit_s:.2f}s")

        # FUSE variants x SSL toggle
        for kind, op in ops.items():
            for use_ssl in (False, True):
                t0 = time.perf_counter()
                S, info = F.fuse_embedding(op, k=k, n_iters=n_iters, seed=seed,
                                           ssl=use_ssl, **(SSL_CFG if use_ssl else {}))
                lab = MET.kmeans_labels(S, K, seed)
                fit = time.perf_counter() - t0
                record(f"FUSE[{kind}]" + ("+SSL" if use_ssl else ""), lab, fit)

        # baselines
        for bname, bfn in (("HG-Spectral(Zhou07)", B.zhou_spectral),
                           ("HNX-Kumar(KPT)", B.hnx_kumar)):
            try:
                t0 = time.perf_counter()
                lab, _ = bfn(H, K, seed)
                record(bname, lab, time.perf_counter() - t0)
            except Exception as ex:
                logger.error(f"    {bname} FAILED: {ex}")

        if checkpoint_csv:
            pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)

    return pd.DataFrame(rows)
