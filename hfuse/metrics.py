"""Clustering of embeddings + recovery metrics (ARI/AMI/NMI)."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score)


def kmeans_labels(emb, K, seed=0):
    n = emb.shape[0]
    if n > 20000:
        km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=5, batch_size=2048)
    else:
        km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    return km.fit_predict(emb)


def recovery(y_true, y_pred):
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "AMI": float(adjusted_mutual_info_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
    }
