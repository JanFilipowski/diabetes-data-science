from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

PathLike = Union[str, Path]


def _ensure_output_path(path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def kmeans_fit_predict(X: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km

def dbscan_fit_predict(X: np.ndarray, eps: float = 0.7, min_samples: int = 50) -> Tuple[np.ndarray, DBSCAN]:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, db


def hierarchical_fit_predict(
    X: np.ndarray,
    n_clusters: int,
    linkage_method: str = "ward",
) -> Tuple[np.ndarray, AgglomerativeClustering]:
    """Fit Agglomerative clustering and return labels/model."""

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    return labels, model


def gmm_fit_predict(
    X: np.ndarray,
    n_components: int,
    covariance_type: str = "full",
    random_state: int = 42,
    max_iter: int = 100,
) -> Tuple[np.ndarray, GaussianMixture]:
    """Fit Gaussian Mixture Model and return labels/model."""

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=max_iter,
    )
    labels = gmm.fit_predict(X)
    return labels, gmm


def plot_dendrogram(
    X: np.ndarray,
    out_path: PathLike,
    method: str = "ward",
    max_samples: int = 1000,
    random_state: int = 42,
) -> Path:
    """Create and save a dendrogram for a (possibly subsampled) dataset."""

    X = np.asarray(X)
    if len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        data = X[idx]
    else:
        data = X
    Z = linkage(data, method=method)
    out_file = _ensure_output_path(out_path)
    plt.figure(figsize=(10, 4))
    dendrogram(Z, truncate_mode="lastp", p=30, no_labels=True)
    plt.title("Hierarchical clustering dendrogram")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return out_file


def plot_gmm_probabilities(
    X: np.ndarray,
    gmm_model: GaussianMixture,
    out_path: PathLike,
    n_samples: int = 500,
    random_state: int = 42,
) -> Path:
    """Plot stacked probabilities for a sample of points."""

    probs = gmm_model.predict_proba(X)
    rng = np.random.default_rng(random_state)
    if probs.shape[0] > n_samples:
        idx = np.sort(rng.choice(probs.shape[0], size=n_samples, replace=False))
    else:
        idx = np.arange(probs.shape[0])
    sampled = probs[idx]
    palette = sns.color_palette("Set2", sampled.shape[1])
    bottoms = np.zeros(sampled.shape[0])
    out_file = _ensure_output_path(out_path)

    plt.figure(figsize=(10, 4))
    x = np.arange(sampled.shape[0])
    for k in range(sampled.shape[1]):
        plt.bar(x, sampled[:, k], bottom=bottoms, color=palette[k], label=f"Cluster {k}")
        bottoms += sampled[:, k]
    plt.xlabel("Sample (subset)")
    plt.ylabel("Probability")
    plt.title("GMM soft assignments")
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return out_file


def get_gmm_soft_assignments(X: np.ndarray, gmm_model: GaussianMixture) -> np.ndarray:
    """Return the probability matrix produced by a fitted GMM instance."""

    return gmm_model.predict_proba(X)
