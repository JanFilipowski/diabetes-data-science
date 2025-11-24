import itertools
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score

from .cluster import (
    dbscan_fit_predict,
    gmm_fit_predict,
    hierarchical_fit_predict,
    kmeans_fit_predict,
)


def _score_labels(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels)
    mask = labels >= 0
    if mask.any() and np.unique(labels[mask]).size > 1:
        eval_labels = labels[mask]
        eval_X = X[mask]
    else:
        eval_labels = labels
        eval_X = X
    if np.unique(eval_labels).size <= 1:
        return float("nan"), float("nan")
    return (
        float(silhouette_score(eval_X, eval_labels)),
        float(davies_bouldin_score(eval_X, eval_labels)),
    )


def sweep_kmeans(X, k_min=2, k_max=8, random_state=42, out_dir='outputs'):
    rows = []
    for k in range(k_min, k_max + 1):
        labels, _ = kmeans_fit_predict(X, n_clusters=k, random_state=random_state)
        sil, db = _score_labels(X, labels)
        rows.append({'k': k, 'silhouette': sil, 'davies_bouldin': db})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'kmeans_sweep.csv'), index=False)

    plt.figure(); plt.plot(df['k'], df['silhouette'], marker='o')
    plt.xlabel('k'); plt.ylabel('Silhouette'); plt.title('KMeans silhouette vs k')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'kmeans_sweep_silhouette.png'), dpi=200); plt.close()

    plt.figure(); plt.plot(df['k'], df['davies_bouldin'], marker='o')
    plt.xlabel('k'); plt.ylabel('Davies–Bouldin (lower)'); plt.title('KMeans DB vs k')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'kmeans_sweep_db.png'), dpi=200); plt.close()
    return df


def sweep_hierarchical(
    X,
    k_min=2,
    k_max=8,
    linkage='ward',
    out_dir='outputs',
):
    rows = []
    for k in range(k_min, k_max + 1):
        labels, _ = hierarchical_fit_predict(X, n_clusters=k, linkage_method=linkage)
        sil, db = _score_labels(X, labels)
        rows.append({'k': k, 'silhouette': sil, 'davies_bouldin': db})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'hierarchical_sweep.csv'), index=False)

    plt.figure(); plt.plot(df['k'], df['silhouette'], marker='o')
    plt.xlabel('k'); plt.ylabel('Silhouette'); plt.title('Hierarchical silhouette vs k')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'hierarchical_sweep_silhouette.png'), dpi=200); plt.close()

    plt.figure(); plt.plot(df['k'], df['davies_bouldin'], marker='o')
    plt.xlabel('k'); plt.ylabel('Davies–Bouldin (lower)'); plt.title('Hierarchical DB vs k')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'hierarchical_sweep_db.png'), dpi=200); plt.close()
    return df


def sweep_gmm(
    X,
    k_min=2,
    k_max=8,
    covariance_type='full',
    random_state=42,
    max_iter=100,
    out_dir='outputs',
):
    rows = []
    for k in range(k_min, k_max + 1):
        labels, model = gmm_fit_predict(
            X,
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter,
        )
        sil, db = _score_labels(X, labels)
        rows.append({
            'k': k,
            'silhouette': sil,
            'davies_bouldin': db,
            'bic': float(model.bic(X)),
            'aic': float(model.aic(X)),
        })
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'gmm_sweep.csv'), index=False)

    plt.figure(); plt.plot(df['k'], df['silhouette'], marker='o')
    plt.xlabel('k'); plt.ylabel('Silhouette'); plt.title('GMM silhouette vs components')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'gmm_sweep_silhouette.png'), dpi=200); plt.close()

    plt.figure(); plt.plot(df['k'], df['davies_bouldin'], marker='o')
    plt.xlabel('k'); plt.ylabel('Davies–Bouldin (lower)'); plt.title('GMM DB vs components')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'gmm_sweep_db.png'), dpi=200); plt.close()

    plt.figure(); plt.plot(df['k'], df['bic'], marker='o', label='BIC')
    plt.plot(df['k'], df['aic'], marker='o', label='AIC')
    plt.xlabel('k'); plt.ylabel('Information Criterion (lower better)'); plt.title('GMM BIC/AIC')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'gmm_sweep_ic.png'), dpi=200); plt.close()
    return df


def sweep_dbscan(
    X,
    eps_values: Iterable[float] = (0.3, 0.5, 0.7, 0.9, 1.1),
    min_samples_values: Iterable[int] = (10, 30, 50, 70),
    out_dir='outputs',
):
    rows = []
    for eps, min_samples in itertools.product(eps_values, min_samples_values):
        labels, _ = dbscan_fit_predict(X, eps=eps, min_samples=min_samples)
        valid = labels >= 0
        n_clusters = len(np.unique(labels[valid])) if valid.any() else 0
        n_noise = int(np.sum(~valid))
        sil, db = _score_labels(X, labels)
        rows.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': sil,
            'davies_bouldin': db,
        })
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'dbscan_sweep.csv'), index=False)

    pivot = df.pivot(index='min_samples', columns='eps', values='silhouette')
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Silhouette'})
    plt.title('DBSCAN silhouette heatmap')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'dbscan_sweep_heatmap.png'), dpi=200); plt.close()
    return df


def sweep_method(X, method='kmeans', out_dir='outputs', **kwargs):
    method = method.lower()
    if method == 'kmeans':
        return sweep_kmeans(X, out_dir=out_dir, **kwargs)
    if method == 'hierarchical':
        return sweep_hierarchical(X, out_dir=out_dir, **kwargs)
    if method == 'gmm':
        return sweep_gmm(X, out_dir=out_dir, **kwargs)
    if method == 'dbscan':
        return sweep_dbscan(X, out_dir=out_dir, **kwargs)
    raise ValueError(f"Unknown method '{method}'")
