import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score


def clustering_scores(X: np.ndarray, labels: np.ndarray, method: str = 'kmeans') -> Dict[str, Optional[float]]:
    labels = np.asarray(labels)
    X = np.asarray(X)
    if method == 'dbscan':
        mask = labels >= 0
        if mask.sum() < 2:
            return {'silhouette': None, 'davies_bouldin': None}
        eval_X, eval_labels = X[mask], labels[mask]
    else:
        eval_X, eval_labels = X, labels
    unique = np.unique(eval_labels)
    if unique.size <= 1:
        return {'silhouette': None, 'davies_bouldin': None}
    return {
        'silhouette': float(silhouette_score(eval_X, eval_labels)),
        'davies_bouldin': float(davies_bouldin_score(eval_X, eval_labels))
    }


def gmm_information_criteria(X: np.ndarray, gmm_model) -> Dict[str, float]:
    return {'bic': float(gmm_model.bic(X)), 'aic': float(gmm_model.aic(X))}


def dbscan_cluster_stats(labels: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels)
    noise = labels == -1
    clusters = labels[~noise]
    n_clusters = int(len(np.unique(clusters))) if clusters.size else 0
    n_noise = int(noise.sum())
    noise_pct = float(n_noise / len(labels) * 100) if len(labels) else 0.0
    return {'n_clusters': n_clusters, 'n_noise': n_noise, 'noise_pct': noise_pct}


def method_specific_scores(
    X: np.ndarray,
    labels: np.ndarray,
    model=None,
    method: str = 'kmeans'
) -> Dict[str, Optional[float]]:
    results = clustering_scores(X, labels, method=method)
    if method == 'gmm' and model is not None:
        results.update(gmm_information_criteria(X, model))
    if method == 'dbscan':
        results.update(dbscan_cluster_stats(labels))
    return results

# evaluate.py
def profile_by_cluster(df_eng, labels, numeric_keep):
    df = df_eng.copy()
    df['cluster'] = labels
    keep = [c for c in numeric_keep if c in df.columns]
    means = df.groupby('cluster')[keep].mean().sort_index()
    if 'readmitted' in df.columns:
        readm = df.groupby('cluster')['readmitted'] \
                  .value_counts(normalize=True).rename('ratio').reset_index()
    else:
        readm = pd.DataFrame()
    return means, readm


def save_profiles(means, readm, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    means.to_csv(os.path.join(out_dir, 'cluster_feature_means.csv'))
    if not readm.empty:
        pivot = readm.pivot(index='cluster', columns='readmitted', values='ratio').fillna(0)
        pivot.to_csv(os.path.join(out_dir, 'readmission_by_cluster.csv'))

def cluster_summary(labels, readmitted=None):
    import pandas as pd, numpy as np
    s = pd.Series(labels, name='cluster')
    out = s.value_counts(dropna=False).sort_index().rename('count').to_frame()
    out['percent'] = (out['count'] / len(s) * 100).round(2)
    if readmitted is not None:
        ct = pd.crosstab(s, readmitted, normalize='index') * 100
        for col in ['<30','>30','No']:
            if col not in ct.columns: ct[col] = 0.0
        out = out.join(ct[['<30','>30','No']].round(2))
        out = out.rename(columns={'<30':'pct_<30','>30':'pct_>30','No':'pct_No'})
    return out.reset_index()
