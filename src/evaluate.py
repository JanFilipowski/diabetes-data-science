import os
import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.metrics import silhouette_score, davies_bouldin_score

def clustering_scores(X: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    unique = np.unique(labels[labels >= 0])
    if unique.size <= 1:
        return {'silhouette': None, 'davies_bouldin': None}
    return {
        'silhouette': float(silhouette_score(X, labels)),
        'davies_bouldin': float(davies_bouldin_score(X, labels))
    }

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
