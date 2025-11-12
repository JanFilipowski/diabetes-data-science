import os, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

def sweep_k(X, k_min=2, k_max=8, random_state=42, out_dir='outputs'):
    rows = []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            sil, db = float('nan'), float('nan')
        else:
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
        rows.append({'k': k, 'silhouette': sil, 'davies_bouldin': db})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'k_sweep.csv'), index=False)

    plt.figure(); plt.plot(df['k'], df['silhouette'], marker='o')
    plt.xlabel('k'); plt.ylabel('Silhouette'); plt.title('Silhouette vs k')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'k_sweep_silhouette.png'), dpi=200); plt.close()

    plt.figure(); plt.plot(df['k'], df['davies_bouldin'], marker='o')
    plt.xlabel('k'); plt.ylabel('Davies–Bouldin (lower)'); plt.title('DB vs k')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'k_sweep_db.png'), dpi=200); plt.close()
    return df
