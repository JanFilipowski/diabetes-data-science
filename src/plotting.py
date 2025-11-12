import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def cluster_scatter(X, labels, out_path='outputs/cluster_viz.png', method='pca', title=None):
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        title = title or 'PCA (2D) — Clusters'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=50, learning_rate='auto', init='random', random_state=42)
        title = title or 't-SNE (2D) — Clusters'
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    X2 = reducer.fit_transform(X)

    plt.figure(figsize=(6,5))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        plt.scatter(X2[mask,0], X2[mask,1], s=10, label=f'Cluster {lab}')

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(markerscale=2, fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
