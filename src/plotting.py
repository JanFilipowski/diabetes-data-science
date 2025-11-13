import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def cluster_scatter(X, labels, out_path='outputs/cluster_viz.png', method='pca', title=None):
    """
    Visualizacja klastrów z różnymi metodami redukcji wymiaru.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    labels : array-like, shape (n_samples,)
        Etykiety klastrów.
    out_path : str
        Ścieżka do zapisu wykresu (PNG).
    method : str
        'pca'        -> 2D PCA
        'tsne'       -> 2D t-SNE
        'umap'       -> 2D UMAP (wymaga pakietu umap-learn)
        'pca_tsne'   -> subplot: PCA (2D) vs t-SNE (2D)
        'pca_umap'   -> subplot: PCA (2D) vs UMAP (2D)
    title : str or None
        Tytuł główny wykresu / figure.
    """
    method = method.lower()
    unique_labels = np.unique(labels)

    # Upewniamy się, że katalog istnieje
    dir_name = os.path.dirname(out_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # --- helper do rysowania jednego scattera na podanym ax ---
    def _scatter_on_ax(ax, X2, labels, subtitle=None):
        for lab in unique_labels:
            mask = labels == lab
            ax.scatter(
                X2[mask, 0],
                X2[mask, 1],
                s=10,
                label=f'Cluster {lab}',
                alpha=0.8,
                edgecolors='none'
            )
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        if subtitle is not None:
            ax.set_title(subtitle)
        # legenda tylko jeśli liczba klastrów nie zabija czytelności
        if len(unique_labels) <= 12:
            ax.legend(markerscale=2, fontsize=8)

    # --- tryb: pojedynczy embedding (PCA / t-SNE / UMAP) ---
    if method in {'pca', 'tsne', 'umap'}:
        if method == 'pca':
            reducer = PCA(n_components=2)
            default_title = 'PCA (2D) — Clusters'
        elif method == 'tsne':
            reducer = TSNE(
                n_components=2,
                perplexity=50,
                learning_rate='auto',
                init='random',
                random_state=42
            )
            default_title = 't-SNE (2D) — Clusters'
        elif method == 'umap':
            try:
                import umap
            except ImportError as e:
                raise ImportError(
                    "Metoda 'umap' wymaga zainstalowanego pakietu 'umap-learn' "
                    "(pip install umap-learn)."
                ) from e
            reducer = umap.UMAP(n_components=2, random_state=42)
            default_title = 'UMAP (2D) — Clusters'
        else:
            # teoretycznie nieosiągalne
            raise ValueError("Unsupported method")

        X2 = reducer.fit_transform(X)
        plt.figure(figsize=(6, 5))
        _scatter_on_ax(plt.gca(), X2, labels)
        plt.title(title or default_title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    # --- tryb: porównanie kilku metod w jednym obrazku ---
    elif method in {'pca_tsne', 'pca_umap'}:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        ax_left, ax_right = axes

        # PCA zawsze po lewej
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        _scatter_on_ax(ax_left, X_pca, labels, subtitle='PCA (2D)')

        if method == 'pca_tsne':
            tsne = TSNE(
                n_components=2,
                perplexity=50,
                learning_rate='auto',
                init='random',
                random_state=42
            )
            X_tsne = tsne.fit_transform(X)
            _scatter_on_ax(ax_right, X_tsne, labels, subtitle='t-SNE (2D)')
            default_title = 'PCA vs t-SNE — Clusters'
        elif method == 'pca_umap':
            try:
                import umap
            except ImportError as e:
                raise ImportError(
                    "Metoda 'pca_umap' wymaga zainstalowanego pakietu 'umap-learn' "
                    "(pip install umap-learn)."
                ) from e
            umap_model = umap.UMAP(n_components=2, random_state=42)
            X_umap = umap_model.fit_transform(X)
            _scatter_on_ax(ax_right, X_umap, labels, subtitle='UMAP (2D)')
            default_title = 'PCA vs UMAP — Clusters'
        else:
            # też teoretycznie nieosiągalne
            raise ValueError("Unsupported method")

        fig.suptitle(title or default_title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    else:
        raise ValueError(
            "method must be one of: 'pca', 'tsne', 'umap', "
            "'pca_tsne', 'pca_umap'"
        )
