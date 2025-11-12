from typing import Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

def kmeans_fit_predict(X: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km

def dbscan_fit_predict(X: np.ndarray, eps: float = 0.7, min_samples: int = 50) -> Tuple[np.ndarray, DBSCAN]:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, db
