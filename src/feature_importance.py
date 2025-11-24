"""Feature importance utilities for clustering workflows.

The functions in this module operate on cached design matrices (``X``),
cluster assignments, and feature metadata to quantify which variables drive
separation between clusters. Visual helpers are also provided to create
publication-ready figures saved inside run-specific artifact directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, kruskal
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")

PathLike = Union[str, Path]


def _ensure_output_path(out_path: PathLike) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_feature_names(feature_names: Optional[Sequence[str]], n_features: int) -> list[str]:
    if feature_names is None:
        return [f"feature_{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names length does not match X shape")
    return list(feature_names)


def _valid_mask(labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(labels)
    return labels >= 0


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    unique = np.unique(labels)
    if unique.size <= 1:
        return None
    return float(silhouette_score(X, labels))


def permutation_importance_clusters(
    X: np.ndarray,
    labels: Sequence[int],
    feature_names: Optional[Sequence[str]] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate feature importance via silhouette drops after column permutations."""

    X = np.asarray(X)
    labels = np.asarray(labels)
    mask = _valid_mask(labels)
    if mask.sum() <= 1:
        raise ValueError("Not enough labeled samples for permutation importance.")
    X_masked = X[mask]
    labels_masked = labels[mask]
    baseline = _silhouette_safe(X_masked, labels_masked)
    if baseline is None:
        raise ValueError("Silhouette undefined (need at least two clusters).")

    names = _coerce_feature_names(feature_names, X.shape[1])
    rng = np.random.default_rng(random_state)
    records = []
    for idx, name in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_masked.copy()
            rng.shuffle(X_perm[:, idx])
            score = _silhouette_safe(X_perm, labels_masked)
            if score is None:
                continue
            drops.append(baseline - score)
        if drops:
            records.append({
                "feature": name,
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            })
    return pd.DataFrame(records).sort_values("importance", ascending=False).reset_index(drop=True)


def cluster_centroids_analysis(
    X: np.ndarray,
    labels: Sequence[int],
    feature_names: Optional[Sequence[str]] = None,
    method: str = "kmeans",
    model: Optional[object] = None,
) -> pd.DataFrame:
    """Compute representative feature values per cluster (centroids/means)."""

    X = np.asarray(X)
    labels = np.asarray(labels)
    names = _coerce_feature_names(feature_names, X.shape[1])
    df = pd.DataFrame(X, columns=names)
    df["cluster"] = labels
    valid = df[df["cluster"] >= 0]
    if valid.empty:
        raise ValueError("No valid cluster assignments for centroid analysis.")

    rows = []
    grouped = valid.groupby("cluster")
    for cluster, group in grouped:
        stats = group[names].agg(["mean", "std"]).T
        stats.columns = ["centroid", "std"]
        stats = stats.reset_index().rename(columns={"index": "feature"})
        for _, row in stats.iterrows():
            rows.append({
                "cluster": int(cluster),
                "feature": row["feature"],
                "centroid": float(row["centroid"]),
                "std": float(row["std"]),
                "cluster_size": int(len(group)),
            })
    return pd.DataFrame(rows)


def feature_contribution_scores(
    X: np.ndarray,
    labels: Sequence[int],
    feature_names: Optional[Sequence[str]] = None,
    test: str = "anova",
) -> pd.DataFrame:
    """Quantify per-feature contribution via ANOVA or Kruskal-Wallis tests."""

    X = np.asarray(X)
    labels = np.asarray(labels)
    names = _coerce_feature_names(feature_names, X.shape[1])
    mask = _valid_mask(labels)
    X_valid = X[mask]
    labels_valid = labels[mask]

    grouped_indices = [np.where(labels_valid == c)[0] for c in np.unique(labels_valid)]
    grouped_indices = [idx for idx in grouped_indices if idx.size > 1]
    if len(grouped_indices) < 2:
        raise ValueError("Statistical test requires at least two clusters with >=2 samples.")

    records = []
    for idx, name in enumerate(names):
        samples = [X_valid[group_idx, idx] for group_idx in grouped_indices]
        if test == "anova":
            score, p_val = f_oneway(*samples)
            test_name = "anova"
        else:
            score, p_val = kruskal(*samples)
            test_name = "kruskal"
        records.append({
            "feature": name,
            "score": float(score),
            "p_value": float(p_val),
            "test": test_name,
        })
    return pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)


def silhouette_per_feature(
    X: np.ndarray,
    labels: Sequence[int],
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Evaluate silhouette score when each feature is used individually."""

    X = np.asarray(X)
    labels = np.asarray(labels)
    names = _coerce_feature_names(feature_names, X.shape[1])
    mask = _valid_mask(labels)
    X_valid = X[mask]
    labels_valid = labels[mask]

    records = []
    for idx, name in enumerate(names):
        col = X_valid[:, idx : idx + 1]
        score = _silhouette_safe(col, labels_valid)
        records.append({"feature": name, "silhouette": score})
    return pd.DataFrame(records).sort_values("silhouette", ascending=False).reset_index(drop=True)


def top_distinguishing_features(
    X: np.ndarray,
    labels: Sequence[int],
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Combine multiple importance signals to identify leading features."""

    perm_df = permutation_importance_clusters(X, labels, feature_names)
    contrib_df = feature_contribution_scores(X, labels, feature_names)
    merged = perm_df.merge(contrib_df, on="feature", how="outer")
    merged["rank_perm"] = merged["importance"].rank(ascending=False, method="min")
    merged["rank_score"] = merged["score"].rank(ascending=False, method="min")
    merged["mean_rank"] = merged[["rank_perm", "rank_score"]].mean(axis=1)
    return merged.sort_values("mean_rank").head(top_k).reset_index(drop=True)


def plot_feature_importance_heatmap(
    importance_df: pd.DataFrame,
    out_path: PathLike,
    title: str = "Feature importance",
) -> Path:
    """Render a heatmap based on feature importance values."""

    if importance_df.empty:
        raise ValueError("importance_df cannot be empty")
    importance_df = importance_df.set_index("feature")
    out_file = _ensure_output_path(out_path)

    plt.figure(figsize=(max(6, len(importance_df) * 0.4), 4))
    sns.heatmap(
        importance_df[[col for col in importance_df.columns if importance_df[col].dtype != object]],
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Importance"},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return out_file


def plot_feature_importance_barplot(
    importance_df: pd.DataFrame,
    out_path: PathLike,
    top_k: int = 15,
) -> Path:
    """Draw a horizontal bar plot of the top-k features."""

    if importance_df.empty:
        raise ValueError("importance_df cannot be empty")
    subset = importance_df.nlargest(top_k, "importance")
    out_file = _ensure_output_path(out_path)

    plt.figure(figsize=(8, max(4, 0.35 * len(subset))))
    ax = sns.barplot(
        data=subset,
        x="importance",
        y="feature",
        orient="h",
        palette="viridis",
    )
    if "std" in subset.columns:
        ax.errorbar(
            subset["importance"],
            np.arange(len(subset)),
            xerr=subset["std"],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    ax.set_title("Top feature importances")
    ax.set_xlabel("Silhouette drop (higher = more important)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return out_file


def plot_centroid_comparison(
    centroids_df: pd.DataFrame,
    out_path: PathLike,
    title: str = "Centroid comparison",
) -> Path:
    """Visualize centroids/means for each feature and cluster."""

    if centroids_df.empty:
        raise ValueError("centroids_df cannot be empty")
    pivot = centroids_df.pivot_table(
        index="feature",
        columns="cluster",
        values="centroid",
    )
    out_file = _ensure_output_path(out_path)

    plt.figure(figsize=(max(6, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.3)))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0,
        annot=True if pivot.shape[0] <= 20 else False,
        fmt=".2f",
        cbar_kws={"label": "Value"},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    return out_file
