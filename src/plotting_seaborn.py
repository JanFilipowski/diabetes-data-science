"""Advanced seaborn visualizations tailored for the clustering workflow.

This module complements the existing plotting helpers by providing richer
cluster-aware diagnostics (pairplots, violin/box distributions, correlation
heatmaps, FacetGrid explorations, and cluster size summaries).
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterable, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("notebook")

PathLike = Union[str, Path]


def _ensure_output_path(out_path: PathLike) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_features(df: pd.DataFrame, features: Sequence[str]) -> list[str]:
    if not features:
        raise ValueError("`features` must contain at least one column name.")
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataframe: {missing}")
    return list(features)


def _prepare_plot_frame(df: pd.DataFrame, labels: Sequence, columns: Sequence[str]) -> pd.DataFrame:
    if len(df) != len(labels):
        raise ValueError("Length of labels must match dataframe length.")
    feats = _ensure_features(df, columns)
    frame = df.loc[:, feats].copy()
    frame["cluster"] = np.asarray(labels)
    return frame


def pairplot_clusters(
    df: pd.DataFrame,
    labels: Sequence,
    features: Sequence[str],
    out_path: PathLike,
    palette: str = "Set2",
) -> Path:
    """Generate a seaborn pairplot colored by cluster membership.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (typically engineered features).
    labels : Sequence
        Cluster assignments aligned with ``df`` rows.
    features : Sequence[str]
        Numeric features to visualize pairwise.
    out_path : PathLike
        Destination path for the saved figure.
    palette : str, default "Set2"
        Seaborn palette name for cluster hues.
    """

    data = _prepare_plot_frame(df, labels, features)
    out_file = _ensure_output_path(out_path)

    g = sns.pairplot(
        data=data,
        vars=features,
        hue="cluster",
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 20},
        corner=False,
    )
    g.fig.suptitle("Pairwise feature relationships by cluster", y=1.02)
    g.fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    return out_file


def violin_box_plots(
    df: pd.DataFrame,
    labels: Sequence,
    features: Sequence[str],
    out_path: PathLike,
    plot_type: str = "violin",
) -> Path:
    """Visualize feature distributions per cluster using violin or box plots."""

    plot_type = plot_type.lower()
    if plot_type not in {"violin", "box"}:
        raise ValueError("plot_type must be 'violin' or 'box'.")

    data = _prepare_plot_frame(df, labels, features)
    out_file = _ensure_output_path(out_path)

    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]

    palette = "muted" if plot_type == "violin" else "pastel"
    for ax, feature in zip(axes, features):
        if plot_type == "violin":
            sns.violinplot(
                data=data,
                x="cluster",
                y=feature,
                palette=palette,
                inner="quartile",
                ax=ax,
            )
        else:
            sns.boxplot(
                data=data,
                x="cluster",
                y=feature,
                palette=palette,
                ax=ax,
            )
        ax.set_title(f"{feature} distribution by cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(feature)
        for label in ax.get_xticklabels():
            label.set_rotation(15)
            label.set_horizontalalignment("right")

    plt.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_file


def correlation_heatmap(
    df: pd.DataFrame,
    labels: Sequence,
    features: Sequence[str],
    out_path: PathLike,
    method: str = "pearson",
) -> Path:
    """Plot cluster-specific correlation heatmaps for selected features."""

    data = _prepare_plot_frame(df, labels, features)
    clusters = sorted(data["cluster"].unique())
    if not clusters:
        raise ValueError("No clusters provided for correlation heatmap.")

    out_file = _ensure_output_path(out_path)
    n_clusters = len(clusters)
    n_cols = 2
    n_rows = ceil(n_clusters / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)
    annot = len(features) <= 10

    for ax in axes[n_clusters:]:
        ax.axis("off")

    for ax, cluster in zip(axes, clusters):
        cluster_df = data.loc[data["cluster"] == cluster, features]
        if len(cluster_df) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_axis_off()
            continue
        corr = cluster_df.corr(method=method)
        sns.heatmap(
            corr,
            ax=ax,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            annot=annot,
            fmt=".2f",
            cbar=ax is axes[0],
            cbar_kws={"label": "Correlation"} if ax is axes[0] else None,
        )
        ax.set_title(f"Cluster {cluster} correlation")

    plt.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_file


def facetgrid_analysis(
    df: pd.DataFrame,
    labels: Sequence,
    cat_feature: str,
    num_feature: str,
    out_path: PathLike,
    col_wrap: int = 3,
) -> Path:
    """Explore categorical vs numeric relationships inside each cluster."""

    if cat_feature not in df.columns:
        raise ValueError(f"Missing categorical feature: {cat_feature}")
    if num_feature not in df.columns:
        raise ValueError(f"Missing numeric feature: {num_feature}")

    data = df[[cat_feature, num_feature]].copy()
    if len(data) != len(labels):
        raise ValueError("Length of labels must match dataframe length.")
    data["cluster"] = np.asarray(labels)
    out_file = _ensure_output_path(out_path)

    grid = sns.FacetGrid(
        data=data,
        col="cluster",
        col_wrap=max(1, col_wrap),
        height=4,
        aspect=1.2,
        sharey=False,
    )

    def _boxplot(data, color, **kws):
        sns.boxplot(
            data=data,
            x=cat_feature,
            y=num_feature,
            palette="Set3",
        )

    grid.map_dataframe(_boxplot)
    for ax in grid.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_horizontalalignment("right")
    grid.fig.suptitle(f"{num_feature} by {cat_feature} across clusters", y=1.02)
    plt.tight_layout()
    grid.figure.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)
    return out_file


def cluster_size_distribution(labels: Sequence, out_path: PathLike) -> Path:
    """Plot the number (and proportion) of samples assigned to each cluster."""

    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        raise ValueError("`labels` cannot be empty.")

    out_file = _ensure_output_path(out_path)
    counts = pd.Series(labels_arr, name="cluster").value_counts().sort_index()
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="deep", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Cluster size distribution")
    for idx, (count, pct) in enumerate(zip(counts.values, percentages.values)):
        ax.text(idx, count + counts.max() * 0.01, f"{pct:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_file
