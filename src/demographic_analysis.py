"""Utilities for evaluating demographic distributions versus cluster assignments.

The helpers in this module load the preserved ``df_raw.parquet`` artifacts,
attach cluster labels, and provide statistical/visual summaries (chi-square,
Cramér's V, entropy, cross-tabulations) to audit fairness and bias.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, entropy

from src.io_utils import load_npz, load_parquet

sns.set_style("whitegrid")

PathLike = Union[str, Path]


def safe_json_dump(obj: Any, path: PathLike) -> None:
    """Dump JSON handling numpy/scalar types gracefully."""

    def convert(value: Any):
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def _resolve_artifacts_dir(artifacts_dir: PathLike) -> Path:
    path = Path(artifacts_dir)
    if path.exists():
        return path
    candidate = Path("artifacts") / path
    if candidate.exists():
        return candidate
    return path  # let downstream file I/O raise informative errors


def _label_path(run_path: Path, method: str, k: int) -> Path:
    candidates = []
    if method == "dbscan":
        candidates.append(run_path / "labels_dbscan.npz")
    else:
        candidates.append(run_path / f"labels_{method}_k{k}.npz")
        candidates.append(run_path / f"labels_k{k}.npz")  # legacy fallback
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No labels file found for method '{method}' (searched: {candidates})")


def load_raw_with_clusters(artifacts_dir: PathLike, k: int, method: str = "kmeans") -> pd.DataFrame:
    """Load ``df_raw.parquet`` and attach cluster labels for the given method/k."""

    run_path = _resolve_artifacts_dir(artifacts_dir)
    df_raw_path = run_path / "df_raw.parquet"
    labels_path = _label_path(run_path, method, k)

    if not df_raw_path.exists():
        raise FileNotFoundError(f"Missing df_raw.parquet at {df_raw_path}")

    df_raw = load_parquet(df_raw_path)
    labels_dict = load_npz(labels_path)
    if "labels" not in labels_dict:
        raise KeyError(f"{labels_path.name} does not contain 'labels'")
    labels = labels_dict["labels"]

    if len(df_raw) != len(labels):
        raise ValueError("Length mismatch between df_raw and labels array")

    df = df_raw.copy()
    df["cluster"] = labels
    return df


def chi_square_test(
    df: pd.DataFrame,
    demographic_col: str,
    cluster_col: str = "cluster",
) -> dict[str, Any]:
    """Run chi-square test between a demographic column and cluster labels."""

    if demographic_col not in df.columns:
        raise ValueError(f"Column '{demographic_col}' not found in dataframe")
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in dataframe")

    contingency = pd.crosstab(df[demographic_col], df[cluster_col])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    r, c = contingency.shape
    denom = min(r - 1, c - 1)
    cramers_v = float(np.sqrt(chi2 / (n * denom))) if denom > 0 else 0.0

    if cramers_v >= 0.5:
        association = "strong"
    elif cramers_v >= 0.3:
        association = "moderate"
    elif cramers_v >= 0.1:
        association = "weak"
    else:
        association = "negligible"

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "significant": bool(p_value < 0.05),
        "cramers_v": cramers_v,
        "association": association,
        "contingency_table": contingency.to_dict(),
        "expected_freq": expected.tolist(),
    }


def demographic_distribution_plot(
    df: pd.DataFrame,
    demographic_col: str,
    cluster_col: str = "cluster",
    out_path: PathLike | None = None,
    plot_type: str = "stacked_bar",
) -> plt.Figure | None:
    """Visualize demographic distribution by cluster using several plot styles."""

    if demographic_col not in df.columns:
        raise ValueError(f"Column '{demographic_col}' not found in dataframe")
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in dataframe")

    plot_type = plot_type.lower()
    palette_map = {
        "gender": "Set2",
        "race": "tab10",
        "age": "viridis",
    }
    palette = palette_map.get(demographic_col, "Set2")

    fig, ax = plt.subplots(figsize=(7, 5))
    if plot_type == "stacked_bar":
        counts = pd.crosstab(df[cluster_col], df[demographic_col])
        pct = counts.divide(counts.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(counts))
        colors = sns.color_palette(palette, len(pct.columns))
        for color, category in zip(colors, pct.columns):
            values = pct[category].values
            ax.bar(counts.index.astype(str), values, bottom=bottom, label=category, color=color)
            for idx, (b, v) in enumerate(zip(bottom, values)):
                if v > 4:
                    ax.text(idx, b + v / 2, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9)
            bottom += values
        ax.set_ylabel("Percentage")
    elif plot_type == "grouped_bar":
        sns.countplot(data=df, x=cluster_col, hue=demographic_col, palette=palette, ax=ax)
        ax.set_ylabel("Count")
        for bar in ax.patches:
            height = bar.get_height()
            if height:
                ax.annotate(f"{height:.0f}", (bar.get_x() + bar.get_width() / 2, height),
                            ha="center", va="bottom", fontsize=8)
    elif plot_type == "heatmap":
        counts = pd.crosstab(df[demographic_col], df[cluster_col])
        sns.heatmap(
            counts,
            annot=True,
            fmt="d",
            cmap="viridis",
            cbar_kws={"label": "Count"},
            ax=ax,
        )
    else:
        raise ValueError("plot_type must be one of {'stacked_bar','grouped_bar','heatmap'}")

    ax.set_title(f"Distribution of {demographic_col} by cluster")
    ax.set_xlabel("Cluster")
    ax.legend(title=demographic_col.capitalize(), loc="best") if plot_type != "heatmap" else None
    plt.tight_layout()

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def cross_tabulation_summary(
    df: pd.DataFrame,
    demographic_cols: Iterable[str],
    cluster_col: str = "cluster",
    out_dir: PathLike | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Create count and percentage cross-tabulations for demographics vs clusters."""

    results: dict[str, dict[str, pd.DataFrame]] = {}
    out_dir_path = Path(out_dir) if out_dir else None
    if out_dir_path:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    for col in demographic_cols:
        if col not in df.columns:
            continue
        counts = pd.crosstab(df[col], df[cluster_col])
        pct_by_cluster = pd.crosstab(df[col], df[cluster_col], normalize="columns") * 100
        pct_by_demo = pd.crosstab(df[col], df[cluster_col], normalize="index") * 100
        results[col] = {
            "counts": counts,
            "pct_by_cluster": pct_by_cluster,
            "pct_by_demographic": pct_by_demo,
        }
        if out_dir_path:
            counts.to_csv(out_dir_path / f"{col}_counts.csv")
            pct_by_cluster.round(2).to_csv(out_dir_path / f"{col}_pct_by_cluster.csv")
            pct_by_demo.round(2).to_csv(out_dir_path / f"{col}_pct_by_demographic.csv")
    return results


def demographic_statistical_summary(
    df: pd.DataFrame,
    demographic_cols: Iterable[str],
    cluster_col: str = "cluster",
) -> dict[str, Mapping[str, Any]]:
    """Compute descriptive statistics (mode, entropy, numeric moments) per cluster."""

    summary: dict[str, dict[str, Any]] = defaultdict(dict)
    grouped = df.groupby(cluster_col)

    for col in demographic_cols:
        if col not in df.columns:
            continue
        col_summary: dict[str, Any] = {}
        for cluster, group in grouped:
            values = group[col].dropna()
            entry: dict[str, Any] = {
                "counts": values.value_counts().to_dict(),
                "mode": values.mode().tolist(),
            }
            probs = values.value_counts(normalize=True)
            entry["entropy"] = float(entropy(probs, base=2)) if not probs.empty else 0.0
            if pd.api.types.is_numeric_dtype(values):
                entry["mean"] = float(values.mean()) if not values.empty else None
                entry["median"] = float(values.median()) if not values.empty else None
                entry["std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            col_summary[str(cluster)] = entry
        summary[col] = col_summary
    return summary


def analyze_all_demographics(
    artifacts_dir: PathLike,
    k: int,
    demographic_cols: Sequence[str] | None = None,
    out_dir: PathLike | None = None,
    method: str = "kmeans",
) -> dict[str, Any]:
    """Run the full demographic analysis pipeline for the specified run."""

    demographic_cols = demographic_cols or ["gender", "race", "age"]
    df = load_raw_with_clusters(artifacts_dir, k, method=method)
    output_dir = Path(out_dir) if out_dir else _resolve_artifacts_dir(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"chi_square": {}, "plots": {}, "cross_tabulations": {}, "stats": {}}

    for col in demographic_cols:
        if col not in df.columns:
            continue
        chi_res = chi_square_test(df, col)
        report["chi_square"][col] = chi_res
        print(
            f"[demographics] {col}: chi2={chi_res['chi2']:.3f}, p={chi_res['p_value']:.4f}"
            f" ({'significant' if chi_res['significant'] else 'ns'})"
        )
        plot_paths = {}
        for plot_type in ("stacked_bar", "grouped_bar", "heatmap"):
            path = output_dir / f"{col}_{plot_type}.png"
            demographic_distribution_plot(df, col, out_path=path, plot_type=plot_type)
            plot_paths[plot_type] = str(path)
        report["plots"][col] = plot_paths

    report["cross_tabulations"] = {
        col: {name: df_.round(2) for name, df_ in tables.items()}
        for col, tables in cross_tabulation_summary(
            df, demographic_cols, out_dir=output_dir / "tables"
        ).items()
    }
    report["stats"] = demographic_statistical_summary(df, demographic_cols)

    report_path = output_dir / f"demographic_report_k{k}.json"
    safe_json_dump(report, report_path)
    print(f"[demographics] saved report to {report_path}")
    return report


def compare_demographics_across_runs(
    run_ids: Sequence[PathLike],
    k: int,
    demographic_col: str = "gender",
    out_path: PathLike | None = None,
) -> pd.DataFrame:
    """Compare demographic percentages across multiple artifact runs."""

    records = []
    for run in run_ids:
        run_id = str(run)
        run_path = _resolve_artifacts_dir(run)
        if not run_path.exists():
            raise FileNotFoundError(f"Artifacts directory not found for run '{run_id}'")
        df = load_raw_with_clusters(run_path, k)
        if demographic_col not in df.columns:
            continue
        pct = pd.crosstab(df["cluster"], df[demographic_col], normalize="index") * 100
        for cluster, row in pct.iterrows():
            for category, value in row.items():
                records.append(
                    {
                        "run": run_id,
                        "cluster": cluster,
                        demographic_col: category,
                        "percentage": value,
                    }
                )

    comparison_df = pd.DataFrame(records)
    if comparison_df.empty:
        raise ValueError("No demographic data available for comparison.")

    g = sns.catplot(
        data=comparison_df,
        x="cluster",
        y="percentage",
        hue=demographic_col,
        col="run",
        kind="bar",
        col_wrap=3,
        sharey=True,
        palette="Set2",
        height=4,
    )
    g.fig.suptitle(f"{demographic_col.capitalize()} distribution across runs", y=1.02)

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
    return comparison_df
