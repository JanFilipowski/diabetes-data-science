# Diabetes Readmission — Clustering Workbench

Toolkit for exploring the **UCI Diabetes 130-US Hospitals** dataset with multiple clustering methods, cached design matrices, fairness diagnostics, and publication-ready visualizations.

## Highlights
- **Single CLI** (`cli.py`) orchestrates caching, tuning, fitting, reporting, and deep-dive analyses.
- Supports **KMeans, Agglomerative, Gaussian Mixture, DBSCAN** with dedicated tuning parameters.
- Keeps both `df_raw.parquet` and engineered `df_eng.parquet` so downstream tools can access demographic columns without recomputing pipelines.
- Built-in modules for **demographic bias checks**, **feature attribution**, **advanced seaborn plots**, and **GMM probability exports**.
- Artifacts stored per run inside `artifacts/<run_id>`; reruns reuse cached matrices to keep iterations fast.

## Installation
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
`src/data.py` downloads the dataset automatically via `ucimlrepo` unless you drop `diabetic_data.csv` under `data/raw/`.

## Core Workflow (cli.py)
```powershell
# 1) Materialize raw + engineered frames, transformer, and X matrix
python cli.py prepare --tag baseline

# 2) Sweep your preferred method
python cli.py tune --method kmeans --k-min 3 --k-max 8 --tag baseline

# 3) Fit + save labels/report/model
python cli.py fit --method kmeans --k 5 --tag baseline

# 4) Generate cluster report (+UMAP/PCA/t-SNE, summaries, optional demographics)
python cli.py report --method kmeans --k 5 --tag baseline --include-demographics
```

Each command honors `--tag` (default pulled from `src/config.py`). Run IDs are hashed from configuration to keep artifacts organized.

### CLI Reference
| Command | Primary Role | Notable Flags |
| --- | --- | --- |
| `python cli.py prepare` | Load data, save `df_raw`/`df_eng`, fit preprocessor, emit `X.npz`, feature metadata. | `--tag` (namespaces artifacts). |
| `python cli.py tune` | Grid-sweep clustering hyperparameters using cached `X`. Saves sweep CSV/plots per method. | `--method`, `--k-min/--k-max`, `--linkage`, `--covariance-type`, `--eps-values`, `--min-samples-values`. |
| `python cli.py fit` | Train the selected clustering model, persist labels, model, score report, and method-specific extras (e.g., dendrograms, GMM probs). | `--method`, `--k`, `--linkage`, `--covariance-type`, `--eps`, `--min-samples`. |
| `python cli.py report` | Produce PCA/t-SNE/UMAP plots, cluster summaries, readmission tables, and optional demographic bundle. | `--method`, `--k`, `--include-demographics`. |
| `python cli.py analyze-demographics` | Full fairness suite: chi-square/Cramér’s V, stacked/grouped/heatmap plots, cross-tabs, stats written under `demographics*/`. | `--method`, `--k`, `--demographic-cols`, `--out-dir`. |
| `python cli.py feature-importance` | Permutation importance, ANOVA/Kruskal tests, silhouette-per-feature, centroid tables + visualizations. | `--method`, `--k`, `--top-k`, `--test`, `--n-repeats`. |
| `python cli.py advanced-viz` | Seaborn-driven diagnostics (pairplots, violin/box, correlation grids, FacetGrid, cluster-size bars) saved under `advanced_viz*/`. | `--method`, `--k`, `--pairplot-features`, `--violin-features`, `--violin-kind`, `--corr-features`, `--corr-method`, `--facet-cat`, `--facet-num`, `--facet-col-wrap`. |
| `python cli_viz.py` | Convenience script for the legacy heatmap/readmission/demographic trio plus optional seaborn plots. | `--method`, `--k`, `--tag`, `--features`, `--violin-kind`, `--corr-method`, `--facet-cat`, `--facet-num`, `--skip-basic`, `--skip-advanced`. |

All commands obey `--tag`, so you can maintain multiple experiment tracks concurrently. Methods default to the value in `src/config.py`, but overriding `--method` lets every tool operate on KMeans, hierarchical, GMM, or DBSCAN artifacts explicitly.

## Artifact Layout
After `cli.py prepare` the folder `artifacts/<run_id>` contains:

| File | Description |
| --- | --- |
| `df_raw.parquet` | Untouched dataframe (keeps demographic columns for fairness review). |
| `df_eng.parquet` | Engineered frame (encoded + feature-selected columns). |
| `preprocessor.joblib` | Fitted column transformer/pipeline. |
| `X.npz` | Cached feature matrix (post-selection). |
| `features.json` | Names of numeric/categorical columns pre-selection. |
| `selected_features.json` | Ordered list of kept feature names (if selection active). |
| `labels_<method>_k<k>.npz` | Cluster assignments from `cli.py fit`. |
| `report_<method>_k<k>.json` | Scores + params for the fitted run. |
| `cluster_feature_means.csv`, `readmission_by_cluster.csv`, `cluster_summary_<method>.csv` | Tables produced by `cli.py report`. |
| `demographics*/` | Outputs from `--include-demographics` or `analyze-demographics`. |
| `feature_importance*/` | CSVs + plots from `feature-importance`. |
| `advanced_viz*/` | Seaborn figures from `advanced-viz` (pairplots, violin, correlation, etc.). |

## Visualization Modules
- `src/plotting.py` — dimensionality reduction scatter plots (UMAP/PCA/t-SNE combos).
- `src/plotting_seaborn.py` — pairplots, violin/box, correlation heatmaps, FacetGrid, and cluster size summaries.
- `src/plotting_extra.py` — legacy heatmap + readmission/demographic bars used by `cli_viz.py`.
- `src/demographic_analysis.py` — chi-square, cross-tabs, and multi-plot fairness reporting.
- `src/feature_importance.py` — permutation/ANOVA/Kruskal importance, centroid pivots, and convenience plotting helpers.

## Tips
- Re-run `cli.py fit` whenever you change the method or `k`; artifacts are namespaced, so multiple models can coexist under the same tag.
- DBSCAN does not require `--k`, but other methods do. The CLI guards against missing arguments.
- If UMAP is unavailable, the report command automatically falls back to PCA/t-SNE plots and logs the skipped step.
- Keep an eye on `selected_features.json`; `feature-importance` will fall back to generic names if selection metadata is missing.
