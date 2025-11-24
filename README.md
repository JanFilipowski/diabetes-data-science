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

### Extended CLI Commands
| Command | Purpose | Key Flags |
| --- | --- | --- |
| `python cli.py analyze-demographics` | Full fairness report (chi-square, Cramér's V, stacked/grouped/heatmap plots, cross-tabs) sourced from preserved `df_raw`. | `--method`, `--k`, `--demographic-cols` (`gender,race,age` default), `--out-dir`. |
| `python cli.py feature-importance` | Permutation importance, ANOVA/Kruskal scores, silhouette-per-feature, centroid tables, and heatmaps for a fitted model. | `--method`, `--k`, `--top-k`, `--test` (`anova`/`kruskal`), `--n-repeats`. |
| `python cli.py advanced-viz` | Seaborn diagnostics (pairplot, violin/box, correlation grids, FacetGrid, cluster size bars). | `--pairplot-features`, `--violin-features`, `--corr-features`, `--violin-kind`, `--corr-method`, `--facet-cat`, `--facet-num`. |

### Legacy helper
`python cli_viz.py --method kmeans --k 5 --tag baseline [--features age,avg_glucose_level]` quickly regenerates the original heatmap/readmission/demographics trio and now piggybacks the same seaborn utilities as the advanced CLI. Use `--skip-basic` or `--skip-advanced` to control output volume.

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
