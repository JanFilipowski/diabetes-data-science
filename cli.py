import argparse
import json
from pathlib import Path

import numpy as np

from src.config import Config
from src.data import load_data
from src.preprocess import engineer_features, build_preprocessor
from src.cluster import (
    kmeans_fit_predict,
    hierarchical_fit_predict,
    gmm_fit_predict,
    dbscan_fit_predict,
    plot_dendrogram,
    plot_gmm_probabilities,
    get_gmm_soft_assignments,
)
from src.demographic_analysis import analyze_all_demographics
from src.evaluate import method_specific_scores, profile_by_cluster, save_profiles, cluster_summary
from src.feature_importance import (
    cluster_centroids_analysis,
    feature_contribution_scores,
    permutation_importance_clusters,
    plot_centroid_comparison,
    plot_feature_importance_barplot,
    plot_feature_importance_heatmap,
    silhouette_per_feature,
    top_distinguishing_features,
)
from src.plotting import cluster_scatter
from src.plotting_seaborn import (
    cluster_size_distribution,
    correlation_heatmap,
    facetgrid_analysis,
    pairplot_clusters,
    violin_box_plots,
)
from src.tune import sweep_method
from src.io_utils import run_id_from_cfg, artdir, save_json, save_parquet, load_parquet, save_npz, load_npz, dump_joblib
from src.preprocess import apply_feature_selection

METHOD_CHOICES = ("kmeans", "hierarchical", "gmm", "dbscan")


def _resolve_method(arg_value: str | None, cfg: Config) -> str:
    method = (arg_value or cfg.default_method).lower()
    if method not in METHOD_CHOICES:
        raise ValueError(f"Unsupported method '{method}'.")
    return method


def _model_filename(method: str, k: int | None, suffix: str = "joblib") -> str:
    if method == "dbscan":
        return f"dbscan_model.{suffix}"
    return f"{method}_k{k}.{suffix}"


def _labels_filename(method: str, k: int | None) -> str:
    if method == "dbscan":
        return "labels_dbscan.npz"
    return f"labels_{method}_k{k}.npz"


def _report_filename(method: str, k: int | None) -> str:
    if method == "dbscan":
        return "report_dbscan.json"
    return f"report_{method}_k{k}.json"


def _suffix(method: str, k: int | None) -> str:
    suffix = f"_{method}"
    if k is not None:
        suffix += f"_k{k}"
    return suffix


def _load_labels_array(artifact_dir: Path, method: str, k: int | None) -> np.ndarray:
    file_path = artifact_dir / _labels_filename(method, k)
    data = load_npz(file_path)
    if "labels" not in data:
        raise KeyError(f"{file_path.name} missing 'labels'")
    return data["labels"]


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in raw.split(',') if token.strip()]


def _load_feature_names(artifact_dir: Path, n_features: int) -> list[str]:
    sel_path = artifact_dir / "selected_features.json"
    if sel_path.exists():
        data = json.loads(sel_path.read_text())
        selected = data.get("selected") if isinstance(data, dict) else data
        if isinstance(selected, list) and len(selected) == n_features:
            return selected
        if isinstance(selected, list) and selected:
            print("[feature-importance] selected feature names length mismatch; falling back to generic names")
    feats_path = artifact_dir / "features.json"
    if feats_path.exists():
        data = json.loads(feats_path.read_text())
        names = data.get("selected")
        if isinstance(names, list) and names:
            return names[:n_features]
        numeric = data.get("numeric", [])
        categorical = data.get("categorical", [])
        combined = list(numeric) + list(categorical)
        if combined:
            return (combined + combined)[:n_features]
    return [f"feature_{i}" for i in range(n_features)]

def cmd_prepare(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    print("[prepare] loading + engineering…")
    df = load_data()
    # Save raw dataframe (keeps demographic columns for post-hoc analysis)
    save_parquet(A / "df_raw.parquet", df)
    df_eng = engineer_features(df)

    # Zapisz przetworzony dataframe – już nie będziesz go liczył ponownie
    save_parquet(A / "df_eng.parquet", df_eng)

    # Zbuduj i dopasuj preprocessor, przetransformuj raz na zawsze
    drop_cols = {"readmitted", "readmit_30"}
    feat_cols = [c for c in df_eng.columns if c not in drop_cols]
    preproc, num_feats, cat_feats = build_preprocessor(df_eng[feat_cols], cfg)
    X = preproc.fit_transform(df_eng[feat_cols])

    print("[prepare] applying feature selection...")
    feature_names = list(preproc.get_feature_names_out())
    X_sel, selected_names = apply_feature_selection(X, df_eng, cfg, feature_names)

    if selected_names:
        save_json(A / "selected_features.json", {"selected": selected_names})
        print(f"[prepare] kept {len(selected_names)} / {len(feature_names)} features")
    else:
        print("[prepare] feature selection skipped")

    save_npz(A / "X.npz", X=X_sel)

    dump_joblib(A / "preprocessor.joblib", preproc)
    save_json(A / "features.json", {"numeric": num_feats, "categorical": cat_feats})

    print(f"[prepare] saved: {A}/df_raw.parquet, df_eng.parquet, preprocessor.joblib, X.npz, features.json")

def cmd_tune(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    X = load_npz(A / "X.npz")["X"]
    method = _resolve_method(args.method, cfg)
    print(f"[tune] sweeping parameters for {method}…")

    kwargs: dict = {}
    if method in {"kmeans", "hierarchical", "gmm"}:
        kwargs.update({"k_min": args.k_min, "k_max": args.k_max})
    if method == "kmeans":
        kwargs["random_state"] = cfg.random_state
    elif method == "hierarchical":
        kwargs["linkage"] = args.linkage or cfg.hierarchical_linkage
    elif method == "gmm":
        kwargs.update(
            {
                "covariance_type": args.covariance_type or cfg.gmm_covariance_type,
                "random_state": cfg.random_state,
                "max_iter": cfg.gmm_max_iter,
            }
        )
    elif method == "dbscan":
        eps_vals = args.eps_values or "0.3,0.5,0.7,0.9,1.1"
        min_vals = args.min_samples_values or "10,30,50,70"
        kwargs = {
            "eps_values": [float(v.strip()) for v in eps_vals.split(',') if v.strip()],
            "min_samples_values": [int(v.strip()) for v in min_vals.split(',') if v.strip()],
        }

    sweep_method(X, method=method, out_dir=str(A), **kwargs)
    print(f"[tune] saved sweep artifacts under {A}")

def cmd_fit(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    X = load_npz(A / "X.npz")["X"]
    method = _resolve_method(args.method, cfg)

    params: dict[str, object] = {}
    if method == "kmeans":
        k = args.k or cfg.n_clusters
        params["k"] = k
        labels, model = kmeans_fit_predict(X, n_clusters=k, random_state=cfg.random_state)
        extra_artifacts = []
    elif method == "hierarchical":
        k = args.k or cfg.hierarchical_n_clusters
        linkage = args.linkage or cfg.hierarchical_linkage
        params.update({"k": k, "linkage": linkage})
        labels, model = hierarchical_fit_predict(X, n_clusters=k, linkage_method=linkage)
        dendro = plot_dendrogram(X, A / f"dendrogram_{method}_k{k}.png", method=linkage)
        extra_artifacts = [dendro]
    elif method == "gmm":
        k = args.k or cfg.gmm_n_components
        cov_type = args.covariance_type or cfg.gmm_covariance_type
        params.update({"k": k, "covariance_type": cov_type})
        labels, model = gmm_fit_predict(
            X,
            n_components=k,
            covariance_type=cov_type,
            random_state=cfg.random_state,
            max_iter=cfg.gmm_max_iter,
        )
        prob_plot = plot_gmm_probabilities(X, model, A / f"gmm_probs_k{k}.png")
        save_npz(A / f"gmm_probabilities_k{k}.npz", probabilities=get_gmm_soft_assignments(X, model))
        extra_artifacts = [prob_plot, str(A / f"gmm_probabilities_k{k}.npz")]
    else:  # dbscan
        k = None
        eps = args.eps or cfg.dbscan_eps
        min_samples = args.min_samples or cfg.dbscan_min_samples
        params.update({"eps": eps, "min_samples": min_samples})
        labels, model = dbscan_fit_predict(X, eps=eps, min_samples=min_samples)
        extra_artifacts = []

    scores = method_specific_scores(X, labels, model=model, method=method)
    model_file = A / _model_filename(method, k)
    dump_joblib(model_file, model)
    labels_file = A / _labels_filename(method, k)
    save_npz(labels_file, labels=labels)
    report_payload = {
        "method": method,
        "params": params,
        "scores": scores,
        "n_features": int(X.shape[1]),
    }
    report_file = A / _report_filename(method, k)
    save_json(report_file, report_payload)

    print(f"[fit] method={method}, scores={scores}")
    saved = [model_file, labels_file, report_file] + [Path(p) if isinstance(p, str) else p for p in extra_artifacts]
    print("[fit] saved: " + ", ".join(str(p) for p in saved if p))

def cmd_report(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    method = _resolve_method(args.method, cfg)
    needs_k = method in {"kmeans", "hierarchical", "gmm"}
    if needs_k and args.k is None:
        raise ValueError("--k is required for the selected method.")
    k = args.k if needs_k else None

    df_eng = load_parquet(A / "df_eng.parquet")
    labels = _load_labels_array(A, method, k)
    X = load_npz(A / "X.npz")["X"]

    print(f"[report] generating visualizations for {method}…")

    def _fig_name(prefix: str) -> str:
        suffix = f"_{method}"
        if k is not None:
            suffix += f"_k{k}"
        return str(A / f"{prefix}{suffix}.png")

    umap_ok = True
    try:
        cluster_scatter(X, labels, out_path=_fig_name("umap"), method='umap')
    except ImportError:
        umap_ok = False
        print("[report] skipped UMAP plot (umap-learn not installed)")

    cluster_scatter(X, labels, out_path=_fig_name("pca"), method='pca')
    cluster_scatter(X, labels, out_path=_fig_name("tsne"), method='tsne')
    cluster_scatter(X, labels, out_path=_fig_name("pca_tsne"), method='pca_tsne')

    pca_umap_ok = True
    if umap_ok:
        try:
            cluster_scatter(X, labels, out_path=_fig_name("pca_umap"), method='pca_umap')
        except ImportError:
            pca_umap_ok = False
            print("[report] skipped PCA vs UMAP plot (umap-learn not installed)")
    else:
        pca_umap_ok = False

    numeric_keep = list(cfg.numeric_features) + (
        ["readmit_30"] if "readmit_30" in df_eng.columns else []
    )

    means, readm = profile_by_cluster(df_eng, labels, numeric_keep=numeric_keep)
    save_profiles(means, readm, out_dir=str(A))

    summary_name = f"cluster_summary_{method}{'_k'+str(k) if k is not None else ''}.csv"
    summary = cluster_summary(labels, readmitted=df_eng.get("readmitted"))
    summary.to_csv(A / summary_name, index=False)

    outputs = [
        _fig_name("pca"),
        _fig_name("tsne"),
        _fig_name("pca_tsne"),
    ]
    if umap_ok:
        outputs.append(_fig_name("umap"))
    if pca_umap_ok:
        outputs.append(_fig_name("pca_umap"))
    outputs.extend([
        str(A / "cluster_feature_means.csv"),
        str(A / "readmission_by_cluster.csv"),
        str(A / summary_name),
    ])
    print("[report] saved: " + ", ".join(outputs))

    if getattr(args, "include_demographics", False):
        raw_path = A / "df_raw.parquet"
        if raw_path.exists():
            demo_out = A / "demographics"
            analyze_all_demographics(
                A,
                k or 0,
                demographic_cols=["gender", "race", "age"],
                out_dir=demo_out,
                method=method,
            )
            print(f"[report] demographic analysis saved to {demo_out}")
        else:
            print("[report] df_raw.parquet missing; skipping demographic analysis")


def cmd_analyze_demographics(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    method = _resolve_method(args.method, cfg)
    needs_k = method != "dbscan"
    if needs_k and args.k is None:
        raise ValueError("--k is required for the selected method.")
    k = args.k if needs_k else None

    demo_cols = _parse_csv_list(args.demographic_cols) or ["gender", "race", "age"]
    out_dir = Path(args.out_dir) if args.out_dir else A / f"demographics{_suffix(method, k)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demographics] analyzing {demo_cols} for method={method}, k={k}")
    analyze_all_demographics(A, k or 0, demographic_cols=demo_cols, out_dir=out_dir, method=method)
    print(f"[demographics] saved outputs under {out_dir}")


def cmd_feature_importance(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    method = _resolve_method(args.method, cfg)
    needs_k = method != "dbscan"
    if needs_k and args.k is None:
        raise ValueError("--k is required for the selected method.")
    k = args.k if needs_k else None

    X = load_npz(A / "X.npz")["X"]
    labels = _load_labels_array(A, method, k)
    feature_names = _load_feature_names(A, X.shape[1])

    suffix = _suffix(method, k)
    out_dir = A / f"feature_importance{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    print("[feature-importance] computing permutation/importances…")
    try:
        perm_df = permutation_importance_clusters(
            X,
            labels,
            feature_names,
            n_repeats=args.n_repeats,
            random_state=cfg.random_state,
        )
        perm_path = out_dir / "permutation_importance.csv"
        perm_df.to_csv(perm_path, index=False)
        outputs.append(str(perm_path))
        bar_path = plot_feature_importance_barplot(perm_df, out_dir / "perm_barplot.png", top_k=args.top_k)
        heat_path = plot_feature_importance_heatmap(perm_df, out_dir / "perm_heatmap.png")
        outputs.extend([str(bar_path), str(heat_path)])
    except ValueError as exc:
        print(f"[feature-importance] permutation importance skipped: {exc}")

    print(f"[feature-importance] running {args.test} tests…")
    try:
        contrib_df = feature_contribution_scores(
            X,
            labels,
            feature_names,
            test=args.test,
        )
        contrib_path = out_dir / f"feature_contributions_{args.test}.csv"
        contrib_df.to_csv(contrib_path, index=False)
        outputs.append(str(contrib_path))
    except ValueError as exc:
        print(f"[feature-importance] contribution tests skipped: {exc}")

    try:
        silhouette_df = silhouette_per_feature(X, labels, feature_names)
        silhouette_path = out_dir / "silhouette_per_feature.csv"
        silhouette_df.to_csv(silhouette_path, index=False)
        outputs.append(str(silhouette_path))
    except ValueError as exc:
        print(f"[feature-importance] silhouette per feature skipped: {exc}")

    try:
        top_df = top_distinguishing_features(
            X,
            labels,
            feature_names,
            top_k=args.top_k,
        )
        top_path = out_dir / "top_features.csv"
        top_df.to_csv(top_path, index=False)
        outputs.append(str(top_path))
    except ValueError as exc:
        print(f"[feature-importance] top feature ranking skipped: {exc}")

    try:
        centroids_df = cluster_centroids_analysis(X, labels, feature_names, method=method)
        centroids_path = out_dir / "centroids.csv"
        centroids_df.to_csv(centroids_path, index=False)
        outputs.append(str(centroids_path))
        centroid_plot = plot_centroid_comparison(centroids_df, out_dir / "centroid_heatmap.png")
        outputs.append(str(centroid_plot))
    except ValueError as exc:
        print(f"[feature-importance] centroid analysis skipped: {exc}")

    if outputs:
        print("[feature-importance] saved: " + ", ".join(outputs))
    else:
        print("[feature-importance] no outputs generated (check logs)")


def cmd_advanced_viz(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    method = _resolve_method(args.method, cfg)
    needs_k = method != "dbscan"
    if needs_k and args.k is None:
        raise ValueError("--k is required for the selected method.")
    k = args.k if needs_k else None

    df_eng = load_parquet(A / "df_eng.parquet")
    labels = _load_labels_array(A, method, k)
    suffix = _suffix(method, k)
    out_dir = A / f"advanced_viz{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []

    pair_features = _parse_csv_list(args.pairplot_features)
    if pair_features:
        try:
            path = pairplot_clusters(df_eng, labels, pair_features, out_dir / "pairplot.png")
            outputs.append(str(path))
        except ValueError as exc:
            print(f"[advanced-viz] pairplot skipped: {exc}")

    violin_features = _parse_csv_list(args.violin_features)
    if violin_features:
        try:
            path = violin_box_plots(
                df_eng,
                labels,
                violin_features,
                out_dir / "violin.png",
                plot_type=args.violin_kind,
            )
            outputs.append(str(path))
        except ValueError as exc:
            print(f"[advanced-viz] violin/box plots skipped: {exc}")

    corr_features = _parse_csv_list(args.corr_features)
    if corr_features:
        try:
            path = correlation_heatmap(
                df_eng,
                labels,
                corr_features,
                out_dir / "correlation.png",
                method=args.corr_method,
            )
            outputs.append(str(path))
        except ValueError as exc:
            print(f"[advanced-viz] correlation heatmaps skipped: {exc}")

    if args.facet_cat and args.facet_num:
        try:
            path = facetgrid_analysis(
                df_eng,
                labels,
                args.facet_cat,
                args.facet_num,
                out_dir / "facetgrid.png",
                col_wrap=args.facet_col_wrap,
            )
            outputs.append(str(path))
        except ValueError as exc:
            print(f"[advanced-viz] facetgrid skipped: {exc}")
    elif args.facet_cat or args.facet_num:
        print("[advanced-viz] facetgrid requires both --facet-cat and --facet-num")

    try:
        size_path = cluster_size_distribution(labels, out_dir / "cluster_sizes.png")
        outputs.append(str(size_path))
    except ValueError as exc:
        print(f"[advanced-viz] cluster size plot skipped: {exc}")

    if outputs:
        print("[advanced-viz] saved: " + ", ".join(outputs))
    else:
        print("[advanced-viz] nothing generated; check feature arguments")

def main():
    p = argparse.ArgumentParser(description="Diabetes clustering CLI with caching")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Load, engineer, fit preprocessor, cache X/df")
    sp.add_argument("--tag", default=None, help="Optional run tag")
    sp.set_defaults(fn=cmd_prepare)

    sp = sub.add_parser("tune", help="Sweep hyperparameters using cached X")
    sp.add_argument("--k-min", type=int, default=2)
    sp.add_argument("--k-max", type=int, default=8)
    sp.add_argument("--tag", default=None)
    sp.add_argument("--method", choices=METHOD_CHOICES, default=None)
    sp.add_argument("--linkage", help="Linkage for hierarchical sweep")
    sp.add_argument("--covariance-type", help="Covariance type for GMM sweep")
    sp.add_argument("--eps-values", help="Comma-separated eps grid for DBSCAN sweep")
    sp.add_argument("--min-samples-values", help="Comma-separated min_samples grid for DBSCAN sweep")
    sp.set_defaults(fn=cmd_tune)

    sp = sub.add_parser("fit", help="Fit a clustering model on cached X and save artifacts")
    sp.add_argument("--k", type=int, help="Override k")
    sp.add_argument("--tag", default=None)
    sp.add_argument("--method", choices=METHOD_CHOICES, default=None)
    sp.add_argument("--linkage", help="Hierarchical linkage method (ward, complete, average, single)")
    sp.add_argument("--covariance-type", help="GMM covariance type (full, tied, diag, spherical)")
    sp.add_argument("--eps", type=float, help="DBSCAN eps")
    sp.add_argument("--min-samples", type=int, help="DBSCAN min_samples")
    sp.set_defaults(fn=cmd_fit)

    sp = sub.add_parser("report", help="Make plots/tables from cached labels")
    sp.add_argument("--k", type=int, help="Number of clusters/components (if required by method)")
    sp.add_argument("--tag", default=None)
    sp.add_argument("--method", choices=METHOD_CHOICES, default=None)
    sp.add_argument("--include-demographics", action="store_true", help="Attach demographic bias analysis to the report")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("analyze-demographics", help="Deep dive into demographic fairness metrics")
    sp.add_argument("--k", type=int, help="Number of clusters/components (if required by method)")
    sp.add_argument("--tag", default=None)
    sp.add_argument("--method", choices=METHOD_CHOICES, default=None)
    sp.add_argument("--demographic-cols", help="Comma separated list of demographic columns")
    sp.add_argument("--out-dir", help="Optional override for output directory")
    sp.set_defaults(fn=cmd_analyze_demographics)

    sp = sub.add_parser("feature-importance", help="Quantify and visualize feature influence per cluster")
    sp.add_argument("--k", type=int, help="Number of clusters/components (if required by method)")
    sp.add_argument("--tag", default=None)
    sp.add_argument("--method", choices=METHOD_CHOICES, default=None)
    sp.add_argument("--top-k", type=int, default=15, help="Top features to highlight in plots/tables")
    sp.add_argument("--test", choices=("anova", "kruskal"), default="anova")
    sp.add_argument("--n-repeats", type=int, default=10, help="Permutation repeats for importance estimation")
    sp.set_defaults(fn=cmd_feature_importance)

    sp = sub.add_parser("advanced-viz", help="Generate seaborn-heavy exploratory visualizations")
    sp.add_argument("--k", type=int, help="Number of clusters/components (if required by method)")
    sp.add_argument("--tag", default=None)
    sp.add_argument("--method", choices=METHOD_CHOICES, default=None)
    sp.add_argument("--pairplot-features", help="Comma separated features for pairplot")
    sp.add_argument("--violin-features", help="Comma separated features for violin/box plots")
    sp.add_argument("--violin-kind", choices=("violin", "box"), default="violin")
    sp.add_argument("--corr-features", help="Comma separated features for correlation heatmaps")
    sp.add_argument("--corr-method", choices=("pearson", "spearman", "kendall"), default="pearson")
    sp.add_argument("--facet-cat", help="Categorical feature for FacetGrid overview")
    sp.add_argument("--facet-num", help="Numeric feature for FacetGrid overview")
    sp.add_argument("--facet-col-wrap", type=int, default=3)
    sp.set_defaults(fn=cmd_advanced_viz)

    args = p.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
