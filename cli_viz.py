import argparse
from pathlib import Path

from src.config import Config
from src.io_utils import artdir, load_npz, load_parquet, run_id_from_cfg
from src.plotting_extra import demographics_bar, feature_heatmap, readmission_bar
from src.plotting_seaborn import (
    cluster_size_distribution,
    correlation_heatmap,
    facetgrid_analysis,
    pairplot_clusters,
    violin_box_plots,
)

METHOD_CHOICES = ("kmeans", "hierarchical", "gmm", "dbscan")


def _label_path(art_dir: Path, method: str, k: int) -> Path:
    if method == "dbscan":
        candidates = [art_dir / "labels_dbscan.npz"]
    else:
        candidates = [art_dir / f"labels_{method}_k{k}.npz", art_dir / f"labels_k{k}.npz"]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find labels for method {method} (searched: {candidates})")

def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in raw.split(',') if token.strip()]


def _default_features(df, candidates, limit=4):
    feat = [c for c in candidates if c in df.columns][:limit]
    if len(feat) < limit:
        inferred = [c for c in df.select_dtypes(include="number").columns if c not in feat]
        feat.extend(inferred[: max(0, limit - len(feat))])
    return feat


def main():
    p = argparse.ArgumentParser(description="Generate legacy + seaborn visualizations for a cached run")
    p.add_argument("--tag", default=None, help="Run tag (used for folder lookup)")
    p.add_argument("--k", type=int, help="Number of clusters/components used")
    p.add_argument("--method", choices=METHOD_CHOICES, default="kmeans")
    p.add_argument("--features", help="Comma separated numeric features for seaborn plots")
    p.add_argument("--violin-kind", choices=("violin", "box"), default="violin")
    p.add_argument("--corr-method", choices=("pearson", "spearman", "kendall"), default="pearson")
    p.add_argument("--facet-cat", default="race", help="Categorical column for FacetGrid (default race)")
    p.add_argument("--facet-num", help="Numeric column for FacetGrid (defaults to first feature)")
    p.add_argument("--facet-col-wrap", type=int, default=3)
    p.add_argument("--skip-basic", action="store_true", help="Skip feature/readmission/demographic charts")
    p.add_argument("--skip-advanced", action="store_true", help="Skip seaborn diagnostics")
    args = p.parse_args()

    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    if args.method != "dbscan" and args.k is None:
        raise ValueError("--k is required for non-DBSCAN methods")

    print(f"[viz] loading cached data for {run_id}…")
    df_eng = load_parquet(A / "df_eng.parquet")
    labels = load_npz(_label_path(A, args.method, args.k or 0))["labels"]
    suffix = f"_{args.method}{'_k'+str(args.k) if args.k is not None else ''}"

    outputs = []

    if not args.skip_basic:
        print("[viz] generating heatmap of numeric features…")
        heat_path = A / f"feature_heatmap{suffix}.png"
        feature_heatmap(df_eng, labels, cfg.numeric_features, heat_path)
        if heat_path.exists():
            outputs.append(str(heat_path))

        print("[viz] generating readmission rate bar chart…")
        readm_path = A / f"readmission_bar{suffix}.png"
        readmission_bar(df_eng, labels, readm_path)
        if readm_path.exists():
            outputs.append(str(readm_path))

        print("[viz] generating demographic distributions…")
        demo_path = A / f"demographics{suffix}.png"
        demographics_bar(df_eng, labels, demo_path)
        if demo_path.exists():
            outputs.append(str(demo_path))

    if not args.skip_advanced:
        features = _parse_csv_list(args.features) or _default_features(df_eng, cfg.numeric_features)
        if features:
            print(f"[viz] seaborn pairplot with features: {features}")
            try:
                outputs.append(str(pairplot_clusters(df_eng, labels, features, A / f"pairplot{suffix}.png")))
            except ValueError as exc:
                print(f"[viz] pairplot skipped: {exc}")

            print("[viz] violin/box plots…")
            try:
                outputs.append(str(
                    violin_box_plots(
                        df_eng,
                        labels,
                        features,
                        A / f"violin_{args.violin_kind}{suffix}.png",
                        plot_type=args.violin_kind,
                    )
                ))
            except ValueError as exc:
                print(f"[viz] violin plots skipped: {exc}")

            print("[viz] correlation heatmaps…")
            try:
                outputs.append(str(
                    correlation_heatmap(
                        df_eng,
                        labels,
                        features,
                        A / f"correlation{suffix}.png",
                        method=args.corr_method,
                    )
                ))
            except ValueError as exc:
                print(f"[viz] correlation skipped: {exc}")
        else:
            print("[viz] no numeric features discovered; skipping seaborn plots")

        facet_num = args.facet_num or (features[0] if features else None)
        if args.facet_cat and facet_num and args.facet_cat in df_eng.columns and facet_num in df_eng.columns:
            print(f"[viz] facet grid for {args.facet_cat} vs {facet_num}")
            try:
                outputs.append(str(
                    facetgrid_analysis(
                        df_eng,
                        labels,
                        args.facet_cat,
                        facet_num,
                        A / f"facetgrid{suffix}.png",
                        col_wrap=args.facet_col_wrap,
                    )
                ))
            except ValueError as exc:
                print(f"[viz] facet grid skipped: {exc}")
        else:
            print("[viz] facet grid skipped (missing --facet-cat/--facet-num or columns absent)")

        try:
            outputs.append(str(cluster_size_distribution(labels, A / f"cluster_sizes{suffix}.png")))
        except ValueError as exc:
            print(f"[viz] cluster size plot skipped: {exc}")

    print(f"[viz] done. Files saved to: {A}")
    if outputs:
        print("[viz] generated: " + ", ".join(outputs))

if __name__ == "__main__":
    main()
