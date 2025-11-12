import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import Config
from src.data import load_data
from src.preprocess import engineer_features, build_preprocessor
from src.cluster import kmeans_fit_predict
from src.evaluate import clustering_scores, profile_by_cluster, save_profiles, cluster_summary
from src.plotting import cluster_scatter
from src.tune import sweep_k
from src.io_utils import run_id_from_cfg, artdir, save_json, save_parquet, load_parquet, save_npz, load_npz, dump_joblib, load_joblib
from src.preprocess import apply_feature_selection

def cmd_prepare(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    print("[prepare] loading + engineering…")
    df = load_data()
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

    print(f"[prepare] saved: {A}/df_eng.parquet, preprocessor.joblib, X.npz, features.json")

def cmd_tune(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    X = load_npz(A / "X.npz")["X"]
    print("[tune] sweeping k…")
    sweep_k(X, k_min=args.k_min, k_max=args.k_max, random_state=cfg.random_state, out_dir=str(A))
    print(f"[tune] saved: {A}/k_sweep.csv, k_sweep_silhouette.png, k_sweep_db.png")

def cmd_fit(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    X = load_npz(A / "X.npz")["X"]
    k = args.k if args.k else cfg.n_clusters

    print(f"[fit] KMeans k={k}…")
    labels, km = kmeans_fit_predict(X, n_clusters=k, random_state=cfg.random_state)
    scores = clustering_scores(X, labels)

    # zapisz model + etykiety + metryki
    dump_joblib(A / f"kmeans_k{k}.joblib", km)
    save_npz(A / f"labels_k{k}.npz", labels=labels)
    save_json(A / f"report_k{k}.json", {"method": "KMeans", "k": k, "scores": scores, "n_features": int(X.shape[1])})

    print(f"[fit] silhouette={scores['silhouette']}, DB={scores['davies_bouldin']}")
    print(f"[fit] saved: {A}/kmeans_k{k}.joblib, labels_k{k}.npz, report_k{k}.json")

def cmd_report(args):
    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    df_eng = load_parquet(A / "df_eng.parquet")
    labels = load_npz(A / f"labels_k{args.k}.npz")["labels"]
    X = load_npz(A / "X.npz")["X"]

    print("[report] generating visualizations...")

    # PCA – szybka, interpretowalna
    cluster_scatter(X, labels, out_path=str(A / f"pca_kmeans_k{args.k}.png"), method='pca')

    # t-SNE – efektowna, nieliniowa
    cluster_scatter(X, labels, out_path=str(A / f"tsne_kmeans_k{args.k}.png"), method='tsne')

    # Profile i podsumowania
    numeric_keep = list(cfg.numeric_features) + (["readmit_30"] if "readmit_30" in df_eng.columns else [])
    means, readm = profile_by_cluster(df_eng, labels, numeric_keep=numeric_keep)
    save_profiles(means, readm, out_dir=str(A))

    summary = cluster_summary(labels, readmitted=df_eng.get("readmitted"))
    summary.to_csv(A / f"cluster_summary_k{args.k}.csv", index=False)
    print(f"[report] saved: {A}/pca_kmeans_k{args.k}.png, tsne_kmeans_k{args.k}.png, and cluster_feature_means.csv, readmission_by_cluster.csv, cluster_summary_k{args.k}.csv")

def main():
    p = argparse.ArgumentParser(description="Diabetes clustering CLI with caching")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Load, engineer, fit preprocessor, cache X/df")
    sp.add_argument("--tag", default=None, help="Optional run tag")
    sp.set_defaults(fn=cmd_prepare)

    sp = sub.add_parser("tune", help="Sweep k using cached X")
    sp.add_argument("--k-min", type=int, default=2)
    sp.add_argument("--k-max", type=int, default=8)
    sp.add_argument("--tag", default=None)
    sp.set_defaults(fn=cmd_tune)

    sp = sub.add_parser("fit", help="Fit KMeans on cached X and save model/labels")
    sp.add_argument("--k", type=int, help="Override k")
    sp.add_argument("--tag", default=None)
    sp.set_defaults(fn=cmd_fit)

    sp = sub.add_parser("report", help="Make plots/tables from cached labels")
    sp.add_argument("--k", type=int, required=True)
    sp.add_argument("--tag", default=None)
    sp.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
