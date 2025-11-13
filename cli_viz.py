import argparse
from src.io_utils import run_id_from_cfg, artdir, load_parquet, load_npz
from src.config import Config
from src.plotting_extra import feature_heatmap, readmission_bar, demographics_bar

def main():
    p = argparse.ArgumentParser(description="Generate extended visualizations (A,B,C)")
    p.add_argument("--tag", default=None, help="Run tag (used for folder lookup)")
    p.add_argument("--k", type=int, required=True, help="Number of clusters used")
    args = p.parse_args()

    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    print(f"[viz] loading cached data for {run_id}…")
    df_eng = load_parquet(A / "df_eng.parquet")
    labels = load_npz(A / f"labels_k{args.k}.npz")["labels"]

    print("[viz] generating heatmap of numeric features…")
    feature_heatmap(df_eng, labels, cfg.numeric_features, A / f"feature_heatmap_k{args.k}.png")

    print("[viz] generating readmission rate bar chart…")
    readmission_bar(df_eng, labels, A / f"readmission_bar_k{args.k}.png")

    print("[viz] generating demographic distributions…")
    demographics_bar(df_eng, labels, A / f"demographics_k{args.k}.png")

    print(f"[viz] done. Files saved to: {A}")

if __name__ == "__main__":
    main()
