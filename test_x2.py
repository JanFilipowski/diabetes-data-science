import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency

from src.config import Config
from src.io_utils import run_id_from_cfg, artdir, load_parquet, load_npz

def safe_json_dump(obj, path):
    """Zapisz do JSON konwertując obiekty numpy i bool na typy Pythona."""
    def convert(o):
        if hasattr(o, "item"):
            return o.item()  # np.float32, np.int64, np.bool_
        if isinstance(o, (set,)):
            return list(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


def analyze_gender_bias(df_eng: pd.DataFrame, labels, out_dir: Path):
    """Analiza zależności płeć ↔ klaster (Chi-square + wykres)."""

    if 'gender' not in df_eng.columns:
        print("[bias] brak kolumny 'gender' w danych — pomijam analizę.")
        return None

    # Tablica krzyżowa (counts)
    ct = pd.crosstab(df_eng['gender'], labels)
    chi2, p, dof, exp = chi2_contingency(ct)

    # Przelicz na procenty dla czytelności
    ct_pct = pd.crosstab(df_eng['gender'], labels, normalize='columns') * 100

    # Wykres rozkładu procentowego płci w klastrach
    plt.figure(figsize=(6, 4))
    ct_pct.T.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#1f77b4', '#ff7f0e'])
    plt.ylabel("Percentage within cluster")
    plt.xlabel("Cluster")
    plt.title("Gender distribution by cluster")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(out_dir / "gender_distribution.png", dpi=200)
    plt.close()

    # Zapis wyników testu
    result = {
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "significant": p < 0.05,
        "table_counts": ct.to_dict(),
        "table_percent": ct_pct.round(2).to_dict(),
    }

    safe_json_dump(result, out_dir / "chi2_gender.json")

    print(f"[bias] Chi² test done — chi2={chi2:.3f}, p={p:.4f} {'*' if p<0.05 else '(ns)'}")
    print(f"[bias] saved: {out_dir}/chi2_gender.json, gender_distribution.png")

    return result


def main():
    p = argparse.ArgumentParser(description="Analyze demographic bias (e.g. gender × cluster)")
    p.add_argument("--tag", default=None, help="Run tag used in artifacts folder")
    p.add_argument("--k", type=int, required=True, help="Number of clusters used")
    args = p.parse_args()

    cfg = Config()
    run_id = run_id_from_cfg(cfg, tag=args.tag)
    A = artdir(run_id)

    print(f"[bias] Loading cached data for {run_id} (k={args.k})...")
    df_eng = load_parquet(A / "df_eng.parquet")
    labels = load_npz(A / f"labels_k{args.k}.npz")["labels"]

    analyze_gender_bias(df_eng, labels, A)

    print(f"[bias] Analysis completed. Results stored in: {A}")


if __name__ == "__main__":
    main()
