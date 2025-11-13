import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def feature_heatmap(df_eng, labels, numeric_features, out_path):
    """A: heatmap średnich wartości cech numerycznych dla każdego klastra."""
    cluster_means = df_eng.groupby(labels)[numeric_features].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means.T, cmap="coolwarm", cbar=True)
    plt.title("Feature means per cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def readmission_bar(df_eng, labels, out_path):
    """B: słupki średnich wskaźników readmisji."""
    if 'readmit_30' not in df_eng.columns:
        print("[warn] no readmit_30 column found")
        return
    rates = df_eng.groupby(labels)['readmit_30'].mean() * 100
    plt.figure(figsize=(6,4))
    sns.barplot(x=rates.index, y=rates.values, palette="crest")
    plt.ylabel("Readmission <30 days (%)")
    plt.xlabel("Cluster")
    plt.title("Average readmission rate by cluster")
    for i, v in enumerate(rates.values):
        plt.text(i, v + 0.2, f"{v:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def demographics_bar(df_eng, labels, out_path):
    """C: prosty wykres demografii dla klastrów."""
    # tylko jeśli kolumny istnieją
    available = [c for c in ['age', 'gender', 'race'] if c in df_eng.columns]
    if not available:
        print("[warn] no demographic columns found")
        return

    plt.figure(figsize=(12, 4 * len(available)))
    for i, col in enumerate(available):
        plt.subplot(len(available), 1, i + 1)
        tab = pd.crosstab(df_eng[col], labels, normalize='columns') * 100
        tab.plot(kind='bar', stacked=True, ax=plt.gca(), legend=False)
        plt.title(f"Distribution of {col} by cluster")
        plt.ylabel("% within cluster")
        plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
