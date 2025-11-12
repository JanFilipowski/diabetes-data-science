# Quick EDA (console prints only, no heavy plots)
import pandas as pd
from src.data import load_data
from src.preprocess import engineer_features

if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3))

    df = engineer_features(df)
    print("\nMissing values per column (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    if 'readmitted' in df.columns:
        print("\nReadmitted value counts:")
        print(df['readmitted'].value_counts())
