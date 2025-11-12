import os
import pandas as pd

def load_local_csv(path='data/raw/diabetic_data.csv'):
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        return df
    return None

def load_ucimlrepo():
    from ucimlrepo import fetch_ucirepo
    diabetes = fetch_ucirepo(id=296)
    X = diabetes.data.features
    y = diabetes.data.targets
    # Merge into one DF for convenience (targets has 'readmitted')
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X)
    if isinstance(y, pd.DataFrame):
        for col in y.columns:
            df[col] = y[col]
    else:
        df['readmitted'] = y
    return df

def load_data():
    df = load_local_csv()
    if df is None:
        df = load_ucimlrepo()
    return df
