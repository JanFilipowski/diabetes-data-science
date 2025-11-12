import re
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from .config import Config

ICD9_RANGES = [
    ('Diabetes', [(250, 250)]),
    ('Circulatory', [(390, 459), (785, 785)]),
    ('Respiratory', [(460, 519)]),
    ('Digestive', [(520, 579)]),
    ('Injury', [(800, 999)]),
    ('Musculoskeletal', [(710, 739)]),
    ('Genitourinary', [(580, 629)]),
    ('Neoplasms', [(140, 239)]),
]

def parse_icd9_primary(code: str) -> str:
    if pd.isna(code):
        return 'Unknown'
    s = str(code).strip()
    if s.startswith(('V','E')):
        return 'Other'
    m = re.match(r'^(\d{3})', s)
    if not m:
        return 'Other'
    val = int(m.group(1))
    for label, ranges in ICD9_RANGES:
        for lo, hi in ranges:
            if lo <= val <= hi:
                return label
    return 'Other'

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifiers
    for col in ['encounter_id', 'patient_nbr']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Fill some categorical NaNs
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Unknown')

    # ICD9 grouping
    if 'diag_1' in df.columns:
        df['diag_1_group'] = df['diag_1'].apply(parse_icd9_primary)
    else:
        df['diag_1_group'] = 'Unknown'

    # Derived features
    for c in ['number_outpatient','number_inpatient','number_emergency']:
        if c not in df.columns:
            df[c] = 0
    df['total_visits'] = df['number_outpatient'].fillna(0) + \
                         df['number_inpatient'].fillna(0) + \
                         df['number_emergency'].fillna(0)

    # Binary target (for feature selection only)
    if 'readmitted' in df.columns:
        df['readmit_30'] = (df['readmitted'].astype(str) == '<30').astype(int)

    return df

def build_preprocessor(df: pd.DataFrame, cfg: Config):
    """Return fitted ColumnTransformer and feature lists."""
    numeric_features = [f for f in cfg.numeric_features if f in df.columns]
    categorical_features = [f for f in cfg.categorical_features if f in df.columns]

    transformers = []
    if numeric_features:
        transformers.append(('num', MinMaxScaler(), numeric_features))
    if categorical_features:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append(('cat', ohe, categorical_features))

    ct = ColumnTransformer(
        transformers,
        transformer_weights={'num': cfg.num_block_weight, 'cat': 1.0}
    )
    return ct, numeric_features, categorical_features

def apply_feature_selection(X, df_eng, cfg: Config, feature_names=None):
    """Reduce dimensionality using mutual information with readmit_30."""
    if not cfg.use_feature_selection or 'readmit_30' not in df_eng.columns:
        return X, feature_names

    y = df_eng['readmit_30'].values
    selector = SelectKBest(mutual_info_classif, k=min(cfg.select_k_best, X.shape[1]))
    X_sel = selector.fit_transform(X, y)

    if feature_names is not None:
        selected_mask = selector.get_support()
        selected_names = [n for n, keep in zip(feature_names, selected_mask) if keep]
    else:
        selected_names = None

    return X_sel, selected_names
