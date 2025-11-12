from dataclasses import dataclass

@dataclass
class Config:
    # General
    random_state: int = 42
    cache_processed: bool = True

    # Clustering
    n_clusters: int = 4  # k for KMeans
    dbscan_eps: float = 0.7
    dbscan_min_samples: int = 50
    num_block_weight = 1.5

    use_feature_selection: bool = True
    select_k_best: int = 25  # liczba cech po selekcji

    # PCA
    pca_components: int = 2

    # Feature lists
    numeric_features = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses',
        'total_visits'
    ]

    categorical_features = [
        'age',
        'gender',
        'race',
        'A1Cresult',
        'max_glu_serum',
        'insulin',
        'metformin',
        'change',
        'diabetesMed',
        'diag_1_group'
    ]

    # Output paths
    out_dir: str = 'outputs'
    processed_dir: str = 'data/processed'
