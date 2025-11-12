# Diabetes Readmission — Clustering Risk Profiles

Minimal, clean repository for *unsupervised* clustering of the **UCI Diabetes 130-US Hospitals (1999–2008)** dataset.

## Goals
- Cluster patients into risk profiles using **KMeans/DBSCAN**.
- Use **MinMaxScaler** for numeric features (your preference).
- Reduce dimensionality with **PCA** for 2D visualization.
- Compare **readmission rates** across clusters for interpretation.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the end-to-end pipeline
python main.py

# (optional) quick EDA
python notebooks/eda_quicklook.py
```
Artifacts saved to `outputs/`.

## Project layout
```
.
├── data/
│   ├── raw/           # optional: place diabetic_data.csv here to load locally
│   └── processed/     # cached processed frames / feature names
├── notebooks/
│   └── eda_quicklook.py
├── outputs/           # figures, csv summaries, report
├── poster/            # poster outline (Markdown)
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocess.py
│   ├── cluster.py
│   ├── evaluate.py
│   └── plotting.py
├── main.py
└── requirements.txt
```

## Notes
- If `data/raw/diabetic_data.csv` is present, the pipeline loads it.
- Otherwise, it fetches via `ucimlrepo` (dataset id=296).
- No PHI or direct identifiers are used; `encounter_id`/`patient_nbr` are dropped.
