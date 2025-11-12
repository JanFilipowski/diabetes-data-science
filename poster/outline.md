# Poster Outline — *Finding Patterns in 100,000 Patients: Clustering Risk Profiles in Diabetes Care*

## Goal
Group diabetic patients into **natural clusters** and examine whether clusters differ in **readmission risk**.

## Dataset
UCI Diabetes 130-US Hospitals (1999–2008), 47 features, 101,766 rows. Cite: Strack et al., 2014.

## Methods
Preprocessing (missing, encoding, MinMax scaling) → PCA (2D) → KMeans (k=3–6) → Cluster profiling (feature means) → Readmission comparison

## Results (place figures)
- PCA 2D scatter colored by cluster (Fig. 1)
- Heatmap of feature averages per cluster (Fig. 2)
- Readmission rate per cluster (Fig. 3)

## Discussion
- Which clusters look high-risk? Which features dominate?
- Limitations: administrative codes, missingness, observational nature.

## Ethics
Anonymized dataset, exploratory analysis, no individual-level decisions.

## References
Strack, B. et al. (2014). *BioMed Research International*.
UCI ML Repository: Diabetes 130-US Hospitals.
