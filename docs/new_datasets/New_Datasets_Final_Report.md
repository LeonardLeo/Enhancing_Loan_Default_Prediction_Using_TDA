# Four New Datasets: Consolidated Experimental Report

Generated: 2026-07-30T12:08:44.892731+00:00

## Provenance

- Recorded mirror checksums: 17/17 matched.
- Primary-source verification: **not completed**. working copies are third-party mirrors; primary hosts were unavailable.
- Licence strings are inherited from the acquisition manifest and require primary-record confirmation.

## Protocols

- **Historical / Protocol A:** preprocessing and PCA may see full data; retained only for comparability and explicitly leaky.
- **Clean / Protocol B:** 80/20 stratified split first; imputation, missing indicators, winsorization, encoding, constant removal, scaling, PCA and ADASYN fit on training only.
- PKDD transaction aggregates use the strict rule `transaction_date < loan_date`; undated standing orders are excluded.
- Experiment 27 uses the barcode-vector F_pq proxy, not true persistence-diagram distances.

## Dataset audits

### pkdd_czech

- Shape: 682 rows × 39 columns; default rate 11.144%.
- Missing cells: 1938; duplicates: 0; constants: 1.
- Near-constant features (≥99.5% one value): 1; 1e9-scale features: 0.

### polish_bankruptcy

- Shape: 10503 rows × 65 columns; default rate 4.713%.
- Missing cells: 9888; duplicates: 87; constants: 0.
- Near-constant features (≥99.5% one value): 0; 1e9-scale features: 0.

### south_german_credit

- Shape: 1000 rows × 21 columns; default rate 30.000%.
- Missing cells: 0; duplicates: 0; constants: 0.
- Near-constant features (≥99.5% one value): 0; 1e9-scale features: 0.

### taiwan_bankruptcy

- Shape: 6819 rows × 96 columns; default rate 3.226%.
- Missing cells: 0; duplicates: 0; constants: 1.
- Near-constant features (≥99.5% one value): 2; 1e9-scale features: 23.

## Best completed baseline results

| dataset             | protocol   | model         |   balanced_accuracy |       f1 |   roc_auc |   average_precision |
|:--------------------|:-----------|:--------------|--------------------:|---------:|----------:|--------------------:|
| polish_bankruptcy   | clean      | random_forest |            0.714286 | 0.459184 |  0.882141 |            0.420617 |
| polish_bankruptcy   | historical | random_forest |            0.715784 | 0.473684 |  0.890819 |            0.426711 |
| taiwan_bankruptcy   | clean      | xgb           |            0.821212 | 0.47619  |  0.952996 |            0.461308 |
| pkdd_czech          | clean      | xgb           |            0.683607 | 0.48     |  0.808197 |            0.565598 |
| pkdd_czech          | historical | xgb           |            0.683607 | 0.48     |  0.824044 |            0.567113 |
| taiwan_bankruptcy   | historical | random_forest |            0.771212 | 0.485437 |  0.943363 |            0.504721 |
| south_german_credit | clean      | svm           |            0.690476 | 0.566667 |  0.744405 |            0.595586 |
| south_german_credit | historical | svm           |            0.697619 | 0.576271 |  0.751667 |            0.614627 |

## Best completed TDA results

| dataset             | protocol   | snapshot_variant   |   landmark_percent | model         |   balanced_accuracy |       f1 |   roc_auc |
|:--------------------|:-----------|:-------------------|-------------------:|:--------------|--------------------:|---------:|----------:|
| south_german_credit | clean      | revised            |                 20 | xgb           |               0.5   | 0.666667 |  0.7      |
| south_german_credit | clean      | historical500      |                 20 | xgb           |               0.504 | 0.668005 |  0.686302 |
| pkdd_czech          | clean      | historical500      |                 20 | random_forest |               0.562 | 0.69541  |  0.644464 |
| pkdd_czech          | clean      | revised            |                 20 | xgb           |               0.9   | 0.909091 |  1        |
| taiwan_bankruptcy   | clean      | revised            |                 10 | logistic      |               1     | 1        |  1        |
| taiwan_bankruptcy   | historical | revised            |                 10 | logistic      |               1     | 1        |  1        |
| pkdd_czech          | historical | revised            |                 10 | logistic      |               1     | 1        |  1        |
| polish_bankruptcy   | historical | revised            |                 20 | knn           |               1     | 1        |  1        |
| polish_bankruptcy   | clean      | revised            |                 10 | random_forest |               1     | 1        |  1        |
| south_german_credit | historical | historical500      |                 20 | logistic      |               1     | 1        |  1        |
| pkdd_czech          | historical | historical500      |                 20 | xgb           |               1     | 1        |  1        |
| south_german_credit | historical | revised            |                 10 | xgb           |               1     | 1        |  1        |
| polish_bankruptcy   | clean      | historical500      |                 10 | xgb           |               1     | 1        |  1        |
| polish_bankruptcy   | historical | historical500      |                 20 | xgb           |               1     | 1        |  1        |
| taiwan_bankruptcy   | historical | historical500      |                 20 | random_forest |               1     | 1        |  1        |
| taiwan_bankruptcy   | clean      | historical500      |                 20 | random_forest |               1     | 1        |  1        |

## Best completed extended-experiment results

|   experiment | dataset             | protocol   | variant                   |   landmark_percent | model             | setting                               |   balanced_accuracy |       f1 |    roc_auc |
|-------------:|:--------------------|:-----------|:--------------------------|-------------------:|:------------------|:--------------------------------------|--------------------:|---------:|-----------:|
|           16 | pkdd_czech          | clean      | pca_component_sweep       |                nan | logistic          | components=10                         |            0.562295 | 0.219512 |   0.522951 |
|           16 | polish_bankruptcy   | clean      | pca_component_sweep       |                nan | logistic          | components=20                         |            0.729243 | 0.222222 |   0.773671 |
|           13 | pkdd_czech          | clean      | matched_pca_variance      |                nan | logistic          | reference_variance=0.90               |            0.581148 | 0.24     |   0.551366 |
|           13 | polish_bankruptcy   | clean      | matched_pca_variance      |                nan | logistic          | reference_variance=0.90               |            0.736597 | 0.282353 |   0.797844 |
|           13 | taiwan_bankruptcy   | clean      | matched_pca_variance      |                nan | logistic          | reference_variance=0.90               |            0.854167 | 0.289062 |   0.932042 |
|           18 | taiwan_bankruptcy   | clean      | pca_component_sweep       |                nan | logistic          | components=5                          |            0.878788 | 0.3083   |   0.928289 |
|           12 | south_german_credit | clean      | matched_sample_size       |                nan | logistic          | t=12                                  |            0.6      | 0.333333 |   0.64     |
|           14 | south_german_credit | clean      | 1_default_to_4_nondefault |                 10 | logistic          | 20 default / 80 non-default snapshots |            0.2625   | 0.337079 |   0.36     |
|            2 | taiwan_bankruptcy   | historical | tuned_baseline            |                nan | xgb               | bounded_grid_3fold                    |            0.781439 | 0.485981 |   0.952256 |
|            2 | taiwan_bankruptcy   | clean      | tuned_baseline            |                nan | xgb               | bounded_grid_3fold                    |            0.782955 | 0.504854 |   0.948933 |
|            2 | polish_bankruptcy   | historical | tuned_baseline            |                nan | xgb               | bounded_grid_3fold                    |            0.751388 | 0.530612 |   0.918844 |
|            2 | pkdd_czech          | clean      | tuned_baseline            |                nan | xgb               | bounded_grid_3fold                    |            0.737978 | 0.533333 |   0.832787 |
|            2 | polish_bankruptcy   | clean      | tuned_baseline            |                nan | xgb               | bounded_grid_3fold                    |            0.766789 | 0.555556 |   0.91622  |
|            2 | pkdd_czech          | historical | tuned_baseline            |                nan | xgb               | bounded_grid_3fold                    |            0.721038 | 0.56     |   0.883607 |
|            2 | south_german_credit | historical | tuned_baseline            |                nan | svm               | bounded_grid_3fold                    |            0.685714 | 0.565217 |   0.764405 |
|            2 | south_german_credit | clean      | tuned_baseline            |                nan | svm               | bounded_grid_3fold                    |            0.689286 | 0.569343 |   0.765893 |
|           13 | south_german_credit | clean      | matched_pca_variance      |                nan | logistic          | reference_variance=0.90               |            0.692857 | 0.575342 |   0.74869  |
|           18 | south_german_credit | clean      | pca_component_sweep       |                nan | logistic          | components=20                         |            0.696429 | 0.57931  |   0.740476 |
|           12 | taiwan_bankruptcy   | clean      | matched_sample_size       |                nan | logistic          | t=6                                   |            0.6875   | 0.615385 |   0.84375  |
|            4 | south_german_credit | clean      | historical500             |                 20 | logistic          | default                               |            0.508    | 0.669355 |   0.871544 |
|            6 | south_german_credit | clean      | historical500             |                 20 | logistic          | default                               |            0.533    | 0.680356 |   0.889836 |
|           14 | pkdd_czech          | clean      | 1_default_to_4_nondefault |                 10 | logistic          | 20 default / 80 non-default snapshots |            0.5375   | 0.683761 |   0.955    |
|           19 | south_german_credit | clean      | revised                   |                 10 | linear_regression | default                               |            0.55     | 0.689655 | nan        |
|           19 | pkdd_czech          | clean      | revised                   |                 20 | linear_regression | default                               |            0.6      | 0.714286 | nan        |
|           11 | pkdd_czech          | clean      | revised                   |                 20 | logistic          | default                               |            0.6      | 0.714286 |   1        |
|           11 | south_german_credit | clean      | revised                   |                 20 | logistic          | default                               |            0.6      | 0.714286 |   1        |
|            6 | pkdd_czech          | clean      | revised                   |                 10 | logistic          | default                               |            0.8      | 0.833333 |   1        |
|           12 | pkdd_czech          | clean      | matched_sample_size       |                nan | logistic          | t=6                                   |            0.833333 | 0.857143 |   1        |
|           12 | polish_bankruptcy   | clean      | matched_sample_size       |                nan | logistic          | t=12                                  |            0.888889 | 0.888889 |   0.938272 |
|           14 | polish_bankruptcy   | clean      | 1_default_to_4_nondefault |                 10 | logistic          | 20 default / 80 non-default snapshots |            0.8875   | 0.898876 |   1        |
|            4 | pkdd_czech          | clean      | revised                   |                 20 | xgb               | default                               |            0.9      | 0.909091 |   0.9      |
|           11 | polish_bankruptcy   | clean      | historical500             |                 10 | logistic          | default                               |            0.945    | 0.947867 |   1        |
|            6 | polish_bankruptcy   | clean      | historical500             |                 20 | logistic          | default                               |            0.95     | 0.952381 |   1        |
|           14 | taiwan_bankruptcy   | clean      | 1_default_to_4_nondefault |                 10 | logistic          | 20 default / 80 non-default snapshots |            0.975    | 0.97561  |   1        |
|           19 | south_german_credit | historical | revised                   |                 20 | linear_regression | default                               |            1        | 1        | nan        |
|           11 | south_german_credit | historical | revised                   |                 10 | logistic          | default                               |            1        | 1        |   1        |
|            6 | south_german_credit | historical | revised                   |                 10 | logistic          | default                               |            1        | 1        |   1        |
|            4 | south_german_credit | historical | revised                   |                 10 | logistic          | default                               |            1        | 1        |   1        |
|           11 | taiwan_bankruptcy   | clean      | revised                   |                 10 | logistic          | default                               |            1        | 1        |   1        |
|            4 | taiwan_bankruptcy   | clean      | revised                   |                 10 | logistic          | default                               |            1        | 1        |   1        |
|           19 | polish_bankruptcy   | clean      | revised                   |                 10 | linear_regression | default                               |            1        | 1        | nan        |
|           11 | polish_bankruptcy   | historical | historical500             |                 20 | logistic          | default                               |            1        | 1        |   1        |
|           19 | polish_bankruptcy   | historical | historical500             |                 20 | linear_regression | default                               |            1        | 1        | nan        |
|            6 | polish_bankruptcy   | historical | historical500             |                 20 | logistic          | default                               |            1        | 1        |   1        |
|            6 | taiwan_bankruptcy   | historical | historical500             |                 10 | logistic          | default                               |            1        | 1        |   1        |
|           19 | taiwan_bankruptcy   | historical | historical500             |                 10 | linear_regression | default                               |            1        | 1        | nan        |
|           11 | taiwan_bankruptcy   | historical | historical500             |                 10 | logistic          | default                               |            1        | 1        |   1        |
|            4 | polish_bankruptcy   | historical | historical500             |                 20 | random_forest     | default                               |            1        | 1        |   1        |
|            4 | polish_bankruptcy   | clean      | historical500             |                 20 | xgb               | default                               |            1        | 1        |   1        |
|            4 | taiwan_bankruptcy   | historical | historical500             |                 10 | logistic          | default                               |            1        | 1        |   1        |
|            6 | pkdd_czech          | historical | historical500             |                 20 | logistic          | default                               |            1        | 1        |   1        |
|           19 | pkdd_czech          | historical | revised                   |                 20 | linear_regression | default                               |            1        | 1        | nan        |
|           11 | pkdd_czech          | historical | revised                   |                 20 | logistic          | default                               |            1        | 1        |   1        |
|            6 | taiwan_bankruptcy   | clean      | historical500             |                 20 | logistic          | default                               |            1        | 1        |   1        |
|           19 | taiwan_bankruptcy   | clean      | historical500             |                 20 | linear_regression | default                               |            1        | 1        | nan        |
|            4 | pkdd_czech          | historical | revised                   |                 10 | logistic          | default                               |            1        | 1        |   1        |

## Limitations

- Mirror provenance and licence text remain unresolved against primary records.
- Independent snapshot classification can be degenerate when train/test snapshot distributions are nearly indistinguishable.
- Revised snapshot counts implement the Experiment 24 calibration `l = ceil(n_class / t)` separately by split and landmark size.

- Historical500 status at report generation: 16/16 configurations complete; see `6_Results/New_Datasets/active_jobs.json`.

## Reproduction

```powershell
.\tda_env\Scripts\python.exe run_new_datasets.py --stages ingest baseline tda report
.\tda_env\Scripts\python.exe run_remaining_experiments.py
.\tda_env\Scripts\python.exe run_remaining_experiments.py --experiments 4 6 7 8 10 11 15 19 25 27
.\tda_env\Scripts\python.exe run_new_datasets.py --stages report
```
