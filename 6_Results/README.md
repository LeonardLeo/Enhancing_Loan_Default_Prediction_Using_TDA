# Results

`6_Results/` mirrors the protocol buckets in `5_Experiments/`. Numbered experiments live *inside* a bucket. **All generated figures live in one folder per experiment:**

`6_Results/{Bucket}/{Experiment}/Visualizations/`

Do not look in `model_viz/`, `cv_viz/`, `cross_dataset_viz/`, or `plots/` on the active tree — those names are retired. Folder map: `docs/Repository_Layout.md`.

## Top-level layout

| Item | Role |
|------|------|
| `Default_Parameters/` | Tabular ML baselines (old Exp 1–2). |
| `Historical_Late_Split_Balanced_TDA/` | Original TDA protocol (late split + undersample). |
| `Early_Split_TDA/` | Split customers first; still undersample. |
| `No_Undersampling/` | Late split, no majority downsample. |
| `Early_Split_TDA_And_No_Undersampling/` | Early split + no undersample. |
| `Statistics/` | Protocol-independent geometry (intrinsic dimension). |
| `Snapshot_Sample_Size/` | Sample-size study. |
| `Archives/` | Retired exploratory experiments (original numbers). |
| `Paper_Tables/` | Aggregated LaTeX/CSV tables for the paper. |
| `Run_Queue/` | Operational Ripser/consumer queues, logs, and run registries. |
| `results.py` | Documented aggregator. Run from this directory; it **writes** into `Paper_Tables/`. |
| `_ripser_queue.py`, `_consumer_queue.py` | Compatibility shims that execute the scripts in `Run_Queue/`. |

## Paper tables

```powershell
cd 6_Results
python results.py
```

Outputs:

- `Paper_Tables/clean_experiment_results.csv`
- `Paper_Tables/results_table.tex`
- `Paper_Tables/statlog_german_credit_results_table.tex`
- `Paper_Tables/default_of_credit_card_client_results_table.tex`
- `Paper_Tables/results_experiment_1.tex` … `results_experiment_10.tex`

## Queues

Prefer the new location:

```powershell
.\tda_env\Scripts\python.exe 6_Results\Run_Queue\_ripser_queue.py
.\tda_env\Scripts\python.exe 6_Results\Run_Queue\_consumer_queue.py
```

The root shims still work if an in-flight command used the old path. Logs and registry JSON/CSV live in `Run_Queue/`.

## Visualization catalog

Every active experiment has `5_Experiments/{Bucket}/{Experiment}/visualize_results.py`. Run that script; figures land **only** in the matching `6_Results/{Bucket}/{Experiment}/Visualizations/` folder. If artefacts are missing, the script exits with `results not generated yet`.

| Bucket | Experiment | Visualizations folder | Figures |
|--------|------------|----------------------|---------|
| `Default_Parameters` | `1_ML_Default_Parameters` | `6_Results/Default_Parameters/1_ML_Default_Parameters/Visualizations/` | Per-dataset test dashboards (`{Dataset}_test_metrics_by_model.png`); cross-dataset facets (`f1_by_model_faceted.png`, `accuracy_by_model_faceted.png`, `precision_by_model_faceted.png`, `recall_by_model_faceted.png`); CV fold and mean accuracy (`cv_{Dataset}_*.png`) |
| `Default_Parameters` | `2_ML_Tuned_Parameters` | `6_Results/Default_Parameters/2_ML_Tuned_Parameters/Visualizations/` | Same figure set as Exp 1 (tuned models) |
| `Historical_Late_Split_Balanced_TDA` | `1_PH_Default_Parameters` | `6_Results/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Visualizations/` | Test dashboards + cross-dataset metric facets (hue = landmark %); CV if present |
| `Historical_Late_Split_Balanced_TDA` | `2_PH_Tuned_Parameters` | `6_Results/Historical_Late_Split_Balanced_TDA/2_PH_Tuned_Parameters/Visualizations/` | Same as arm Exp 1 (tuned) |
| `Historical_Late_Split_Balanced_TDA` | `3_H0_Only` | `6_Results/Historical_Late_Split_Balanced_TDA/3_H0_Only/Visualizations/` | Same as arm Exp 1 (H₀ features only) |
| `Historical_Late_Split_Balanced_TDA` | `4_Dropping_Correlated_Barcode_Statistics_Columns` | `6_Results/Historical_Late_Split_Balanced_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/Visualizations/` | Same as arm Exp 1 (decorrelated barcode stats) |
| `Historical_Late_Split_Balanced_TDA` | `5_Linear_Regression_For_Prediction` | `6_Results/Historical_Late_Split_Balanced_TDA/5_Linear_Regression_For_Prediction/Visualizations/` | Same as arm Exp 1 (linear models) |
| `Historical_Late_Split_Balanced_TDA` | `6_Sampling_Ratio_Audit` | `6_Results/Historical_Late_Split_Balanced_TDA/6_Sampling_Ratio_Audit/Visualizations/` | `sampling_reuse_by_rule_faceted.png`, `sampling_reuse_revised_rule_faceted.png` |
| `Historical_Late_Split_Balanced_TDA` | `7_Snapshot_Mean_Variance` | `6_Results/Historical_Late_Split_Balanced_TDA/7_Snapshot_Mean_Variance/Visualizations/` | `barcode_mean_g2_0_g3_1_faceted.png`, `barcode_variance_g2_0_g3_1_faceted.png`, `barcode_mean_heatmap.png` |
| `Historical_Late_Split_Balanced_TDA` | `8_Null_Hypothesis_Algorithm2` | `6_Results/Historical_Late_Split_Balanced_TDA/8_Null_Hypothesis_Algorithm2/Visualizations/` | `algorithm2_pvalues_faceted.png`, `algorithm2_observed_F_faceted.png` |
| `Historical_Late_Split_Balanced_TDA` | `9_Revised_Snapshot_Protocol` | `6_Results/Historical_Late_Split_Balanced_TDA/9_Revised_Snapshot_Protocol/Visualizations/` | `balanced_accuracy_by_model_faceted.png`, `f1_by_model_faceted.png`, concern A/B and overlap facets when those CSVs exist |
| `Early_Split_TDA` | `1_PH_Default_Parameters` … `9_Revised_Snapshot_Protocol` | `6_Results/Early_Split_TDA/{Experiment}/Visualizations/` | Same figure types as the Historical arm, experiment by experiment |
| `No_Undersampling` | `1_PH_Default_Parameters` … `9_Revised_Snapshot_Protocol` | `6_Results/No_Undersampling/{Experiment}/Visualizations/` | Same figure types as the Historical arm, experiment by experiment |
| `Early_Split_TDA_And_No_Undersampling` | `1_PH_Default_Parameters` … `9_Revised_Snapshot_Protocol` | `6_Results/Early_Split_TDA_And_No_Undersampling/{Experiment}/Visualizations/` | Same figure types as the Historical arm, experiment by experiment |
| `Statistics` | `1_Intrinsic_Dimension_Estimation` | `6_Results/Statistics/1_Intrinsic_Dimension_Estimation/Visualizations/` | `two_nn_before_after_pca.png`, `id_estimator_suite_faceted.png`, `pca_rank_vs_90pct.png`, `pca_variance_retained.png` |
| `Snapshot_Sample_Size` | `1_Snapshot_Count_Sweep` | `6_Results/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/Visualizations/` | Number of snapshots on the x-axis; each cloud has the default point count. Combined overlay = mean trend (no error bars); companion `*_ci_panels.png` hold 95% ribbons |
| `Snapshot_Sample_Size` | `2_Points_Per_Snapshot_Sweep` | `6_Results/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/Visualizations/` | Points per snapshot on the x-axis; always 60 snapshots. Combined overlay = mean trend (no error bars); companion panels hold 95% ribbons |
| `Snapshot_Sample_Size` | `3_Snapshot_Count_Across_Cloud_Sizes` | `6_Results/Snapshot_Sample_Size/3_Snapshot_Count_Across_Cloud_Sizes/Visualizations/` | Number of snapshots on the x-axis; one curve per cloud size. SVM/Logistic focus + 5×2 small multiples as mean trends; companion ribbons faceted by model × cloud size |

Example:

```powershell
.\tda_env\Scripts\python.exe 5_Experiments\Default_Parameters\1_ML_Default_Parameters\visualize_results.py
```

opens the artefacts under `6_Results/Default_Parameters/1_ML_Default_Parameters/` and writes figures to `6_Results/Default_Parameters/1_ML_Default_Parameters/Visualizations/`.
