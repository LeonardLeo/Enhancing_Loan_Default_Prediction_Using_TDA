# Experiments

Top-level buckets live under both `5_Experiments/` and `6_Results/`. Numbered experiments live *inside* a bucket.

| Bucket | What it is |
|--------|------------|
| `Default_Parameters/` | Tabular ML: old Exp 1 (defaults) and old Exp 2 (tuned). |
| `Early_Split_And_Undersample_H0/` | Early split and undersample, using just H0 |
| `Early_Split_And_Undersample_H0_And_H1/` | Early split and undersample, using both H0 and H1 |
| `Early_Split_No_Undersample_H0/` | Early split, no undersample, using just H0 |
| `Early_Split_No_Undersample_H0_And_H1/` | Early split, no undersample, using both H0 and H1 |
| `Late_Split_And_Undersample_H0/` | Late split and undersample (the original historical run), using just H0 |
| `Late_Split_And_Undersample_H0_And_H1/` | Late split and undersample (the original historical run), using both H0 and H1 |
| `Late_Split_No_Undersample_H0/` | Late split, no undersample, using just H0 |
| `Late_Split_No_Undersample_H0_And_H1/` | Late split, no undersample, using both H0 and H1 |
| `Statistics/` | Protocol-independent geometry (intrinsic dimension). |
| `Snapshot_Sample_Size/` | Dated 13/08/2026. Items 1, 2, and 4 (item 3 is this study, not a third grid). English snapshot wording throughout. |
| `Archives/` | Retired experiments kept as a museum of old work (original numbers), plus `Four_Arm_Nested_Experiments/`. |

The eight TDA folders are the live processes (split × undersample × just H0 vs both H0 and H1). Public names always use “and”, never “+”, and are defined once in `utils.TDA_PROCESS_REGISTRY`. Every live TDA process runs the same five experiments (`utils.ACTIVE_TDA_EXPERIMENT_NAMES`). H0 processes slice barcode tables from the matching H0-and-H1 run; they do not run Ripser.

TDA artefacts are mirrored at:

```
1_Data/TDA_Datasets/{ProtocolBucket}/{ExperimentName}/{Dataset}/
1_Data/Landmark_Sets/{ProtocolBucket}/{ExperimentName}/{Dataset}/
1_Data/Barcode_Statistics/{ProtocolBucket}/{ExperimentName}/{Dataset}/
```

Landmarks and Ripser output live on the four `*_H0_And_H1` folders. `1_Data/Processed_Datasets/` is shared and is not re-bucketed.

Folder map for a non-technical reader: `docs/Repository_Layout.md`. Bucket READMEs live in each top-level folder. Experiment write-ups are `REPORT.md`.

## Visualization

Every **active** experiment folder has `visualize_results.py` at the experiment root (not inside a dataset subfolder). Run that script; figures land in `6_Results/{Bucket}/{Experiment}/Visualizations/` only — see the catalog in `6_Results/README.md`. Figure titles use `utils.process_display_name()`.

```
.\tda_env\Scripts\python.exe 5_Experiments/Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/visualize_results.py
```

Default and tuned PH experiments consume `model_results.pkl` / metric CSVs and write per-dataset test dashboards plus cross-dataset metric facets. Sampling-audit, Algorithm 2, and revised-protocol folders plot their CSVs/JSON. `Statistics/1_Intrinsic_Dimension_Estimation` plots Two-NN before/after PCA and the remaining ID estimators. If artefacts are missing, the script exits with `results not generated yet` and the expected path.

Archived experiments keep their original `visualize_results.py` files under `5_Experiments/Archives/`. Historical scripts may still use symbols `t`/`l` in code identifiers; see `docs/Notation.md`.

`Snapshot_Sample_Size` has its own `visualize_results.py` in each of `1_Snapshot_Count_Sweep`, `2_Points_Per_Snapshot_Sweep`, and `3_Snapshot_Count_Across_Cloud_Sizes`. Those call `utils.py`, not `visualize_experiment_folder`. The sample-size tree still uses the four historical protocol keys under `0_Shared_Pools/`.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage.

For the snapshot sample-size study, open `Snapshot_Sample_Size/0_Shared_Pools`. The numbered experiment scripts only select which rows go on which figure.
