# Experiments

Top-level buckets live under both `5_Experiments/` and `6_Results/`. Numbered experiments live *inside* a bucket.

| Bucket | What it is |
|--------|------------|
| `Default_Parameters/` | Tabular ML: old Exp 1 (defaults) and old Exp 2 (tuned). |
| `Historical_Late_Split_Balanced_TDA/` | Original TDA pipeline: full-table PCA, undersample, then 80/20 on barcode rows. |
| `Early_Split_TDA/` | Split customers first; still undersample inside each split. |
| `No_Undersampling/` | Late split + no majority downsample. Points per snapshot = floor(class count × snapshot size percent / 100) per class. |
| `Early_Split_TDA_And_No_Undersampling/` | Early split + no undersample. |
| `Statistics/` | Protocol-independent geometry (intrinsic dimension). |
| `Snapshot_Sample_Size/` | Dated 13/08/2026. Items 1, 2, and 4 (item 3 is this study, not a third grid). English snapshot wording throughout. |
| `Archives/` | Retired experiments kept as a museum of old work (original numbers). |

There is no sixth TDA arm. **No Undersampling** *is* late-split + no undersampling.

Active TDA arms share the same nine experiments, renumbered 1–9 inside the arm (PH default through Revised Snapshot Protocol). TDA artefacts are mirrored at:

```
1_Data/TDA_Datasets/{ProtocolBucket}/{ExperimentName}/{Dataset}/
1_Data/Landmark_Sets/{ProtocolBucket}/{ExperimentName}/{Dataset}/
1_Data/Barcode_Statistics/{ProtocolBucket}/{ExperimentName}/{Dataset}/
```

`1_Data/Processed_Datasets/` is shared and is not re-bucketed.

Folder map for a non-technical reader: `docs/Repository_Layout.md`. Bucket READMEs live in each top-level folder. Experiment write-ups are `REPORT.md`.

## Visualization

Every **active** experiment folder has `visualize_results.py` at the experiment root (not inside a dataset subfolder). Run that script; figures land in `6_Results/{Bucket}/{Experiment}/Visualizations/` only — see the catalog in `6_Results/README.md`.

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/visualize_results.py
```

Exp 1–5 (and Default_Parameters 1–2) consume `model_results.pkl` / metric CSVs and write per-dataset test dashboards plus cross-dataset metric facets. Exp 6–9 plot sampling-audit, snapshot mean/variance, Algorithm 2, and revised-protocol CSVs/JSON. `Statistics/1_Intrinsic_Dimension_Estimation` plots Two-NN before/after PCA and the remaining ID estimators. If artefacts are missing, the script exits with `results not generated yet` and the expected path.

Archived experiments keep their original `visualize_results.py` files under `5_Experiments/Archives/`. Historical scripts may still use symbols `t`/`l` in code identifiers; see `docs/Notation.md`.

`Snapshot_Sample_Size` has its own `visualize_results.py` in each of `1_Snapshot_Count_Sweep`, `2_Points_Per_Snapshot_Sweep`, and `3_Snapshot_Count_Across_Cloud_Sizes`. Those call `sample_size_lib.py`, not `visualize_experiment_folder`.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. `run.py` is an optional convenience launcher and is not the method document.
