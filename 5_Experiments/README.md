# Experiments

Top-level buckets live under both `5_Experiments/` and `6_Results/`. Numbered experiments live *inside* a bucket.

| Bucket | What it is |
|--------|------------|
| `Default_Parameters/` | Tabular ML: old Exp 1 (defaults) and old Exp 2 (tuned). |
| `Historical_Late_Split_Balanced_TDA/` | Original TDA pipeline: full-table PCA, undersample, then 80/20 on barcode rows. |
| `Early_Split_TDA/` | Split customers first; still undersample inside each split. |
| `No_Undersampling/` | Late split + no majority downsample. `t = floor(n_class * L / 100)` per class. |
| `Early_Split_TDA_And_No_Undersampling/` | Early split + no undersample. |
| `Statistics/` | Protocol-independent geometry (intrinsic dimension). |
| `Archives/` | Retired experiments kept as a museum of old work (original numbers). |

There is no sixth TDA arm. **No Undersampling** *is* late-split + no undersampling.

Active TDA arms share the same nine experiments, renumbered 1–9 inside the arm (PH default through Revised Snapshot Protocol). TDA artefacts are mirrored at:

```
1_Data/TDA_Datasets/{ProtocolBucket}/{ExperimentName}/{Dataset}/
1_Data/Landmark_Sets/{ProtocolBucket}/{ExperimentName}/{Dataset}/
1_Data/Barcode_Statistics/{ProtocolBucket}/{ExperimentName}/{Dataset}/
```

`1_Data/Processed_Datasets/` is shared and is not re-bucketed.
