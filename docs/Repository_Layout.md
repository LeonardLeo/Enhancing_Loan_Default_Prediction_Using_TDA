# Repository layout

A map of where code, figures, barcodes, paper tables, and queues live after the 2026 bucket restructure. This is the live layout. Old numbered folders such as `5_Experiments/3_PH_Default_Parameters` or `5_Experiments/23_Early_…` are **not** at the repository root of `5_Experiments/` any more.

Compact snapshot symbols from the methods literature are recorded once in `docs/Notation.md`. Stage order: `docs/Statistical_Approach_Flow.md`.

---

## Top-level buckets

`5_Experiments/` and `6_Results/` share the same top-level names:

| Bucket | What it holds |
|--------|----------------|
| `Default_Parameters/` | Tabular ML: `1_ML_Default_Parameters`, `2_ML_Tuned_Parameters` |
| `Historical_Late_Split_Balanced_TDA/` | Original TDA pipeline (full-table PCA, undersample, late 80/20 on barcode rows). Experiments 1–9 |
| `Early_Split_TDA/` | Split customers first; still undersample inside each split. Experiments 1–9 |
| `No_Undersampling/` | Late split, no majority downsample. Experiments 1–9 |
| `Early_Split_TDA_And_No_Undersampling/` | Early split + no undersample. Experiments 1–9 |
| `Statistics/` | `1_Intrinsic_Dimension_Estimation` (protocol-independent) |
| `Snapshot_Sample_Size/` | Dated 13/08/2026 sample-size study (`0_Shared_Pools`, `1_Snapshot_Count_Sweep`, `2_Points_Per_Snapshot_Sweep`, `3_Snapshot_Count_Across_Cloud_Sizes`) |
| `Archives/` | Retired experiments (old 5, 7–10, 12–18, 20–22). Original numbers kept. A museum, not the live factorial |

`6_Results/` also has:

| Extra folder | Role |
|--------------|------|
| `Paper_Tables/` | LaTeX/CSV tables written by `6_Results/results.py` |
| `Run_Queue/` | Ripser, consumer, and sample-size queues plus logs |

Root shims `6_Results/_ripser_queue.py` and `6_Results/_consumer_queue.py` forward to `Run_Queue/`. They are the public entry if an in-flight command still uses the old path. Prefer `6_Results/Run_Queue/`.

---

## Where the method is written

Every dataset folder contains a readable pipeline script. That file is the method document. `run.py` is an optional launcher.

| Kind of experiment | Typical script name |
|--------------------|---------------------|
| Tabular ML | `*_data.py` / `*_tuned.py` |
| PH default / drop-correlated / linear | `*_PH.py` (linear also has a thin `*_linear.py` launcher) |
| PH tuned | `*_PH_tuned.py` |
| H0-only | `*_H0_only.py` |
| Sampling-ratio audit | `*_audit.py` |
| Snapshot mean/variance | `*_mean_variance.py` |
| Algorithm 2 | `*_algorithm2.py` |
| Revised snapshot protocol | `*_protocol.py` |
| Intrinsic dimension | `run_intrinsic_dimension.py` |
| Snapshot sample size | `*_sample_size.py` / `*_shared_pools.py` |

Example:

```
5_Experiments/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py
```

Heavy Ripser and IO helpers live in `utils.py`. The sample-size study uses `5_Experiments/Snapshot_Sample_Size/sample_size_lib.py` the same way.

Active TDA experiments 1–9 inside every arm: PH default, PH tuned, H0-only, drop correlated, linear regression, sampling-ratio audit, snapshot mean/variance, Algorithm 2, revised snapshot protocol.

---

## Where figures live

All generated figures for an experiment land in **one** folder:

```
6_Results/{Bucket}/{Experiment}/Visualizations/
```

Examples:

- `6_Results/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Visualizations/`
- `6_Results/Statistics/1_Intrinsic_Dimension_Estimation/Visualizations/`
- `6_Results/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/Visualizations/`

Run `5_Experiments/{Bucket}/{Experiment}/visualize_results.py`. Do not look in `model_viz/`, `cv_viz/`, `cross_dataset_viz/`, or `plots/` — those names are retired on the active tree. The catalog is `6_Results/README.md`.

---

## Where barcodes and landmarks live

TDA artefacts are namespaced by protocol bucket and experiment:

```
1_Data/Landmark_Sets/{ProtocolBucket}/{Experiment}/{Dataset}/
1_Data/Barcode_Statistics/{ProtocolBucket}/{Experiment}/{Dataset}/
1_Data/TDA_Datasets/{ProtocolBucket}/{Experiment}/{Dataset}/
```

Processed tables are **shared** and are not re-bucketed: `1_Data/Processed_Datasets/{Dataset}/`.

Historical arm experiment 1 is the barcode factory for that arm. Experiments 2–5 and 7–8 read `data_L*.csv`; they must not regenerate 500 Ripser jobs.

---

## Where paper tables live

`6_Results/results.py` (run from `6_Results/`) writes:

- `6_Results/Paper_Tables/clean_experiment_results.csv`
- `6_Results/Paper_Tables/results_table.tex` and the per-dataset / per-paper-experiment `.tex` files

Paper experiment numbers 1–10 in those tables are **not** the same as arm experiment numbers 1–9. Mapping: root `README.md` § Experiments.

---

## Where queues live

| Command | Role |
|---------|------|
| `6_Results/Run_Queue/_ripser_queue.py` | Sequential Ripser / protocol queue |
| `6_Results/Run_Queue/_consumer_queue.py` | Train consumers once barcodes exist |
| `6_Results/Run_Queue/_snapshot_sample_size_queue.py` | Sample-size study queue (does not interleave the historical queue) |

Logs and registry JSON/CSV sit beside those scripts.

---

## Historical checklist numbers (do not use as folder names)

| Old root folder people still mention | Live home |
|--------------------------------------|-----------|
| `1_ML_Default_Parameters` | `Default_Parameters/1_ML_Default_Parameters` |
| `3_PH_Default_Parameters` | `Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters` |
| `23_Early_…` | `Early_Split_TDA/1_PH_Default_Parameters` |
| `24_…` / `25_…` / `27_…` | arm experiments 6 / 7 / 8 |
| `26_Intrinsic_…` | `Statistics/1_Intrinsic_Dimension_Estimation` |
| `28_Revised_…` | `{TDA arm}/9_Revised_Snapshot_Protocol` |

`Archives/` keeps original numbers for retired work on purpose.
