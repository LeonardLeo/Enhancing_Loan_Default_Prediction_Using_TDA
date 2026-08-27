# Repository layout

A map of where code, figures, barcodes, paper tables, and queues live after the eight named processes. This is the live layout. Old numbered folders such as `5_Experiments/3_PH_Default_Parameters` or `5_Experiments/23_Early_…` are **not** at the repository root of `5_Experiments/` any more.

Compact snapshot symbols from the methods literature are recorded once in `docs/Notation.md`. Stage order: `docs/Statistical_Approach_Flow.md`. Public process names always use “and”, never “+”, and are defined once in `utils.TDA_PROCESS_REGISTRY`.

---

## Top-level buckets

`5_Experiments/` and `6_Results/` share the same top-level names:

| Bucket | What it holds |
|--------|----------------|
| `Default_Parameters/` | Tabular ML: `1_ML_Default_Parameters`, `2_ML_Tuned_Parameters` |
| `Early_Split_And_Undersample_H0/` | Early split and undersample, using just H0 |
| `Early_Split_And_Undersample_H0_And_H1/` | Early split and undersample, using both H0 and H1 |
| `Early_Split_No_Undersample_H0/` | Early split, no undersample, using just H0 |
| `Early_Split_No_Undersample_H0_And_H1/` | Early split, no undersample, using both H0 and H1 |
| `Late_Split_And_Undersample_H0/` | Late split and undersample (the original historical run), using just H0 |
| `Late_Split_And_Undersample_H0_And_H1/` | Late split and undersample (the original historical run), using both H0 and H1 |
| `Late_Split_No_Undersample_H0/` | Late split, no undersample, using just H0 |
| `Late_Split_No_Undersample_H0_And_H1/` | Late split, no undersample, using both H0 and H1 |
| `Statistics/` | `1_Intrinsic_Dimension_Estimation` (protocol-independent) |
| `Snapshot_Sample_Size/` | Dated 13/08/2026 sample-size study. Shared pools still use the four historical protocol keys |
| `Archives/` | Retired experiments (old 5, 7–10, 12–18, 20–22) and `Four_Arm_Nested_Experiments/` |

`6_Results/` also has:

| Extra folder | Role |
|--------------|------|
| `Paper_Tables/` | LaTeX/CSV tables written by `6_Results/results.py` |
| `Run_Queue/` | Ripser, consumer, and sample-size queues plus logs |

Root shims `6_Results/_ripser_queue.py` and `6_Results/_consumer_queue.py` forward to `Run_Queue/`. They are the public entry if an in-flight command still uses the old path. Prefer `6_Results/Run_Queue/`.

H0-and-H1 folders keep default classifiers, retuned classifiers, sampling-ratio audit, Algorithm 2, and the revised snapshot protocol. H0 folders keep default classifiers and Algorithm 2 on sliced H0 tables. Sampling-ratio audit and revised snapshot protocol are not duplicated eight ways — they live next to the H0-and-H1 folder for that split/undersample pair.

---

## Where the method is written

Every dataset folder contains a readable pipeline script. That file is the method document. There is no wrapper launcher: open and run the named dataset script.

| Kind of experiment | Typical script name |
|--------------------|---------------------|
| Tabular ML | `*_data.py` / `*_tuned.py` |
| PH default (H0 and H1) | `*_PH.py` |
| PH tuned | `*_PH_tuned.py` |
| H0-only default | `*_H0_only.py` |
| Sampling-ratio audit | `*_audit.py` |
| Algorithm 2 | `*_algorithm2.py` |
| Revised snapshot protocol | `*_protocol.py` |
| Intrinsic dimension | `run_intrinsic_dimension.py` |
| Snapshot sample size | Each numbered folder's `*_sample_size.py` is the method for that figure (load, PCA, snapshots, Ripser, train). |

Example:

```
5_Experiments/Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py
5_Experiments/Late_Split_And_Undersample_H0/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_H0_only.py
5_Experiments/Snapshot_Sample_Size/0_Shared_Pools/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_shared_pools.py
```

Heavy Ripser and IO helpers live in `utils.py`. The sample-size study uses `5_Experiments/Snapshot_Sample_Size/utils.py` the same way. To see how barcodes were built for the sample-size study, open `0_Shared_Pools`. The numbered experiment scripts only select which rows go on which figure.

---

## Where figures live

All generated figures for an experiment land in **one** folder:

```
6_Results/{Bucket}/{Experiment}/Visualizations/
```

Examples:

- `6_Results/Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters/Visualizations/`
- `6_Results/Late_Split_And_Undersample_H0/1_PH_Default_Parameters/Visualizations/`
- `6_Results/Statistics/1_Intrinsic_Dimension_Estimation/Visualizations/`
- `6_Results/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/Visualizations/`

Run `5_Experiments/{Bucket}/{Experiment}/visualize_results.py`. Do not look in `model_viz/`, `cv_viz/`, `cross_dataset_viz/`, or `plots/` — those names are retired on the active tree. The catalog is `6_Results/README.md`.

---

## Where barcodes and landmarks live

TDA artefacts are namespaced by process folder and experiment:

```
1_Data/Landmark_Sets/{ProtocolBucket}/{Experiment}/{Dataset}/
1_Data/Barcode_Statistics/{ProtocolBucket}/{Experiment}/{Dataset}/
1_Data/TDA_Datasets/{ProtocolBucket}/{Experiment}/{Dataset}/
```

Processed tables are **shared** and are not re-bucketed: `1_Data/Processed_Datasets/{Dataset}/`.

The four `*_H0_And_H1` experiment-1 folders are the barcode factories. H0 processes read those tables and write 12-statistic slices under their own `1_PH_Default_Parameters`. They must not regenerate Ripser jobs.

---

## Where paper tables live

`6_Results/results.py` (run from `6_Results/`) writes:

- `6_Results/Paper_Tables/clean_experiment_results.csv`
- `6_Results/Paper_Tables/results_table.tex` and the per-dataset / per-paper-experiment `.tex` files

Paper experiment numbers 1–10 in those tables are **not** the same as arm experiment numbers. Mapping: root `README.md` § Experiments.

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
| `3_PH_Default_Parameters` | `Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters` |
| `3_H0_Only` (just H0 default models) | `Late_Split_And_Undersample_H0/1_PH_Default_Parameters` |
| `23_Early_…` | `Early_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters` |
| `24_…` | `{H0-and-H1 process}/6_Sampling_Ratio_Audit` |
| `25_…` | `Archives/Four_Arm_Nested_Experiments/{old arm}/7_Snapshot_Mean_Variance` |
| `26_Intrinsic_…` | `Statistics/1_Intrinsic_Dimension_Estimation` |
| `27_…` | `{process}/8_Null_Hypothesis_Algorithm2` |
| `28_Revised_…` | `{H0-and-H1 process}/9_Revised_Snapshot_Protocol` |
| `Historical_Late_Split_Balanced_TDA` | `Late_Split_And_Undersample_H0_And_H1` |
| `Early_Split_TDA` | `Early_Split_And_Undersample_H0_And_H1` |
| `No_Undersampling` | `Late_Split_No_Undersample_H0_And_H1` |
| `Early_Split_TDA_And_No_Undersampling` | `Early_Split_No_Undersample_H0_And_H1` |

`Archives/` keeps original numbers for retired work on purpose. Nested four-arm extras (drop-correlated, linear, mean/variance, nested H0 copies) are under `Archives/Four_Arm_Nested_Experiments/`.
