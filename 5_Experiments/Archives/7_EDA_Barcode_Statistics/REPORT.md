# Experiment 7 — Look at the barcode numbers themselves

## In one sentence

Before we train anything else, do default snapshots have systematically larger persistence? Are some of the 24 columns almost constant?

## Who this is for

This is ordinary table EDA on the **barcode** table, not on customers. If a column is always 0.0, it cannot help a classifier. If default rows have a much larger “mean death”, that is a hint topology saw a different scale.

## Datasets

All six. We describe:

- per-class barcode frames from Experiment 3 (`barcode_stats_default_L*.csv`, `barcode_stats_non-default_L*.csv`)
- the combined `data_L*.csv`
- Experiment 6 H0-only frames **if they exist** (skipped with a printed note otherwise)

**Prerequisite:** Experiment 3 artefacts. This script does **not** rebuild landmarks.

## What we do

Run the same `eda` helper used in Experiment 1 (describe, missingness, simple plots) and save the dict under `6_Results/Archives/7_EDA_Barcode_Statistics/{Folder}/`.



## Results

`6_Results/Archives/7_EDA_Barcode_Statistics/{Folder}/`
