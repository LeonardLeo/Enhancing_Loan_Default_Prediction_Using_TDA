# Experiment 11 — Drop redundant barcode columns and retrain

## In one sentence

Several of the 24 stats move together (mean death vs median death). We drop pairs with |correlation| above 0.80 and ask whether F1 stays the same with fewer columns.

## Who this is for

If two thermometers always agree, you only need one. Same idea here: correlated barcode stats are duplicate thermometers. Keeping both can even hurt some models.

## Datasets

All six. **Prerequisite:** Experiment 3 `data_L*.csv`. Paths are anchored at the repo root so the script does not depend on the working directory.

## What we do (in order)

1. Load `data_L10.csv` / `data_L20.csv` (or the Statlog / DCCCD percents).
2. Rename `g*_` columns to readable names (“Mean Death (Dim 0)”, …).
3. Shuffle, stratified 80/20 split, MinMax-scale (fit on train only).
4. Drop correlated features two ways: keep the higher-variance member of a pair, or keep the member more correlated with the label.
5. Draw correlation graphs of what was dropped.
6. Retrain the five default classifiers on the reduced tables.

## What we look for

If F1 matches Experiment 3 with fewer columns, the extra stats were cosmetic. If F1 collapses, we needed the “duplicate” thermometers after all (or the drop rule was too aggressive).

## Results

`6_Results/11_Dropping_Correlated_Barcode_Statistics_Columns/{Folder}/`
Reduced CSVs also land under `1_Data/TDA_Datasets/{Folder}/11_Dropping_Correlated_Barcode_Statistics_Columns/`.
