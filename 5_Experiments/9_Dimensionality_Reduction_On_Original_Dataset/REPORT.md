# Experiment 9 — PCA / t-SNE / UMAP on the original table

## In one sentence

The ordinary 2D picture of the **customers**, not of barcode snapshots. Experiment 8 is the barcode version; Experiment 17 is a richer barcode version.

## Who this is for

If the two classes already separate on the raw table in 2D, TDA has less room to help. If they overlap completely, any later TDA win is coming from higher-order shape, not from a plot you could have made in Excel.

## Datasets

All six. Newer scripts use the processed `target` column. The historical Statlog script passed `target_column="label"` while the column is actually `Class` — check that if a Statlog plot looks empty.

## Results

`6_Results/9_Dimensionality_Reduction_On_Original_Dataset/{Folder}/`
