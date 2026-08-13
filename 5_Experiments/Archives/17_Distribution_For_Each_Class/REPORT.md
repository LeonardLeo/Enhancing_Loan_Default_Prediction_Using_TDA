# Experiment 17 — Do the two classes occupy different regions of barcode space?

## In one sentence

After Experiment 3 squashes each snapshot into 24 numbers, do defaults and non-defaults form two visible clouds?

## Who this is for

If the two colours sit on top of each other in PCA, t-SNE, **and** UMAP, topology is not giving the classifier an easy visual signal. A later model might still find a thin wrinkle — but you should not claim “clear shape difference” from these plots.

## Datasets

All six. New-dataset scripts are staged (load → 2D → static 3D) and use repo-root paths. Rotating MP4/GIF exports from the original Statlog script are **skipped** on the new tables: they need ffmpeg, take a long time, and are optional extras.

**Prerequisite:** Experiment 3 `data_L*.csv`.

## What we do (in order)

1. Load `data_L10.csv` / `data_L20.csv` (or Statlog / DCCCD percents).
2. Project to 2D with PCA, t-SNE, and UMAP.
3. Repeat with a kernel-density overlay so overlap is easier to see.
4. Repeat in 3D (static figures).

## How to read the figures

Two colours = two classes. Density blobs that overlap = the classes share the same region. Separate blobs = a human could almost classify by eye.

## Results

`6_Results/Archives/17_Distribution_For_Each_Class/{Folder}/`
