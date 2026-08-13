# Experiment 21 — Mapper graphs on the barcode table

## In one sentence

Mapper (Singh, Mémoli, Carlsson) draws a cartoon of the 24-D snapshot cloud: cover a 2D lens with overlapping bins, cluster inside each bin, connect overlapping clusters.

## Who this is for

Imagine laying a piece of tracing paper with overlapping circles on a scatter plot, clustering the points in each circle, and drawing a node per cluster. The resulting graph is a **simplified shape**. We colour nodes by default vs non-default. Mixed-colour nodes = the two classes share that region of barcode space.

## Datasets

All six. **Prerequisite:** Experiment 3 `data_L*.csv`. No landmark rebuild.

## What we do

Grid: resolution {20, 30, 40} × overlap {0.3, 0.4, 0.5} × lens {PCA, UMAP} × k-means (2 clusters). HTML files land under

`6_Results/21_Visualizing_Data_Shape_For_Barcode_Statistics_Using_TDA/{Folder}/L{percent}/`

Folder names are shortened (`pca_r20_o30_kmeans2`) so Windows path-length limits do not break the write.

## How to read an HTML

Open it in a browser. Node colour ≈ class mix. A graph that splits into two almost-pure blobs is the picture you want. A single mixed blob is the picture you should not over-claim.

## Windows note

The original nested folder names exceeded MAX_PATH on this machine. New-dataset runs use short `L10` / `L20` directories plus short experiment tags.
