# Experiment 5 — Mapper graphs on the original customer table

## In one sentence

Draw a cartoon of the **original** feature cloud (not barcodes). Experiment 21 repeats this idea on barcode rows.

## Who this is for

Mapper is not a classifier. It is a sketch: pour the table through a lens (usually PCA or UMAP), slice into overlapping bins, cluster, connect. Colour by default status. Mixed-colour nodes mean that lens does not separate the classes.

## Datasets

All six. Statlog and DCCCD have large existing HTML galleries. Newer tables use `build_mapper_viz` from `utils.py`.

## How to look at a result

Open the HTML in a browser. Tight blobs of one colour = that class occupies a region. Mixed colours = the lens does not separate defaults.

## Results

`6_Results/5_Mapper/{Folder}/`
