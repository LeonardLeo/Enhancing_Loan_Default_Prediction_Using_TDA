# Experiment 6 — What if we throw away loops?

## In one sentence

Persistent homology tracks connected pieces (H0) and loops (H1). This experiment keeps only H0 (12 numbers instead of 24).

## Who this is for

Loops are the “holes” in the point cloud. If F1 barely drops without them, we were mostly measuring clumpiness (H0). If F1 collapses, the holes were carrying the signal.

This **does** rebuild barcodes with `maxdim` restricted — it is one of the few experiments that needs the full Ripser flow, not a consumer of Exp 3 files. Experiment 19 on the new datasets is the cheap cousin: it slices H0 columns out of Exp 3 instead of recomputing.

## Datasets

All six. Landmark percents match Experiment 3.

## What we look for

Exp 3 F1 minus this F1. Near zero → H1 was cosmetic. Large drop → cite H1 in the paper.

## Results

`6_Results/6_Experiment_Impact_of_H0_Only/{Folder}/`
