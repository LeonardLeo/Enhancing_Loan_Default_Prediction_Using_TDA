# Experiment 14 — Stop balancing the snapshot counts

## In one sentence

Experiment 3 writes 500 default + 500 non-default files. Real life is imbalanced. Here we write something like 200 default files and 800 non-default files and train anyway.

## Who this is for

A 50/50 barcode table is a classroom convenience. Lenders do not see 50/50. If TDA only works on the balanced snapshot matrix, say so.

This **does** generate a different landmark collection (`generate_landmark_sets_v2`). It is not a consumer of Exp 3 files.

## Datasets

All six.

## What we look for

Recall on the rare class. Accuracy will look high if the model predicts “everyone is fine”.

## Results

`6_Results/Archives/14_Mixed_Classes_Training_With_Imbalanced_Datasets/{Folder}/`
