# Experiment 23 — Split **before** PCA and landmarks (Protocol B)

## In one sentence

Experiment 3 fits the scaler and PCA on **everyone**, then samples landmarks, then only later splits barcode rows. Test customers have already influenced the axes and may appear in train snapshots.

## Who this is for

If you are writing a paper, this is the historical-TDA protocol you can defend. Experiment 3 is what the original pipeline did. Experiment 28 goes further (no undersampling, fixed t, smaller l).

## Datasets

All six.

## What we do (in order)

1. Stratified 80/20 split of **people**.
2. Fit scaler + PCA on train only; transform test.
3. Undersample each split separately.
4. Draw landmarks for train and test **independently**.
5. Train on train barcodes, test on test barcodes.

## What we look for

A drop vs Experiment 3 is expected if Exp 3 was leaking. Quote both.

## Results

`6_Results/23_Early_Train_Test_Split/{Folder}/`
