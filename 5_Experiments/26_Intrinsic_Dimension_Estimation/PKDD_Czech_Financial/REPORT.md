# Intrinsic Dimension Estimation — PKDD_Czech_Financial

## Dataset

PKDD 1999 Czech Financial (Berka). 682 loans with pre-loan bank activity; target is default vs repaid.

## What this folder is

How many degrees of freedom does this table have? We estimate b twice: before PCA (the scaled table) and after the same PCA Exp 3 uses (the space Ripser samples). Headline: Two-NN (hand-coded Facco formula and skdim). Secondary: Levina-Bickel, MiND_ML, lPCA. No barcodes needed. See docs/Design_Decisions.md.

This folder: run_intrinsic_dimension.py

The experiment-wide walkthrough (all six tables, findings, how to read numbers) is:

5_Experiments/26_Intrinsic_Dimension_Estimation/REPORT.md

## Results

6_Results/26_Intrinsic_Dimension_Estimation/PKDD_Czech_Financial/

Open the CSV first. Pickles are for Python follow-up, not for a first reading.
