# Intrinsic Dimension Estimation — Default_Of_Credit_Card_Client_Data

## Dataset

Default of Credit Card Clients (Taiwan, UCI). About 30,000 people; target is default next month.

## What this folder is

How many degrees of freedom does this table have? We estimate b twice: before PCA (the scaled table) and after the same PCA Exp 3 uses (the space Ripser samples). Headline: Two-NN (hand-coded Facco formula and skdim). Secondary: Levina-Bickel, MiND_ML, lPCA. No barcodes needed. See docs/Design_Decisions.md.

This folder: run_intrinsic_dimension.py

The experiment-wide walkthrough (all six tables, findings, how to read numbers) is:

5_Experiments/Statistics/1_Intrinsic_Dimension_Estimation/REPORT.md

## Results

6_Results/Statistics/1_Intrinsic_Dimension_Estimation/Default_Of_Credit_Card_Client_Data/

Open the CSV first. Pickles are for Python follow-up, not for a first reading.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. `run.py` is an optional convenience launcher and is not the method document.
