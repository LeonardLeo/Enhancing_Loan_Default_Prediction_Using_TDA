# Revised Snapshot Protocol — PKDD_Czech_Financial

## Dataset

PKDD 1999 Czech Financial (Berka). 682 loans with pre-loan bank activity; target is default vs repaid.

## What this folder is

Revised sampling: no undersampling, fixed t, 60/15 snapshots, split before TDA. Launcher in this folder calls protocol_lib / run_protocol.py.

This folder: the *_protocol.py launcher (see the file name in this directory).

The experiment-wide walkthrough (all six tables, findings, how to read numbers) is:

5_Experiments/Early_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/REPORT.md

## Results

6_Results/Early_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/PKDD_Czech_Financial/

Open the CSV first. Pickles are for Python follow-up, not for a first reading.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_protocol.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
