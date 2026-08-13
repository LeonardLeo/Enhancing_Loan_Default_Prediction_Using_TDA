# Experiment 28 — Rebuild sampling the way the meeting asked

## In one sentence

Keep the TDA idea, but stop cheating the sample: **no undersampling**, **one absolute snapshot size t** for train and test, default **60 train snapshots / 15 test snapshots**, and report formula-`l` vs reuse-`l` separately.

## Who this is for

Experiments 3–27 mostly follow the historical paper protocol (undersample, 500 overlapping snapshots, PCA on the full table). This folder is the **revised protocol** from the team discussion. If you are writing the methods section for “what we would do next time”, start here.

## Datasets (all six)

Each dataset folder is a readable launcher, including **Statlog** and **DCCCD**:

```
5_Experiments/Historical_Late_Split_Balanced_TDA/9_Revised_Snapshot_Protocol/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_protocol.py
5_Experiments/Historical_Late_Split_Balanced_TDA/9_Revised_Snapshot_Protocol/Statlog_German_Credit_Data/statlog_german_credit_protocol.py
5_Experiments/Historical_Late_Split_Balanced_TDA/9_Revised_Snapshot_Protocol/PKDD_Czech_Financial/pkdd_czech_financial_protocol.py
...
```

The heavy lifting is **not** duplicated in those files. They call:

- `protocol_lib.py` — sampling, overlap diagnostics, models
- `run_protocol.py` — stages `design` → `split_ml` → `full_ml`

Open the launcher first (it prints dataset key and stage), then step through `run_protocol.py` if you need to debug.

## What we do (stages)

1. **design** — estimate intrinsic dimension, choose a joint t, print reuse R.
2. **split_ml** — split customers into train/test **before** snapshots; draw independent snapshots; fit models.
3. **full_ml** — DCCCD-only extra arm that skips the split (documented in the launcher).

## What we found

Numeric dump: `6_Results/Historical_Late_Split_Balanced_TDA/9_Revised_Snapshot_Protocol/` (including `all_designs.json`).

The design JSON is large because it stores per-dataset t, estimated b, and overlap diagnostics. Read `docs/Revised_Snapshot_Protocol_Deep_Report.md` for the narrative; this REPORT is the map of *where the code lives* and *what each stage means*.

Compared with Experiment 3: you should expect **fewer** barcode rows and **honest** train/test separation. Accuracy may drop. That drop is information, not a failure.

**Why this folder exists in the statistical sequence:** Exp 24 showed `l = 500` over-reuses every table (R = 25–300). Exp 26 estimated `b` before and after PCA (after-PCA Two-NN is 2.8–4.9, not 7). Exp 28 is where those two facts change the protocol: fixed `t`, small `l`, customers split first, no undersampling. Flow: `docs/Statistical_Approach_Flow.md`.

## How to re-run one dataset

```
python 5_Experiments/Historical_Late_Split_Balanced_TDA/9_Revised_Snapshot_Protocol/Statlog_German_Credit_Data/statlog_german_credit_protocol.py
```
