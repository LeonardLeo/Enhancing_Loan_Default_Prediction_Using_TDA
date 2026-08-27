# Late_Split_And_Undersample_H0_And_H1 — Experiment 9 (revised snapshot protocol)

This is arm experiment 9 in **Historical Late Split Balanced TDA** (full-table PCA, undersample, then a late split). The same engine exists in all four TDA arms. The meeting-driven write-up that originally used the label Experiment 28 lives at `docs/Revised_Snapshot_Protocol_Deep_Report.md`. There is no live top-level `28_Revised_…` folder.

## In one sentence

This experiment uses a **fixed absolute points-per-snapshot value** for train and test, default **60 training snapshots / 15 test snapshots**, and reports the formula snapshot count separately from the reuse-ratio constraint.

## This arm's knobs

- Late split + undersample (majority down to minority before PCA/split)
- Fixed absolute points per snapshot (same for train and test)
- Default train/test snapshot counts: **60 / 15**
- Sensitivity sweep: train `{60, 80, 100}`, test `{15, 22, 30}`
- Default of Credit Card Client non-split arm: snapshot counts in {60, 75, 90}
- Formula concern and reuse-ratio concern reported separately
- Overlap: pairwise + reuse + significance tests

## Where the code lives

```
5_Experiments/Late_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_protocol.py
5_Experiments/Late_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/Statlog_German_Credit_Data/statlog_german_credit_protocol.py
…
```

Each dataset `*_protocol.py` is the method document. Helpers:

- `utils.py` — sampling, overlap diagnostics, models
- `run_protocol.py` — stages `design` → `split_ml` → `full_ml`

## Stages

1. **design** — estimate intrinsic dimension, choose a joint points-per-snapshot value, print the reuse ratio.
2. **split_ml** — apply this arm's split and undersample; draw independent snapshots; fit models.
3. **full_ml** — Default of Credit Card Client extra arm that skips the customer split (documented in the launcher).

## Findings

Numeric dump: `6_Results/Late_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/` (including `all_designs.json`). Figures: `6_Results/Late_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/Visualizations/`.

Sampling-ratio audit (arm experiment 6) showed 500 snapshots over-reuse every table (reuse ratio 25–300). Intrinsic dimension (Statistics experiment 1) after PCA is 2.8–4.9, not 7. Arm experiment 9 is where those two facts change the protocol. Flow: `docs/Statistical_Approach_Flow.md`. English names: `docs/Notation.md`.

## How to run

```
.\tda_env\Scripts\python.exe 5_Experiments/Late_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/run_protocol.py --stage all
.\tda_env\Scripts\python.exe 5_Experiments/Late_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/run_protocol.py --stage design
.\tda_env\Scripts\python.exe 5_Experiments/Late_Split_And_Undersample_H0_And_H1/9_Revised_Snapshot_Protocol/Statlog_German_Credit_Data/statlog_german_credit_protocol.py
```

## Where to read the method

Open the dataset `*_protocol.py` in this folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_protocol.py`). That file is the method document.  Helpers live in `utils.py`.
