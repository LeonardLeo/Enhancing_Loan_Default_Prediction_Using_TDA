# Experiment 28 — Revised Snapshot Protocol

Meeting-driven redesign of landmark sampling.

## Rules

- This arm: early split + undersample (independently inside train and test)
- Fixed absolute `t` (same train/test)
- Default train/test snapshot counts: **60 / 15**
- Zaniar sweep (3 points): train `{60,80,100}`, test `{15,22,30}`
- DCCCD non-split arm: `l ∈ {60,75,90}`
- Concern A (formula) and Concern B (reuse) reported separately
- Overlap: pairwise + reuse + significance tests

## Run

```powershell
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/9_Revised_Snapshot_Protocol/run_protocol.py --stage all
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/9_Revised_Snapshot_Protocol/run_protocol.py --stage design
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/9_Revised_Snapshot_Protocol/run_protocol.py --stage split_ml --datasets credit_card_default
```

## Reports

- Deep narrative: `docs/Revised_Snapshot_Protocol_Deep_Report.md`
- Numeric outputs: `6_Results/Early_Split_TDA/9_Revised_Snapshot_Protocol/`
