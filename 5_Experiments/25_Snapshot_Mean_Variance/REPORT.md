# Experiment 25 — Are barcode features stable across snapshots?

## In one sentence

If mean death (and the other 23 barcode numbers) jump around from snapshot to snapshot, we are measuring noise, not a class fingerprint.

## Who this is for

Suppose you photograph a crowd 500 times, each time with a slightly different subset of people. If “average height in the photo” is almost the same every time, that statistic is stable. If it swings wildly, one photo tells you nothing about the crowd.

## Datasets (all six)

Scripts live in each dataset folder, including **Statlog** and **DCCCD** (they were missing from this layout before):

`5_Experiments/25_Snapshot_Mean_Variance/<Dataset>/run_snapshot_mean_variance.py`

**Prerequisite:** Experiment 3 `data_L*.csv` for that dataset. If the file is missing, the script prints a clear message and exits without crashing the rest of the batch.

| Dataset | Files this script expects |
|---------|---------------------------|
| DCCCD | `data_L5.csv`, `data_L15.csv` |
| Statlog | `data_L30.csv`, `data_L60.csv` |
| PKDD, Polish, Taiwan, South German | `data_L10.csv`, `data_L20.csv` |

## What we do (in order)

1. Load each Experiment 3 matrix (1,000 rows × 24 barcode columns + label).
2. For every column, compute the mean and the sample variance across snapshots.
3. Store the 24-number mean vector as a cheap proxy for a persistence-landscape average. This is **not** the landscape itself (Chazal et al.); say that when you cite it.
4. Write a flat CSV: one row per (file, feature).

## What we found

Full numbers: `docs/Statistical_Experiments_24_27_Results.md`.

- **DCCCD / Statlog:** already had Exp 3 matrices; both L-files loaded (1,000 × 24).
- **PKDD L10** Mean Death H0 (`g2_0`) mean 1.66, variance **0.045** — an order of magnitude noisier than DCCCD. Seven-point snapshots on 76 people wiggle. L20 variance drops to 0.011.
- **Polish:** Mean Death H0 is tiny (~0.019) with small variance; H1 persistences are ~0.001. The PCA cloud is very connected.
- **Taiwan:** `g2_0` 0.75 (L10) / 0.65 (L20); variances 0.003 / 0.001.
- **South German:** `g2_0` 0.97 (L10) / 0.85 (L20); variances 0.003 / 0.001.

How to interpret a row: **small variance + different means by class** = a stable, useful fingerprint. **Large variance** = successive snapshots disagree, so a classifier is fitting remix noise (see Experiment 24). Larger `L` generally shrinks variance and raises reuse — those two facts belong in the same sentence.

## Where the files live

`6_Results/25_Snapshot_Mean_Variance/{Folder}/snapshot_mean_variance.csv`
