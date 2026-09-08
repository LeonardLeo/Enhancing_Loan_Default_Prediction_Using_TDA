# Experiment 15 — Which k should k-NN use on barcode rows?

## In one sentence

Library-default k-NN uses k=5. We sweep k on the Experiment 3 barcode table and pick the elbow / best validation k.

## Who this is for

k-NN labels a new snapshot by asking its k nearest training snapshots. Too small k = jumpy; too large k = it just votes the majority class. The elbow plot is the teaching picture.

## Datasets

Both live tables (DCCCD and Statlog). **Prerequisite:** paper Experiment 3 `data_L*.csv` from `Late_Split_And_Undersample_H0_And_H1/1_PH_Default_Parameters`. No landmark rebuild.

Older drafts quoted L10 / L20 on a third table (best k = 16 acc 0.46; best k = 11 acc 0.59). Those files are not in this tree. The on-disk `results.json` files are L5 / L15 / L30 / L60.

## Results (from `results.json` on this machine)

| Dataset | Setting | best_k | Accuracy | Precision | Recall | F1 |
|---------|---------|-------:|---------:|----------:|-------:|---:|
| Statlog German Credit | L30 | 11 | 0.710 | 0.667 | 0.840 | 0.743 |
| Statlog German Credit | L60 | 15 | 0.810 | 0.777 | 0.870 | 0.821 |
| Default of Credit Card Client | L5 | 13 | 0.985 | 1.000 | 0.970 | 0.985 |
| Default of Credit Card Client | L15 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

These sit on the leaky late-split barcode tables. Perfect DCCCD L15 k=1 is the same leakage story as paper Experiment 3, not a k-NN discovery.

`6_Results/Archives/15_Working_With_K_in_KNN/{Folder}/data_L{percent}/elbow_curve.png` plus `results.json`.
