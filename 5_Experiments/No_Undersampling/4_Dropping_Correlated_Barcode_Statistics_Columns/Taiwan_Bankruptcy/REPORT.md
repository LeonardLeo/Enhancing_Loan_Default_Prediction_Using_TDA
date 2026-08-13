# Taiwan_Bankruptcy — No_Undersampling / 4_Dropping_Correlated_Barcode_Statistics_Columns

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on the unbalanced pool |

CONSUMER. Drops correlated barcode columns (threshold 0.80) from experiment 1, then retrains.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/4_Dropping_Correlated_Barcode_Statistics_Columns/Taiwan_Bankruptcy/run.py
```
