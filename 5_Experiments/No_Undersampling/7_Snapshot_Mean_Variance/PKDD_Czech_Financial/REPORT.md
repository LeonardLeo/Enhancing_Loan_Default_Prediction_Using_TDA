# PKDD_Czech_Financial — No_Undersampling / 7_Snapshot_Mean_Variance

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on the unbalanced pool |

CONSUMER. Mean and variance of each barcode column in this arm's `data_L*.csv`.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/7_Snapshot_Mean_Variance/PKDD_Czech_Financial/run.py
```
