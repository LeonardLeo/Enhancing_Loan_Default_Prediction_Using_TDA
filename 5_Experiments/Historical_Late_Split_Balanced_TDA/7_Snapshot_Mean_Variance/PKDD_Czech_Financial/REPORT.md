# PKDD_Czech_Financial — Historical_Late_Split_Balanced_TDA / 7_Snapshot_Mean_Variance

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n1 * L / 100) with n1 = n2 = minority count |

CONSUMER. Mean and variance of each barcode column in this arm's `data_L*.csv`.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/7_Snapshot_Mean_Variance/PKDD_Czech_Financial/run.py
```
