# PKDD_Czech_Financial — Early_Split_TDA / 4_Dropping_Correlated_Barcode_Statistics_Columns

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) on the undersampled pool of that split |

CONSUMER. Drops correlated barcode columns (threshold 0.80) from experiment 1, then retrains.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/PKDD_Czech_Financial/run.py
```
