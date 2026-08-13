# PKDD_Czech_Financial — Early_Split_TDA / 8_Null_Hypothesis_Algorithm2

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) on the undersampled pool of that split |

CONSUMER. Robinson–Turner Algorithm 2 on this arm's barcode matrices.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/8_Null_Hypothesis_Algorithm2/PKDD_Czech_Financial/run.py
```
