# Statlog_German_Credit_Data — Early_Split_TDA / 8_Null_Hypothesis_Algorithm2

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L30 / L60 |
| PCA rank | 15 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) on the undersampled pool of that split |

CONSUMER. Robinson–Turner Algorithm 2 on this arm's barcode matrices.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/8_Null_Hypothesis_Algorithm2/Statlog_German_Credit_Data/run.py
```
