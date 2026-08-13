# Statlog_German_Credit_Data — Historical_Late_Split_Balanced_TDA / 5_Linear_Regression_For_Prediction

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| L percents | L30 / L60 |
| PCA rank | 15 |
| `l` | 500 |
| `t` | t = floor(n1 * L / 100) with n1 = n2 = minority count |

CONSUMER. Linear regression on the H0 slice of experiment 1.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/5_Linear_Regression_For_Prediction/Statlog_German_Credit_Data/run.py
```
