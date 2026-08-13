# Statlog_German_Credit_Data — Historical_Late_Split_Balanced_TDA / 8_Null_Hypothesis_Algorithm2

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| L percents | L30 / L60 |
| PCA rank | 15 |
| `l` | 500 |
| `t` | t = floor(n1 * L / 100) with n1 = n2 = minority count |

CONSUMER. Robinson–Turner Algorithm 2 on this arm's barcode matrices.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/8_Null_Hypothesis_Algorithm2/Statlog_German_Credit_Data/run.py
```
