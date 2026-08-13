# Statlog_German_Credit_Data — No_Undersampling / 8_Null_Hypothesis_Algorithm2

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L30 / L60 |
| PCA rank | 15 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on the unbalanced pool |

CONSUMER. Robinson–Turner Algorithm 2 on this arm's barcode matrices.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/8_Null_Hypothesis_Algorithm2/Statlog_German_Credit_Data/run.py
```
