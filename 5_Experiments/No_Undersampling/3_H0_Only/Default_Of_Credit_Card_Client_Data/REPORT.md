# Default_Of_Credit_Card_Client_Data — No_Undersampling / 3_H0_Only

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L5 / L15 |
| PCA rank | 7 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on the unbalanced pool |

CONSUMER. Keeps H0 (`*_0`) columns from experiment 1 and retrains default classifiers. Does not regenerate Ripser.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/3_H0_Only/Default_Of_Credit_Card_Client_Data/run.py
```
