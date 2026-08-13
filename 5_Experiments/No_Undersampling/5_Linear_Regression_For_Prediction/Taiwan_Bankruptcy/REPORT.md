# Taiwan_Bankruptcy — No_Undersampling / 5_Linear_Regression_For_Prediction

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on the unbalanced pool |

CONSUMER. Linear regression on the H0 slice of experiment 1.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/5_Linear_Regression_For_Prediction/Taiwan_Bankruptcy/run.py
```
