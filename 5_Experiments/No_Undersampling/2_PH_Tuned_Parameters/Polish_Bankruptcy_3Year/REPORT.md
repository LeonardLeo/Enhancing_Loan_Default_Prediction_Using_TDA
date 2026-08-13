# Polish_Bankruptcy_3Year — No_Undersampling / 2_PH_Tuned_Parameters

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on the unbalanced pool |

CONSUMER. GridSearchCV on this arm's experiment-1 barcode matrices. Does not regenerate Ripser.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/2_PH_Tuned_Parameters/Polish_Bankruptcy_3Year/run.py
```
