# PKDD_Czech_Financial — No_Undersampling / 6_Sampling_Ratio_Audit

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on the unbalanced pool |

No Ripser. Scores R = (t * l) / n_class from the protocol's class pools, L percents, and l = 500.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/6_Sampling_Ratio_Audit/PKDD_Czech_Financial/run.py
```
