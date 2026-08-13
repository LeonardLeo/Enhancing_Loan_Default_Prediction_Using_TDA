# Taiwan_Bankruptcy — Early_Split_TDA_And_No_Undersampling / 6_Sampling_Ratio_Audit

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | no — full class pools inside each split |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on that split's available pool |

No Ripser. Scores R = (t * l) / n_class from the protocol's class pools, L percents, and l = 500.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA_And_No_Undersampling/6_Sampling_Ratio_Audit/Taiwan_Bankruptcy/run.py
```
