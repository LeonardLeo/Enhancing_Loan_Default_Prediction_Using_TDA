# Statlog_German_Credit_Data — Early_Split_TDA_And_No_Undersampling / 2_PH_Tuned_Parameters

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | no — full class pools inside each split |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L30 / L60 |
| PCA rank | 15 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on that split's available pool |

CONSUMER. GridSearchCV on this arm's experiment-1 barcode matrices. Does not regenerate Ripser.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA_And_No_Undersampling/2_PH_Tuned_Parameters/Statlog_German_Credit_Data/run.py
```
