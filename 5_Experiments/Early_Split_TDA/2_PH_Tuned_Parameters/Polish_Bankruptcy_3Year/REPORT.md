# Polish_Bankruptcy_3Year — Early_Split_TDA / 2_PH_Tuned_Parameters

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) on the undersampled pool of that split |

CONSUMER. GridSearchCV on this arm's experiment-1 barcode matrices. Does not regenerate Ripser.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/2_PH_Tuned_Parameters/Polish_Bankruptcy_3Year/run.py
```
