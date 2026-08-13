# Default_Of_Credit_Card_Client_Data — Early_Split_TDA / 6_Sampling_Ratio_Audit

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L5 / L15 |
| PCA rank | 7 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) on the undersampled pool of that split |

No Ripser. Scores R = (t * l) / n_class from the protocol's class pools, L percents, and l = 500.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/6_Sampling_Ratio_Audit/Default_Of_Credit_Card_Client_Data/run.py
```
