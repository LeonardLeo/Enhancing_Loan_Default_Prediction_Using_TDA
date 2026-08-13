# Default_Of_Credit_Card_Client_Data — Early_Split_TDA_And_No_Undersampling / 1_PH_Default_Parameters

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | no — full class pools inside each split |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L5 / L15 |
| PCA rank | 7 |
| `l` | 500 |
| `t` | t = floor(n_class * L / 100) per class on that split's available pool |

This experiment BUILDS landmarks and Ripser barcodes. Downstream 2–5, 7–8 consume its `data_L*.csv`.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA_And_No_Undersampling/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/run.py
```
