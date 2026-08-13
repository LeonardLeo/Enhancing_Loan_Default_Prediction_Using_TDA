# Polish_Bankruptcy_3Year — Historical_Late_Split_Balanced_TDA / 1_PH_Default_Parameters

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| `l` | 500 |
| `t` | t = floor(n1 * L / 100) with n1 = n2 = minority count |

This experiment BUILDS landmarks and Ripser barcodes. Downstream 2–5, 7–8 consume its `data_L*.csv`.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Polish_Bankruptcy_3Year/run.py
```
