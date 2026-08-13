# Default_Of_Credit_Card_Client_Data — Historical_Late_Split_Balanced_TDA / 6_Sampling_Ratio_Audit

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| L percents | L5 / L15 |
| PCA rank | 7 |
| `l` | 500 |
| `t` | t = floor(n1 * L / 100) with n1 = n2 = minority count |

No Ripser. Scores R = (t * l) / n_class from the protocol's class pools, L percents, and l = 500.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/6_Sampling_Ratio_Audit/Default_Of_Credit_Card_Client_Data/run.py
```
