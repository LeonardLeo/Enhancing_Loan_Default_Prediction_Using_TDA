# Polish_Bankruptcy_3Year — No_Undersampling / 6_Sampling_Ratio_Audit

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| Number of snapshots | 500 |
| Points per snapshot | points per snapshot = floor(that class's count × snapshot size percent / 100) on the unbalanced pool |

No Ripser. Scores reuse ratio = (points per snapshot × number of snapshots) / class count from the protocol's class pools, snapshot-size percents, and 500 snapshots.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/6_Sampling_Ratio_Audit/Polish_Bankruptcy_3Year/run.py
```

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
