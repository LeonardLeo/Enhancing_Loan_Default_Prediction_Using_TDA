# Statlog_German_Credit_Data — Late_Split_No_Undersample_H0 / 8_Null_Hypothesis_Algorithm2

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L30 / L60 |
| PCA rank | 15 |
| Number of snapshots | 500 |
| Points per snapshot | points per snapshot = floor(that class's count × snapshot size percent / 100) on the unbalanced pool |

CONSUMER. Robinson–Turner Algorithm 2 on this arm's barcode matrices.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Late_Split_No_Undersample_H0/8_Null_Hypothesis_Algorithm2/Statlog_German_Credit_Data/run.py
```

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
