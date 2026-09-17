# Statlog_German_Credit_Data — No_Undersampling / 3_H0_Only

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L30 / L60 |
| PCA rank | 15 |
| Number of snapshots | 500 |
| Points per snapshot | points per snapshot = floor(that class's count × snapshot size percent / 100) on the unbalanced pool |

CONSUMER. Keeps H0 (`*_0`) columns from experiment 1 and retrains default classifiers. Does not regenerate Ripser.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/3_H0_Only/Statlog_German_Credit_Data/run.py
```

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_H0_only.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
