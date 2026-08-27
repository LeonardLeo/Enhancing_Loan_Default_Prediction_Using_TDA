# South_German_Credit — Early_Split_No_Undersample_H0 / 8_Null_Hypothesis_Algorithm2

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | no — full class pools inside each split |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L10 / L20 |
| PCA rank | 10 |
| Number of snapshots | 500 |
| Points per snapshot | points per snapshot = floor(class count × snapshot size percent / 100) on that split's available pool |

CONSUMER. Robinson–Turner Algorithm 2 on this arm's barcode matrices.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_No_Undersample_H0/8_Null_Hypothesis_Algorithm2/South_German_Credit/run.py
```

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
