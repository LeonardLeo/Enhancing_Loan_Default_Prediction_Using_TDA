# PKDD_Czech_Financial — Early_Split_TDA / 3_H0_Only

| Knob | Value |
|------|-------|
| Split timing | early — stratified 80/20 on customers BEFORE scaler/PCA/landmarks |
| Undersample | yes — independently inside the train pool and the test pool |
| PCA | MinMax + PCA fit on TRAIN only; test is transformed |
| L percents | L10 / L20 |
| PCA rank | 10 |
| Number of snapshots | 500 |
| Points per snapshot | points per snapshot = floor(class count × snapshot size percent / 100) on the undersampled pool of that split |

CONSUMER. Keeps H0 (`*_0`) columns from experiment 1 and retrains default classifiers. Does not regenerate Ripser.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Early_Split_TDA/3_H0_Only/PKDD_Czech_Financial/run.py
```

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_H0_only.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
