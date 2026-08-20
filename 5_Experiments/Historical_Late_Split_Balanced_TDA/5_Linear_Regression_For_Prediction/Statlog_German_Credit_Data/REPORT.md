# Statlog_German_Credit_Data — Historical_Late_Split_Balanced_TDA / 5_Linear_Regression_For_Prediction

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| L percents | L30 / L60 |
| PCA rank | 15 |
| Number of snapshots | 500 |
| Points per snapshot | points per snapshot = floor(minority class count × snapshot size percent / 100); after undersampling both classes have the minority count |

CONSUMER. Linear regression on the H0 slice of experiment 1.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/5_Linear_Regression_For_Prediction/Statlog_German_Credit_Data/run.py
```

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
