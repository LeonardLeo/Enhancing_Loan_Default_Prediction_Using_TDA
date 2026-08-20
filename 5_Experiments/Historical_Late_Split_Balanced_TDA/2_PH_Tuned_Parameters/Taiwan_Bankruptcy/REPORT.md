# Taiwan_Bankruptcy — Historical_Late_Split_Balanced_TDA / 2_PH_Tuned_Parameters

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | yes — majority downsampled to minority count before landmarks |
| PCA | MinMax + PCA on the FULL processed table (historical / slightly leaky) |
| L percents | L10 / L20 |
| PCA rank | 10 |
| Number of snapshots | 500 |
| Points per snapshot | points per snapshot = floor(minority class count × snapshot size percent / 100); after undersampling both classes have the minority count |

CONSUMER. GridSearchCV on this arm's experiment-1 barcode matrices. Does not regenerate Ripser.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/Historical_Late_Split_Balanced_TDA/2_PH_Tuned_Parameters/Taiwan_Bankruptcy/run.py
```

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH_tuned.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
