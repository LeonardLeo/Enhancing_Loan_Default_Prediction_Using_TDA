# Default_Of_Credit_Card_Client_Data — No_Undersampling / 1_PH_Default_Parameters

| Knob | Value |
|------|-------|
| Split timing | late (80/20 on barcode rows after Ripser) |
| Undersample | no — full class pools after full-table PCA |
| PCA | MinMax + PCA on the FULL processed table (same ranks as Exp 3) |
| L percents | L5 / L15 |
| PCA rank | 7 |
| Number of snapshots | 500 |
| Points per snapshot | points per snapshot = floor(that class's count × snapshot size percent / 100) on the unbalanced pool |

This experiment BUILDS landmarks and Ripser barcodes. Downstream 2–5, 7–8 consume its `data_L*.csv`.

Run:

```
.\tda_env\Scripts\python.exe 5_Experiments/No_Undersampling/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/run.py
```

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. `run.py` is an optional convenience launcher and is not the method document.
