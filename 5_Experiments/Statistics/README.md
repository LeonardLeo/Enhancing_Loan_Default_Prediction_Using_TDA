# Statistics

Protocol-independent geometry. These numbers do not change when we move the train/test cut or turn undersampling on or off.

## 1_Intrinsic_Dimension_Estimation (old Exp 26)

Before-PCA ID is measured on the MinMax-scaled processed table. After-PCA ID uses the **same Exp 3 PCA rank** (7 / 15 / 10). Early-split PCA (train-only) is a sensitivity, not the Statistics headline.

Does not generate landmarks or barcodes.

## Where to read the method

Open `1_Intrinsic_Dimension_Estimation/<Dataset>/run_intrinsic_dimension.py`. That file shows load → scale → Two-NN before PCA → Exp-3 PCA → Two-NN after PCA → store.
