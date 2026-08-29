# Experiment 15 — Which k should k-NN use on barcode rows?

## In one sentence

Library-default k-NN uses k=5. We sweep k on the Experiment 3 barcode table and pick the elbow / best validation k.

## Who this is for

k-NN labels a new snapshot by asking its k nearest training snapshots. Too small k = jumpy; too large k = it just votes the majority class. The elbow plot is the teaching picture.

## Datasets

All six. **Prerequisite:** Experiment 3 `data_L*.csv`. No landmark rebuild.


- **L10:** best k = 16, test accuracy **0.46**, F1 **0.36**. Worse than a coin flip on recall for defaults. Small landmarks on 76 people are not a stable neighbourhood.
- **L20:** best k = 11, test accuracy **0.59**, F1 **0.55**. Still modest, but the extra points per snapshot helped.


## Results

`6_Results/Archives/15_Working_With_K_in_KNN/{Folder}/data_L{percent}/elbow_curve.png` plus `results.json`.
