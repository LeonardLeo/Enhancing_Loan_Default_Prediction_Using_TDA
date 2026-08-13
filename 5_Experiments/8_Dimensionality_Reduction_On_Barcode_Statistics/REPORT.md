# Experiment 8 — Can we see the two classes after shrinking 24 barcode columns?

## In one sentence

PCA / other reductions on the 24 barcode statistics: do defaults and non-defaults separate in 2D, or sit on top of each other?

## Who this is for

Twenty-four numbers are hard to stare at. This experiment makes a scatter plot. Overlap in the plot does not prove the classes are identical (a later classifier might still find a wrinkle), but **complete overlap** is a warning that topology is not giving a cheap visual signal.

## Datasets

All six. Loads Experiment 3 combined and per-class `data_L*.csv` / barcode frames. **Does not** regenerate landmarks.

## What we do

For each available matrix, run the shared dimensionality-reduction helper and write figures under `6_Results/8_Dimensionality_Reduction_On_Barcode_Statistics/{Folder}/`.

## Results

Look at the PNG files: two colours should be the two classes. If you cannot tell them apart by eye, quote Experiment 17 (more methods, density overlay) and Experiment 27 (a formal same-process test) rather than claiming “clear separation”.
