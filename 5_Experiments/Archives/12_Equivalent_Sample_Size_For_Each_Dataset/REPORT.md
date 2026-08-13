# Experiment 12 — Fair landmark *counts* across datasets

## In one sentence

Statlog L30 draws 90 points; DCCCD L5 draws 331. If TDA “wins” only when subsets are huge, the win is a sample-size artefact.

## Who this is for

Percentages are not comparable when class sizes differ. This experiment reruns the richer tables at percents that match Statlog’s **absolute** subset size `t`.

## Datasets

DCCCD plus the newer tables (PKDD, Polish, Taiwan, South German). Statlog is the reference `t`.

## What we look for

Does DCCCD / Polish TDA F1 drop when we starve each snapshot down to Statlog’s 90–180 points? If yes, quote sample size, not “TDA works better on credit cards”.

## Results

`6_Results/Archives/12_Equivalent_Sample_Size_For_Each_Dataset/{Folder}/`
