# Experiment 16 — PCA component sweep (named after the credit-card table)

## In one sentence

Rebuild the TDA pipeline for 2, 3, 5, … components and plot metrics vs components kept. Is 7 (DCCCD) a sweet spot or just a round number?

## Who this is for

Folder name is historical (DCCCD). Newer tables get the same sweep so the six datasets stay comparable. Experiment 18 is the Statlog-named twin.

This **does** rebuild landmarks per component count — expensive. Do not confuse it with Experiment 26, which estimates dimension **without** Ripser.

## Why this is not the same as the 90% rule

Exp 3’s 7 / 10 / 15 ranks are **design choices** (paper pair vs shared-10 on new tables; see `docs/Design_Decisions.md`). A sweep asks a different question: **does F1 actually care?** If F1 is flat from 6 to 12 components, quoting 90% vs 78% is a methods nicety, not a result. If F1 climbs until we pass 90%, the PKDD/South German misses in Exp 3 are load-bearing.

## Datasets

DCCCD originally; PKDD / Polish / Taiwan / South German / Statlog have matching scripts under this numbered folder.

## Results

`6_Results/Archives/16_Variance_Retained_for_Default_of_Credit_Card_Client_Dataset/{Folder}/`
