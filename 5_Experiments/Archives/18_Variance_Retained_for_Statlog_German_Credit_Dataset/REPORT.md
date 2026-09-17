# Experiment 18 — PCA component sweep (named after Statlog)

## In one sentence

Same idea as Experiment 16, originally for Statlog’s 15-component choice.

## Who this is for

If F1 peaks at 8 components and we used 15, we were keeping noise. If F1 is still climbing at 15, we were starving the pipeline.

This rebuilds landmarks per component count.

## Why Statlog needed 15

Statlog’s 15 is what it takes on *that* table to sit near 90% variance. DCCCD already overshoots 90% at 7. This sweep is how we check whether 15 was kind or cruel on Statlog. Experiment 13 is the sibling that locks variance and lets rank float.

`docs/Design_Decisions.md` §2.

## Results

`6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/{Folder}/`
