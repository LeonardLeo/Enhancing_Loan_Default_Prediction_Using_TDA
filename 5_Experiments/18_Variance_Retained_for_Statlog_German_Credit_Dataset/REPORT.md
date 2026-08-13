# Experiment 18 — PCA component sweep (named after Statlog)

## In one sentence

Same idea as Experiment 16, originally for Statlog’s 15-component choice. Newer tables get the same sweep so the six datasets stay comparable.

## Who this is for

If F1 peaks at 8 components and we used 15, we were keeping noise. If F1 is still climbing at 15, we were starving the pipeline.

This rebuilds landmarks per component count.

## Why Statlog needed 15 and the new tables used 10

Statlog’s 15 is what it takes on *that* table to sit near 90% variance. The new tables were **not** given 15: they share 10 so Ripser spaces match each other (Taiwan ~88%, others short). This sweep is how we check whether that shared-10 was kind or cruel on each table. Experiment 13 is the sibling that locks variance and lets rank float.

`docs/Design_Decisions.md` §2.

## Results

`6_Results/18_Variance_Retained_for_Statlog_German_Credit_Dataset/{Folder}/`
