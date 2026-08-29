# Experiment 1 — Ordinary machine learning on the original table (default settings)

## In one sentence

If we ignore topology and just train five off-the-shelf classifiers on the cleaned credit table, how well can they spot defaults?

## Who this is for

This is the **baseline**. Later TDA experiments are only interesting if they beat, or at least match, these numbers on a comparable task. Note the comparison is subtle: Exp 1 scores **customers**; Exp 3 scores **barcode snapshots**.

## Datasets

| Dataset | What it is | Typical size | Target |
|---------|------------|--------------|--------|
| DCCCD | UCI Taiwan credit-card default | ~30,000 clients | default next month (1) vs not (0) |
| Statlog German Credit | UCI German loan applications | 1,000 applicants | bad credit (1) vs good (0) |

## What we do (in order)

1. Load `1_Data/Processed_Datasets/{Folder}/processed_data.*`.
2. Exploratory data analysis.
3. Encode categories, log a few skewed amounts, fill holes.
4. Split 80/20, **stratified** so both sides keep the same default rate.
5. Drop weak columns with `SelectFpr`.
6. Oversample the minority class on the **training** side only (ADASYN).
7. Scale features 0–1 (`MinMaxScaler` fit on train / resampled data).
8. Train logistic regression, random forest, SVM, k-NN, XGBoost with library defaults.
9. Score them on the untouched test set.

Shared helpers live in `utils.py`.

This **is** the full load → preprocess → split → train flow. Experiments that only consume barcode CSVs should not copy this pipeline.

## What we look for

A model that catches defaults without labelling everyone as risky. Because defaults are rare, **F1 and recall** matter more than raw accuracy.

## Results

`6_Results/Default_Parameters/1_ML_Default_Parameters/{Folder}/model_results.pkl`

Each model has accuracy, precision, recall, F1, a classification report, and a confusion matrix.

## How to read a result without being a TDA expert

- **Accuracy** — share of people labelled correctly. Misleading when few people default.
- **Precision** — of those we flagged as default, how many really defaulted.
- **Recall** — of everyone who defaulted, how many we caught.
- **F1** — balance of precision and recall. This is the number we usually quote.

## Where to read the method

Open the named dataset script in each dataset folder (for example `Default_Of_Credit_Card_Client_Data/default_of_credit_cards_client_PH.py`). That file shows the pipeline in order, with comments at each stage. Open the named dataset script in the dataset folder.
