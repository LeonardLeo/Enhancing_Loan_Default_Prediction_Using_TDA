# Loan & Credit Default — Dataset Acquisition Log

Retrieved 30 July 2026 against the portfolio in `Loan_Default_Dataset_Portfolio.docx`
("Enhancing Loan Default Prediction Using Topological Data Analysis", team briefing 20 July 2026).

---

## ⚠ Read this before using any of these files

**All four primary hosts named in the briefing were unreachable from this environment.**
`archive.ics.uci.edu`, `relational.fel.cvut.cz`, `data.mendeley.com` and
`catalog.data.gov` all return `403 host_not_allowed` from the egress proxy.

Everything below therefore came from **third-party GitHub mirrors**, not from the
primary records. That matters directly for your own gates:

- **Gate "Authenticity"** — satisfied only by inference. The files match the published
  row counts, column counts and class balances (verified below), but no mirror ships
  the originating institution's release metadata.
- **Gate "Rights"** — the licence text was *not* obtained. Licences recorded in
  `MANIFEST.csv` are copied from the briefing, not from a downloaded LICENCE file.
- **Gate "Metadata"** — only German Credit (`german.doc`) and South German Credit
  (`codetable.txt`) came with their data dictionaries. The rest did not.

**Recommendation:** treat this set as a working copy for pipeline development, and
re-download from the primary DOIs before anything goes into the paper. Your own
"non-negotiable controls" section requires frozen versions, checksums and extraction
dates traceable to the source — mirrors cannot supply that.

---

## What was retrieved and verified

| # | Dataset | Files | Verification against briefing |
|---|---------|-------|-------------------------------|
| 01 | Statlog German Credit | `german.data`, `german.data-numeric`, `german.doc` | 1,000 rows; 700 good / 300 bad ✓ · numeric form 24 inputs + target ✓ |
| 02 | Default of Credit Card Clients | `UCI_Credit_Card.csv` | 30,000 × 25 (23 predictors + ID + target) ✓ · 6,636 defaults (22.1%) |
| 03 | PKDD'99 Czech Financial (Berka) | 8 `.asc` tables | All eight record counts exact ✓ · loans 606 successful / 76 unsuccessful ✓ |
| 07 | Polish Companies Bankruptcy | `1year.arff`–`5year.arff` | 5,910–10,503 per file ✓ · 64 ratios + class ✓ · 3-year file 495 / 10,008 ✓ |
| 08 | Taiwanese Bankruptcy Prediction | `data.csv` | 6,819 × 96 (95 predictors + target) ✓ · 220 bankrupt |
| 09 | South German Credit | `SouthGermanCredit.asc`, `codetable.txt`, `read_SouthGermanCredit.R` | 1,000 × 21 ✓ · 700 / 300 ✓ |

PKDD table counts: account 4,500 · card 892 · client 5,369 · disp 5,369 ·
district 77 · loan 682 · order 6,471 · trans 1,056,320 — all exact.

The `.asc` files are semicolon-delimited (`sep=";"`).

---

## What is missing

| # | Dataset | Why | Route |
|---|---------|-----|-------|
| 04 | Ziemba, loan repayment & borrowers (91,759 × 273) | Mendeley blocked; no mirror exists | `fetch_blocked_datasets.sh` · DOI 10.17632/fr99jcnkxg.1 |
| 05 | US P2P lending + state features (2,703,430 rows) | Mendeley blocked; too large for GitHub mirrors | `fetch_blocked_datasets.sh` · DOI 10.17632/wb3ndt69gf |
| 06 | SBA 7(a) & 504 FOIA reports | data.gov blocked | `fetch_blocked_datasets.sh` · data.sba.gov |

### `_unverified/SBAnational_PARTIAL_22427rows.csv`

Quarantined deliberately. This is **not** the SBA dataset your briefing specifies.
It is the derived "SBA National" teaching extract, and this copy is partial:
22,427 rows covering FY1972–2006, against ~899k rows in the full derived file and a
different scope again from the official 7(a)/504 FOIA releases. Schema is right
(`MIS_Status`, `ChgOffPrinGr`, `GrAppv`, `SBA_Appv`), outcome split 17,615 paid-in-full /
4,693 charged off / 119 null.

It fails your Authenticity and Metadata gates on exactly the grounds the briefing uses
to reject the Kaggle "Loan Default Dataset". Use it for schema scaffolding only.

---

## Contents

```
MANIFEST.csv                 sha256, size, primary source, mirror, licence — 22 files
fetch_blocked_datasets.sh    retrieval script for 04, 05, 06
01_german_credit/            3 files
02_credit_card_default/      1 file
03_pkdd_czech/               8 files  (trans.asc is 69 MB)
07_polish_bankruptcy/        5 files
08_taiwan_bankruptcy/        1 file
09_south_german_credit/      3 files
_unverified/                 1 file — do not use as a core dataset
```

Total 109.7 MB. Verify with `sha256sum -c` against `MANIFEST.csv`.

---

## Suggested next steps

1. Re-pull 01, 02, 07, 08, 09 from the UCI DOIs and diff against these checksums. A clean
   diff converts every mirror above into a provenance-clean copy at near-zero cost.
2. Pull 03 from the CTU relational repository and record its current reuse notice — the
   briefing flags this licence as unresolved.
3. Run `fetch_blocked_datasets.sh` for 04, 05, 06, then freeze the SBA snapshot date.
4. Only then start the target/leakage audit. Several files here have no data dictionary,
   and the briefing makes feature-availability mapping a gate rather than a later step.
