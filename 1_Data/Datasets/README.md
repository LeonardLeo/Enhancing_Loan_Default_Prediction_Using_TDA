# Loan & Credit Default — Dataset Acquisition Log

The live study uses two public UCI tables. Retrieved 30 July 2026.

---

## ⚠ Read this before using any of these files

`archive.ics.uci.edu` returned `403 host_not_allowed` from the egress proxy used at retrieval time. The copies below therefore came from **third-party GitHub mirrors**, not from the primary records.

- **Gate "Authenticity"** — satisfied only by inference. The files match the published row counts, column counts and class balances (verified below), but no mirror ships the originating institution's release metadata.
- **Gate "Rights"** — the licence text was *not* obtained. Licences recorded in `MANIFEST.csv` are copied from the briefing, not from a downloaded LICENCE file.
- **Gate "Metadata"** — German Credit (`german.doc`) came with its data dictionary.

**Recommendation:** treat this set as a working copy for pipeline development, and re-download from the primary DOIs before anything goes into the paper.

---

## What was retrieved and verified

| # | Dataset | Files | Verification |
|---|---------|-------|--------------|
| 01 | Statlog German Credit | `german.data`, `german.data-numeric`, `german.doc` | 1,000 rows; 700 good / 300 bad ✓ · numeric form 24 inputs + target ✓ |
| 02 | Default of Credit Card Clients | `UCI_Credit_Card.csv` / processed Excel | 30,000 × 25 (23 predictors + ID + target) ✓ · 6,636 defaults (22.1%) |

---

## Contents

```
MANIFEST.csv                 sha256, size, primary source, mirror, licence
Statlog_German_Credit_Data/  german.data, german.data-numeric, german.doc
Default_Of_Credit_Card_Client_Data/  default of credit card clients.xls
```

Verify with `sha256sum -c` against `MANIFEST.csv`.

---

## Suggested next steps

1. Re-pull both tables from the UCI DOIs and diff against these checksums.
2. Only then start the target/leakage audit for publication freeze.
