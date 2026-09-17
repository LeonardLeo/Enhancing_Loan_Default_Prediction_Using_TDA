# Loan & Credit Default — Dataset Acquisition Log

The live study uses Default of Credit Card Client for TDA, Statistics, and Snapshot Sample Size. Statlog German Credit is retained for tabular Default Parameters and Archives. Retrieved 30 July 2026.

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
| 01 | Default of Credit Card Clients | `UCI_Credit_Card.csv` / processed Excel | 30,000 × 25 (23 predictors + ID + target) ✓ · 6,636 defaults (22.1%) |
| 02 | Statlog German Credit | `german.data` / `german.data-numeric` | Retained for `Default_Parameters/` and `Archives/` only |

---

## Contents

```
MANIFEST.csv                 sha256, size, primary source, mirror, licence
Default_Of_Credit_Card_Client_Data/  default of credit card clients.xls
Statlog_German_Credit_Data/          german.data, german.data-numeric, german.doc  (Default Parameters and Archives only)
```

Verify with `sha256sum -c` against `MANIFEST.csv`.

---

## Suggested next steps

1. Re-pull both tables from the UCI DOIs and diff against these checksums.
2. Only then start the target/leakage audit for publication freeze.
