# Snapshot notation

This glossary maps the compact symbols used in the snapshot-size methods literature onto the English names used in this study. In the snapshot-size literature these quantities are sometimes written t and l. Everywhere else a reader sees — README files, experiment reports, figure labels, CSV headers, captions — the English names are used.

| Symbol in the methods literature | English name used in this study | Definition |
|----------------------------------|---------------------------------|------------|
| `t` | **points per snapshot** | How many customers are drawn into one snapshot (one Vietoris–Rips cloud). Absolute count, not a percent. |
| `l` | **number of snapshots** | How many snapshots are drawn per class. |
| `L` / landmark percent | **snapshot size as a percent of the class** | Relative size: points per snapshot = floor(minority class count × percent / 100) in the historical pipeline. L5 means 5%, L15 means 15%. |
| `n1` | **minority class count** | Number of rows in the smaller class after that protocol’s split and optional undersample. |
| `n2` | **majority class count** | Number of rows in the larger class on the same pool. After undersampling, both classes have the minority count. |
| `R = (t · l) / n1` | **reuse ratio** = (points per snapshot × number of snapshots) / minority class count | Expected times a typical minority customer appears across the snapshot collection. Target is near or below 1. |
| `l_train` / `l_test` | **training snapshot count** / **test snapshot count** | How many snapshots are used to train vs evaluate. Defaults in the revised protocol: 60 / 15. |
| `b` | **intrinsic dimension** | Geometry of the cloud (Two-NN is the primary estimator). Not the PCA rank. |

Historical scripts and archived folders may still use `t` / `l` in Python identifiers and old comments. That is internal. Reader-facing strings use the English names.

The dated sample-size study (`5_Experiments/Snapshot_Sample_Size/`, 13/08/2026) uses the English names as code identifiers too: `points_per_snapshot`, `n_snapshots`, `minority_count`. Folder map: `docs/Repository_Layout.md`.
