# Snapshot sample size — results

Mirrors `5_Experiments/Snapshot_Sample_Size/` (dated 13/08/2026). English labels throughout. The symbol mapping used in the methods literature is in `docs/Notation.md`.

Item 3 is the sample-size study made of items 1, 2, and 4 — not a third grid. Items 1 and 2 are **different x-factors**, not the same sweep twice.

| Path | Contents |
|------|----------|
| `shared/dataset_aware_grid.csv` | Surviving points-per-snapshot values per dataset × protocol |
| `shared/{Protocol}/{Dataset}/` | Design JSON, per-repeat metric CSVs |
| `1_Snapshot_Count_Sweep/all_summary.csv` | Item 1: `n_snapshots` ∈ {15,30,45,60,90,120,180}; `points_per_snapshot` fixed at the default (`is_default_points_per_snapshot` always 1) |
| `2_Points_Per_Snapshot_Sweep/all_summary.csv` | Item 2: `points_per_snapshot` varies; `n_snapshots` always 180 |
| `3_Snapshot_Count_Across_Cloud_Sizes/all_summary.csv` | Item 4: full (`points_per_snapshot` × `n_snapshots`) family |
| `{Experiment}/{Protocol}/{Dataset}/` | Experiment-sliced `repeat_metrics.csv` / `summary.csv` |
| `{Experiment}/Visualizations/` | Combined overlays = mean trend (no error bars); `*_ci_panels.png` hold 95% ribbons; `*_reuse_*.png` and `*_reuse_heatmap.png` flag where reuse exceeds 1 |
| `{Experiment}/reuse_flags.csv` | Training-pool reuse by points per snapshot and number of snapshots; `reuse_exceeds_one` plus the largest still-safe value on each axis |

Human-facing CSV columns: `points_per_snapshot`, `n_snapshots`, `minority_class_count`, `majority_class_count`, `reuse_ratio`, `snapshot_size_percent_of_class`, `f1_mean`, `accuracy_mean`, `f1_ci95_low`, `f1_ci95_high`.

To see how barcodes were built, open `5_Experiments/Snapshot_Sample_Size/0_Shared_Pools`. The numbered experiment scripts only select which rows go on which figure.
