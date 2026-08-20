# Snapshot sample size — results

Mirrors `5_Experiments/Snapshot_Sample_Size/` (dated 13/08/2026). English labels throughout. The symbol mapping used in the methods literature is in `docs/Notation.md`.

Item 3 is the sample-size study made of items 1, 2, and 4 — not a third grid.

| Path | Contents |
|------|----------|
| `shared/dataset_aware_grid.csv` | Surviving points-per-snapshot values per dataset × protocol |
| `shared/{Protocol}/{Dataset}/` | Design JSON, per-repeat metric CSVs |
| `{Experiment}/{Protocol}/{Dataset}/` | Experiment-sliced `repeat_metrics.csv` / `summary.csv` |
| `{Experiment}/Visualizations/` | Figures with methodology notes under every graph |

Human-facing CSV columns: `points_per_snapshot`, `n_snapshots`, `minority_class_count`, `majority_class_count`, `reuse_ratio`, `snapshot_size_percent_of_class`, `f1_mean`, `accuracy_mean`, `f1_ci95_low`, `f1_ci95_high`.
