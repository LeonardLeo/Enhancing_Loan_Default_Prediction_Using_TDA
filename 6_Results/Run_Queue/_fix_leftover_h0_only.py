# Finish leftover live 3_H0_Only strings, then rewrite queues/docs that still use old buckets.
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LIVE = (
    "Early_Split_And_Undersample_H0",
    "Early_Split_And_Undersample_H0_And_H1",
    "Early_Split_No_Undersample_H0",
    "Early_Split_No_Undersample_H0_And_H1",
    "Late_Split_And_Undersample_H0",
    "Late_Split_And_Undersample_H0_And_H1",
    "Late_Split_No_Undersample_H0",
    "Late_Split_No_Undersample_H0_And_H1",
)
H0 = {s for s in LIVE if not s.endswith("_H0_And_H1")}
SKIP = {"Archives", "Snapshot_Sample_Size", "Default_Parameters", "Statistics", "__pycache__"}

DISPLAY = {
    "Early_Split_And_Undersample_H0": "Early split and undersample, using just H0",
    "Early_Split_And_Undersample_H0_And_H1": "Early split and undersample, using both H0 and H1",
    "Early_Split_No_Undersample_H0": "Early split, no undersample, using just H0",
    "Early_Split_No_Undersample_H0_And_H1": "Early split, no undersample, using both H0 and H1",
    "Late_Split_And_Undersample_H0": "Late split and undersample (the original historical run), using just H0",
    "Late_Split_And_Undersample_H0_And_H1": "Late split and undersample (the original historical run), using both H0 and H1",
    "Late_Split_No_Undersample_H0": "Late split, no undersample, using just H0",
    "Late_Split_No_Undersample_H0_And_H1": "Late split, no undersample, using both H0 and H1",
}

n = 0
for dirpath, dirnames, filenames in os.walk(ROOT / "5_Experiments"):
    parts = set(Path(dirpath).parts)
    if SKIP & parts:
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        continue
    slug = next((s for s in LIVE if s in Path(dirpath).parts), None)
    if slug is None:
        continue
    for name in filenames:
        if not name.endswith((".py", ".md", ".txt")):
            continue
        path = Path(dirpath) / name
        text = path.read_text(encoding="utf-8", errors="replace")
        updated = text.replace("3_H0_Only", "1_PH_Default_Parameters")
        if slug in H0 and name == "REPORT.md":
            updated = updated.replace(
                "1_Data/Landmark_Sets/" + slug + "/1_PH_Default_Parameters/{Dataset}/",
                "1_Data/Landmark_Sets/" + slug.replace("_H0", "_H0_And_H1") + "/1_PH_Default_Parameters/{Dataset}/  (sibling H0-and-H1 Ripser run)",
            )
            updated = updated.replace(
                "1_Data/Barcode_Statistics/" + slug + "/1_PH_Default_Parameters/{Dataset}/",
                "1_Data/Barcode_Statistics/" + slug.replace("_H0", "_H0_And_H1") + "/1_PH_Default_Parameters/{Dataset}/  (sibling H0-and-H1 Ripser run)",
            )
        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="\n")
            n += 1
            print("fixed", path.relative_to(ROOT).as_posix())
print(f"fixed {n} leftover live files")
