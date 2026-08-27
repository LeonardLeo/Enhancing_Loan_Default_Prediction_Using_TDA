# One-shot filesystem migration: four protocol arms → eight named processes.
# Uses Windows long-path prefixes. Safe to re-run: skips missing sources.
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import win_long_path

OLD_TO_H0H1 = {
    "Historical_Late_Split_Balanced_TDA": "Late_Split_And_Undersample_H0_And_H1",
    "Early_Split_TDA": "Early_Split_And_Undersample_H0_And_H1",
    "No_Undersampling": "Late_Split_No_Undersample_H0_And_H1",
    "Early_Split_TDA_And_No_Undersampling": "Early_Split_No_Undersample_H0_And_H1",
}
OLD_TO_H0 = {
    "Historical_Late_Split_Balanced_TDA": "Late_Split_And_Undersample_H0",
    "Early_Split_TDA": "Early_Split_And_Undersample_H0",
    "No_Undersampling": "Late_Split_No_Undersample_H0",
    "Early_Split_TDA_And_No_Undersampling": "Early_Split_No_Undersample_H0",
}

ARCHIVE_EXPS = (
    "4_Dropping_Correlated_Barcode_Statistics_Columns",
    "5_Linear_Regression_For_Prediction",
    "7_Snapshot_Mean_Variance",
)
H0_NESTED = "3_H0_Only"

ARCHIVE_SHELF = "Archives/Four_Arm_Nested_Experiments"

DATA_KINDS = ("Landmark_Sets", "Barcode_Statistics", "TDA_Datasets")


def p(path: Path) -> str:
    return str(win_long_path(path))


def exists(path: Path) -> bool:
    return os.path.exists(p(path))


def move_tree(src: Path, dest: Path) -> None:
    if not exists(src):
        print("  skip missing", src)
        return
    dest_parent = dest.parent
    os.makedirs(p(dest_parent), exist_ok=True)
    if exists(dest):
        print("  skip exists", dest)
        return
    print("  MOVE", src, "->", dest)
    shutil.move(p(src), p(dest))


def copy_scripts_only(src: Path, dest: Path) -> None:
    if not exists(src):
        return
    os.makedirs(p(dest), exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(p(src)):
        rel = os.path.relpath(dirpath, p(src))
        out_dir = dest if rel == "." else dest / rel
        os.makedirs(p(out_dir), exist_ok=True)
        for name in filenames:
            if not name.endswith((".py", ".md", ".txt")):
                continue
            shutil.copy2(os.path.join(dirpath, name), p(out_dir / name))


def rename_bucket(root: Path, old: str, new: str) -> None:
    src = root / old
    dest = root / new
    if exists(dest) and not exists(src):
        return
    move_tree(src, dest)


print("=== 1. Archive nested experiments 4, 5, 7 ===")
for old in OLD_TO_H0H1:
    for exp in ARCHIVE_EXPS:
        move_tree(
            ROOT / "5_Experiments" / old / exp,
            ROOT / "5_Experiments" / ARCHIVE_SHELF / old / exp,
        )
        move_tree(
            ROOT / "6_Results" / old / exp,
            ROOT / "6_Results" / ARCHIVE_SHELF / old / exp,
        )
        move_tree(
            ROOT / "1_Data" / "TDA_Datasets" / old / exp,
            ROOT / "1_Data" / "TDA_Datasets" / ARCHIVE_SHELF / old / exp,
        )

print("=== 2. Snapshot nested 3_H0_Only scripts into Archives ===")
for old in OLD_TO_H0H1:
    copy_scripts_only(
        ROOT / "5_Experiments" / old / H0_NESTED,
        ROOT / "5_Experiments" / ARCHIVE_SHELF / old / H0_NESTED,
    )

print("=== 3. Rename four live buckets to H0_And_H1 slugs ===")
for old, new in OLD_TO_H0H1.items():
    rename_bucket(ROOT / "5_Experiments", old, new)
    rename_bucket(ROOT / "6_Results", old, new)
    for kind in DATA_KINDS:
        rename_bucket(ROOT / "1_Data" / kind, old, new)

print("=== 4. Promote 3_H0_Only into sibling *_H0 / 1_PH_Default_Parameters ===")
for old, h0h1 in OLD_TO_H0H1.items():
    h0 = OLD_TO_H0[old]
    move_tree(
        ROOT / "5_Experiments" / h0h1 / H0_NESTED,
        ROOT / "5_Experiments" / h0 / "1_PH_Default_Parameters",
    )
    move_tree(
        ROOT / "6_Results" / h0h1 / H0_NESTED,
        ROOT / "6_Results" / h0 / "1_PH_Default_Parameters",
    )
    move_tree(
        ROOT / "1_Data" / "TDA_Datasets" / h0h1 / H0_NESTED,
        ROOT / "1_Data" / "TDA_Datasets" / h0 / "1_PH_Default_Parameters",
    )

print("=== 5. Copy Algorithm 2 scripts into H0 process folders ===")
for old, h0h1 in OLD_TO_H0H1.items():
    h0 = OLD_TO_H0[old]
    src = ROOT / "5_Experiments" / h0h1 / "8_Null_Hypothesis_Algorithm2"
    dest = ROOT / "5_Experiments" / h0 / "8_Null_Hypothesis_Algorithm2"
    if exists(src) and not exists(dest):
        print("  COPYTREE", src, "->", dest)
        shutil.copytree(p(src), p(dest))

print("DONE filesystem migration")
