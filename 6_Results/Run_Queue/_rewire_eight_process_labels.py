# Rewrite live eight-process scripts, docs, and queues onto the new names.
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

H0H1_OF = {
    "Early_Split_And_Undersample_H0": "Early_Split_And_Undersample_H0_And_H1",
    "Early_Split_No_Undersample_H0": "Early_Split_No_Undersample_H0_And_H1",
    "Late_Split_And_Undersample_H0": "Late_Split_And_Undersample_H0_And_H1",
    "Late_Split_No_Undersample_H0": "Late_Split_No_Undersample_H0_And_H1",
}
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
OLD_SLUGS = (
    "Early_Split_TDA_And_No_Undersampling",
    "Historical_Late_Split_Balanced_TDA",
    "Early_Split_TDA",
    "No_Undersampling",
)
LIVE_SLUGS = tuple(DISPLAY.keys())
SKIP_DIR_PARTS = {
    "Archives",
    "Snapshot_Sample_Size",
    "Default_Parameters",
    "Statistics",
    "__pycache__",
}
H0_ONLY_EXPERIMENT_RE = re.compile(r'EXPERIMENT = "3_H0_Only"')
H0_ONLY_KW_RE = re.compile(r"experiment=(['\"])3_H0_Only\1")


def slug_from_path(path: Path) -> str | None:
    for slug in LIVE_SLUGS:
        if slug in path.parts:
            return slug
    return None


def replace_old_slugs(text: str, slug: str) -> str:
    for old in OLD_SLUGS:
        text = text.replace(old, slug)
    return text


def replace_old_titles(text: str, display: str) -> str:
    text = text.replace("Early Split TDA And No Undersampling", display)
    text = text.replace("Historical Late Split, Balanced TDA", display)
    text = text.replace("Early Split TDA", display)
    text = re.sub(r"\bNo Undersampling\b", display, text)
    return text


def ensure_barcode_source_import(text: str) -> str:
    if "barcode_source_bucket" in text:
        return text
    return text.replace(
        "from utils import (",
        "from utils import (\n    barcode_source_bucket,",
        1,
    )


def rewrite_h0_default_script(text: str, slug: str) -> str:
    text = ensure_barcode_source_import(text)
    text = H0_ONLY_EXPERIMENT_RE.sub('EXPERIMENT = "1_PH_Default_Parameters"', text)
    if "SOURCE_BUCKET" not in text:
        text = text.replace(
            f'PROTOCOL_BUCKET = "{slug}"\nEXPERIMENT = "1_PH_Default_Parameters"\nSOURCE_EXPERIMENT = "1_PH_Default_Parameters"',
            (
                f'PROTOCOL_BUCKET = "{slug}"\n'
                f"SOURCE_BUCKET = barcode_source_bucket(PROTOCOL_BUCKET)\n"
                f'EXPERIMENT = "1_PH_Default_Parameters"\n'
                f'SOURCE_EXPERIMENT = "1_PH_Default_Parameters"'
            ),
            1,
        )
    text = text.replace(
        'os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)',
        'os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", SOURCE_BUCKET, SOURCE_EXPERIMENT, FOLDER)',
    )
    return text


def rewrite_text(text: str, path: Path, slug: str) -> str:
    display = DISPLAY[slug]
    text = replace_old_slugs(text, slug)
    text = replace_old_titles(text, display)
    text = H0_ONLY_KW_RE.sub(r"experiment=\g<1>1_PH_Default_Parameters\g<1>", text)
    text = text.replace("/ 3_H0_Only", "/ 1_PH_Default_Parameters")
    text = text.replace("\\3_H0_Only", "\\1_PH_Default_Parameters")
    text = text.replace("/3_H0_Only", "/1_PH_Default_Parameters")

    parts = set(path.parts)
    is_h0 = slug in H0H1_OF
    is_default_ph = "1_PH_Default_Parameters" in parts
    is_alg2 = "8_Null_Hypothesis_Algorithm2" in parts
    is_dataset_script = path.suffix == ".py" and path.name not in {
        "visualize_results.py",
        "run.py",
        "run_all.py",
    }

    if is_h0 and is_default_ph and is_dataset_script:
        text = rewrite_h0_default_script(text, slug)
    elif is_h0 and is_default_ph:
        text = H0_ONLY_EXPERIMENT_RE.sub('EXPERIMENT = "1_PH_Default_Parameters"', text)
    elif is_h0 and is_alg2:
        text = H0_ONLY_EXPERIMENT_RE.sub('EXPERIMENT = "8_Null_Hypothesis_Algorithm2"', text)
    else:
        text = H0_ONLY_EXPERIMENT_RE.sub('EXPERIMENT = "1_PH_Default_Parameters"', text)
    return text


def rewrite_tree(root: Path) -> int:
    changed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        parts = set(Path(dirpath).parts)
        if SKIP_DIR_PARTS & parts:
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_PARTS]
            continue
        slug = slug_from_path(Path(dirpath))
        if slug is None:
            continue
        for name in filenames:
            if not name.endswith((".py", ".md", ".txt")):
                continue
            path = Path(dirpath) / name
            original = path.read_text(encoding="utf-8", errors="replace")
            updated = rewrite_text(original, path, slug)
            if updated != original:
                path.write_text(updated, encoding="utf-8", newline="\n")
                changed += 1
                print("updated", path.relative_to(ROOT).as_posix())
    return changed


if __name__ == "__main__":
    n = rewrite_tree(ROOT / "5_Experiments")
    print(f"rewrote {n} experiment files")
