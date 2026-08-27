# Compile a downloadable PDF of live results for Statlog and Default of Credit Card Client.
from __future__ import annotations

import os
import re
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from utils import TDA_PROCESS_REGISTRY, process_display_name, win_long_path

OUT_DIR = ROOT / "6_Results" / "Compiled_Reports"
OUT_PDF = OUT_DIR / "Statlog_And_Default_Of_Credit_Card_Client_Results.pdf"

DATASETS = (
    ("Statlog_German_Credit_Data", "Statlog German Credit"),
    ("Default_Of_Credit_Card_Client_Data", "Default of Credit Card Client"),
)
MODELS = ("svm", "knn", "xgb", "logistic", "random_forest")
MODEL_LABEL = {
    "svm": "SVM",
    "knn": "KNN",
    "xgb": "XGBoost",
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
}
NAVY = colors.HexColor("#1B365D")
STEEL = colors.HexColor("#4A6FA5")
ROW_ALT = colors.HexColor("#F3F6FB")
BEST = colors.HexColor("#E8F5E9")
WARN = colors.HexColor("#FFF3E0")
MISS = colors.HexColor("#FFEBEE")


def openable(path: Path) -> str:
    return str(win_long_path(path))


def exists(path: Path) -> bool:
    return os.path.isfile(openable(path))


def load_pkl(path: Path):
    return joblib.load(openable(path))


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def snapshot_label(key: str) -> str:
    text = str(key).replace("\\", "/")
    match = re.search(r"(L_?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return Path(text).stem
    token = match.group(1).upper().replace("L_", "L")
    if not token.startswith("L"):
        token = "L" + token
    return token


def metric_tuple(stats: dict) -> dict:
    return {
        "accuracy": stats.get("accuracy"),
        "precision": stats.get("precision"),
        "recall": stats.get("recall"),
        "f1": stats.get("f1_score", stats.get("f1")),
    }


def walk_model_results(obj) -> list[dict]:
    rows = []
    if not isinstance(obj, dict) or not obj:
        return rows
    sample = next(iter(obj.values()))
    if isinstance(sample, dict) and sample and any(k in sample for k in MODELS):
        if any(isinstance(v, dict) and ("accuracy" in v or "f1_score" in v) for v in sample.values()):
            for setting, models in obj.items():
                if not isinstance(models, dict):
                    continue
                for model, stats in models.items():
                    if not isinstance(stats, dict):
                        continue
                    if "accuracy" not in stats and "f1_score" not in stats:
                        continue
                    row = metric_tuple(stats)
                    row.update({"setting": snapshot_label(setting), "model": model})
                    rows.append(row)
            return rows
    if any(isinstance(v, dict) and ("accuracy" in v or "f1_score" in v) for v in obj.values()):
        for model, stats in obj.items():
            if not isinstance(stats, dict):
                continue
            row = metric_tuple(stats)
            row.update({"setting": "Original features", "model": model})
            rows.append(row)
        return rows
    for setting, inner in obj.items():
        rows.extend(walk_model_results(inner) if isinstance(inner, dict) else [])
        for row in rows:
            if row.get("setting") in {None, "", "Original features"}:
                row["setting"] = snapshot_label(setting)
    return rows


def walk_cv(obj) -> list[dict]:
    rows = []
    if not isinstance(obj, dict) or not obj:
        return rows
    sample = next(iter(obj.values()))
    if isinstance(sample, dict) and ("mean_accuracy" in sample or "mean_accracy" in sample or "cross_val_scores" in sample):
        for model, stats in obj.items():
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean_accuracy", stats.get("mean_accracy"))
            std = stats.get("std_accuracy")
            rows.append({"setting": "5-fold CV", "model": model, "mean": mean, "std": std})
        return rows
    if isinstance(sample, dict) and sample and any(k in sample for k in MODELS):
        for setting, models in obj.items():
            if not isinstance(models, dict):
                continue
            for model, stats in models.items():
                if not isinstance(stats, dict):
                    continue
                mean = stats.get("mean_accuracy", stats.get("mean_accracy"))
                std = stats.get("std_accuracy")
                if mean is None:
                    continue
                rows.append({"setting": snapshot_label(setting), "model": model, "mean": mean, "std": std})
        return rows
    for setting, inner in obj.items():
        nested = walk_cv(inner) if isinstance(inner, dict) else []
        for row in nested:
            if row.get("setting") == "5-fold CV":
                row["setting"] = snapshot_label(setting)
            rows.append(row)
    return rows


def result_dir(bucket: str, experiment: str, dataset: str) -> Path:
    return ROOT / "6_Results" / bucket / experiment / dataset


def load_classifier_table(bucket: str, experiment: str, dataset: str) -> list[dict]:
    path = result_dir(bucket, experiment, dataset) / "model_results.pkl"
    if not exists(path):
        return []
    return walk_model_results(load_pkl(path))


def load_cv_table(bucket: str, experiment: str, dataset: str) -> list[dict]:
    path = result_dir(bucket, experiment, dataset) / "CV_results.pkl"
    if not exists(path):
        return []
    return walk_cv(load_pkl(path))


def best_row(rows: list[dict]) -> dict | None:
    scored = [r for r in rows if r.get("f1") is not None and pd.notna(r["f1"])]
    if not scored:
        return None
    return max(scored, key=lambda r: float(r["f1"]))


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleX", parent=base["Title"], fontName="Times-Bold", fontSize=22,
            leading=26, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubX", parent=base["Normal"], fontName="Times-Italic", fontSize=12,
            leading=16, textColor=STEEL, alignment=TA_CENTER, spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1X", parent=base["Heading1"], fontName="Times-Bold", fontSize=16,
            leading=20, textColor=NAVY, spaceBefore=10, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2X", parent=base["Heading2"], fontName="Times-Bold", fontSize=13,
            leading=16, textColor=NAVY, spaceBefore=8, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyX", parent=base["Normal"], fontName="Times-Roman", fontSize=10,
            leading=14, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "note": ParagraphStyle(
            "NoteX", parent=base["Normal"], fontName="Times-Italic", fontSize=9,
            leading=12, textColor=colors.HexColor("#444444"), spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "CellX", parent=base["Normal"], fontName="Times-Roman", fontSize=8,
            leading=10, alignment=TA_LEFT,
        ),
        "cellc": ParagraphStyle(
            "CellC", parent=base["Normal"], fontName="Times-Roman", fontSize=8,
            leading=10, alignment=TA_CENTER,
        ),
        "head": ParagraphStyle(
            "HeadX", parent=base["Normal"], fontName="Times-Bold", fontSize=8,
            leading=10, textColor=colors.white, alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "FootX", parent=base["Normal"], fontName="Times-Roman", fontSize=8,
            textColor=colors.HexColor("#555555"), alignment=TA_CENTER,
        ),
    }


def P(text: str, style) -> Paragraph:
    return Paragraph(str(text), style)


def make_table(header: list[str], rows: list[list], col_widths=None, highlight_f1=False):
    s = styles()
    data = [[P(h, s["head"]) for h in header]]
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            sty = s["cellc"] if i >= max(1, len(row) - 5) else s["cell"]
            cells.append(P(value, sty))
        data.append(cells)
    table = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C5D0DE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    if highlight_f1 and rows and "F1" in header:
        f1_idx = header.index("F1")
        f1_vals = []
        for i, row in enumerate(rows, start=1):
            try:
                f1_vals.append((i, float(row[f1_idx])))
            except (TypeError, ValueError, IndexError):
                continue
        if f1_vals:
            best_i = max(f1_vals, key=lambda t: t[1])[0]
            cmds.append(("BACKGROUND", (0, best_i), (-1, best_i), BEST))
    table.setStyle(TableStyle(cmds))
    return table


def classifier_detail_table(rows: list[dict], sty) -> list:
    if not rows:
        return [P("Results are not generated yet for this experiment on this dataset.", sty["note"])]
    header = ["Snapshot / setting", "Model", "Accuracy", "Precision", "Recall", "F1"]
    body = []
    for row in sorted(rows, key=lambda r: (str(r.get("setting")), MODELS.index(r["model"]) if r.get("model") in MODELS else 99)):
        body.append([
            row.get("setting", "—"),
            MODEL_LABEL.get(row.get("model"), str(row.get("model"))),
            fmt(row.get("accuracy")),
            fmt(row.get("precision")),
            fmt(row.get("recall")),
            fmt(row.get("f1")),
        ])
    widths = [55*mm, 40*mm, 32*mm, 32*mm, 32*mm, 28*mm]
    return [make_table(header, body, widths, highlight_f1=True)]


def cv_table(rows: list[dict], sty) -> list:
    if not rows:
        return []
    header = ["Snapshot / setting", "Model", "Mean CV accuracy", "Std. deviation"]
    body = []
    for row in rows:
        body.append([
            row.get("setting", "—"),
            MODEL_LABEL.get(row.get("model"), str(row.get("model"))),
            fmt(row.get("mean")),
            fmt(row.get("std")),
        ])
    return [
        P("Cross-validation (training-set resampling; not the held-out test score).", sty["note"]),
        make_table(header, body, [60*mm, 45*mm, 45*mm, 40*mm]),
    ]


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, landscape(A4)[1] - 12*mm, landscape(A4)[0], 12*mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Times-Bold", 9)
    canvas.drawString(14*mm, landscape(A4)[1] - 8*mm, "Enhancing Loan Default Prediction Using TDA")
    canvas.setFont("Times-Roman", 9)
    canvas.drawRightString(landscape(A4)[0] - 14*mm, landscape(A4)[1] - 8*mm, "Statlog and Default of Credit Card Client")
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(14*mm, 8*mm, "Generated from live 6_Results artefacts. Public process names from utils.TDA_PROCESS_REGISTRY.")
    canvas.drawRightString(landscape(A4)[0] - 14*mm, 8*mm, f"Page {doc.page}")
    canvas.restoreState()


def collect():
    processes = list(TDA_PROCESS_REGISTRY.items())
    payload = {}
    for folder, label in DATASETS:
        payload[folder] = {"label": label, "baselines": {}, "processes": {}}
        for exp, desc in (
            ("1_ML_Default_Parameters", "Tabular features, default hyperparameters"),
            ("2_ML_Tuned_Parameters", "Tabular features, GridSearchCV"),
        ):
            payload[folder]["baselines"][exp] = {
                "description": desc,
                "test": load_classifier_table("Default_Parameters", exp, folder),
                "cv": load_cv_table("Default_Parameters", exp, folder),
            }
        for slug, spec in processes:
            if spec["homology"] == "H0":
                experiments = [("1_PH_Default_Parameters", "Default classifiers on H0 barcode statistics")]
            else:
                experiments = [
                    ("1_PH_Default_Parameters", "Default classifiers on H0 and H1 barcode statistics"),
                    ("2_PH_Tuned_Parameters", "Retuned classifiers on H0 and H1 barcode statistics"),
                ]
            block = {"spec": spec, "display": process_display_name(slug), "experiments": {}}
            for exp, desc in experiments:
                block["experiments"][exp] = {
                    "description": desc,
                    "test": load_classifier_table(slug, exp, folder),
                    "cv": load_cv_table(slug, exp, folder),
                }
            a2_csv = result_dir(slug, "8_Null_Hypothesis_Algorithm2", folder) / "algorithm2_permutation_results.csv"
            block["algorithm2"] = pd.read_csv(openable(a2_csv)) if exists(a2_csv) else None
            audit_csv = result_dir(slug, "6_Sampling_Ratio_Audit", folder) / "sampling_ratio_audit.csv"
            block["audit"] = pd.read_csv(openable(audit_csv)) if exists(audit_csv) else None
            ml_csv = result_dir(slug, "9_Revised_Snapshot_Protocol", folder) / "ml_results.csv"
            block["revised"] = pd.read_csv(openable(ml_csv)) if exists(ml_csv) else None
            payload[folder]["processes"][slug] = block
    return payload


def headline_rows(dataset_block: dict) -> list[list[str]]:
    rows = []
    for exp, item in dataset_block["baselines"].items():
        best = best_row(item["test"])
        name = "Tabular ML, default parameters" if exp.startswith("1_") else "Tabular ML, tuned parameters"
        if not best:
            rows.append([name, "Original features", "—", "—", "—", "—", "—", "Not generated"])
            continue
        rows.append([
            name,
            best.get("setting", "Original features"),
            MODEL_LABEL.get(best["model"], best["model"]),
            fmt(best["accuracy"]), fmt(best["precision"]), fmt(best["recall"]), fmt(best["f1"]),
            "Available",
        ])
    for slug, block in dataset_block["processes"].items():
        test = block["experiments"]["1_PH_Default_Parameters"]["test"]
        if not test:
            rows.append([block["display"], "—", "—", "—", "—", "—", "—", "Not generated"])
            continue
        by_setting: dict[str, list] = {}
        for row in test:
            by_setting.setdefault(row["setting"], []).append(row)
        for setting in sorted(by_setting):
            best = best_row(by_setting[setting])
            if not best:
                continue
            rows.append([
                block["display"],
                setting,
                MODEL_LABEL.get(best["model"], best["model"]),
                fmt(best["accuracy"]), fmt(best["precision"]), fmt(best["recall"]), fmt(best["f1"]),
                "Available",
            ])
    return rows


def coverage_rows(payload: dict) -> list[list[str]]:
    rows = []
    for folder, label in DATASETS:
        block = payload[folder]
        for exp, item in block["baselines"].items():
            status = "Test metrics" + (" and CV" if item["cv"] else "")
            if not item["test"]:
                status = "Missing"
            rows.append([label, "Tabular baseline", exp.replace("_", " "), status])
        for slug, proc in block["processes"].items():
            default = proc["experiments"]["1_PH_Default_Parameters"]["test"]
            status = "Default classifiers"
            if proc["experiments"].get("2_PH_Tuned_Parameters", {}).get("test"):
                status += "; tuned"
            if proc["algorithm2"] is not None:
                status += "; Algorithm 2"
            else:
                status += "; Algorithm 2 not generated"
            if proc["revised"] is not None:
                status += "; revised protocol"
            if not default:
                status = "Missing default classifiers"
            rows.append([label, proc["display"], "Live process", status])
    return rows


def algorithm2_table(frame: pd.DataFrame | None, sty):
    if frame is None or frame.empty:
        return [P("Algorithm 2 has not been generated for this process on this dataset.", sty["note"])]
    header = ["Snapshot", "Contrast (p, q)", "Observed statistic", "p-value", "n1", "n2"]
    body = []
    for _, row in frame.iterrows():
        body.append([
            snapshot_label(row.get("source", "")),
            f"({int(row['p'])}, {int(row['q'])})" if pd.notna(row.get("p")) else "—",
            fmt(row.get("observed_F_pq"), 4),
            fmt(row.get("p_value"), 3),
            str(int(row["n1"])) if pd.notna(row.get("n1")) else "—",
            str(int(row["n2"])) if pd.notna(row.get("n2")) else "—",
        ])
    return [
        P("Robinson–Turner Algorithm 2 permutation test on barcode-vector proxies. Small p-values mean the class contrast is unusual under a random label shuffle.", sty["note"]),
        make_table(header, body, [40*mm, 40*mm, 45*mm, 30*mm, 25*mm, 25*mm]),
    ]


def audit_table(frame: pd.DataFrame | None, sty):
    if frame is None or frame.empty:
        return []
    keep = frame.copy()
    if "l_rule" in keep.columns:
        keep = keep[keep["l_rule"].astype(str).str.contains("historical|revised", case=False, na=False)]
    cols = [c for c in ("snapshot_size_percent_of_class", "l_rule", "points_per_snapshot", "n_snapshots", "reuse_ratio", "class") if c in keep.columns]
    if not cols:
        return []
    slim = keep[cols].drop_duplicates()
    header = ["Snapshot size %", "Rule", "Points / snapshot", "Number of snapshots", "Reuse ratio", "Class"]
    body = []
    for _, row in slim.iterrows():
        body.append([
            fmt(row.get("snapshot_size_percent_of_class"), 1) if "snapshot_size_percent_of_class" in slim.columns else "—",
            str(row.get("l_rule", "—")).replace("_", " "),
            fmt(row.get("points_per_snapshot"), 0),
            fmt(row.get("n_snapshots"), 0),
            fmt(row.get("reuse_ratio"), 2),
            str(row.get("class", "—")),
        ])
    return [
        P("Sampling-ratio audit (H0 and H1 processes only). Reuse ratio = (points per snapshot × number of snapshots) / minority class count.", sty["note"]),
        make_table(header, body, [32*mm, 50*mm, 35*mm, 40*mm, 30*mm, 25*mm]),
    ]


def revised_table(frame: pd.DataFrame | None, sty):
    if frame is None or frame.empty:
        return []
    keep = frame.copy()
    if "mode" in keep.columns and (keep["mode"] == "default_60_15").any():
        keep = keep[keep["mode"] == "default_60_15"]
    header = ["Model", "Accuracy", "Balanced acc.", "Precision", "Recall", "F1", "ROC AUC"]
    body = []
    f1_col = "f1" if "f1" in keep.columns else "f1_score"
    for _, row in keep.iterrows():
        body.append([
            MODEL_LABEL.get(str(row.get("model")), str(row.get("model"))),
            fmt(row.get("accuracy")),
            fmt(row.get("balanced_accuracy")),
            fmt(row.get("precision")),
            fmt(row.get("recall")),
            fmt(row.get(f1_col)),
            fmt(row.get("roc_auc")),
        ])
    return [
        P("Revised snapshot protocol (fixed points per snapshot; default 60 training / 15 test snapshots). H0 and H1 processes only.", sty["note"]),
        make_table(header, body, [40*mm, 28*mm, 30*mm, 28*mm, 28*mm, 26*mm, 28*mm], highlight_f1=True),
    ]


def build_story(payload: dict):
    sty = styles()
    story = []
    story.append(Spacer(1, 18*mm))
    story.append(P("Compiled experiment results", sty["title"]))
    story.append(P("Statlog German Credit and Default of Credit Card Client", sty["subtitle"]))
    story.append(P(
        "This report gathers live classifier scores, cross-validation where it exists, "
        "Algorithm 2 permutation tests, sampling-ratio audits, and revised-snapshot protocol "
        "scores for the two paper datasets. It covers the tabular ML baselines and the eight "
        "named TDA processes (split × undersample × just H0 vs both H0 and H1). "
        "Public names always use “and”, never “+”. Nested extras (drop-correlated columns, "
        "linear regression, mean/variance) are archived and are not included.",
        sty["body"],
    ))
    story.append(P(
        f"Generated {date.today().isoformat()} from artefacts under 6_Results/. "
        "H0 processes slice homology-0 barcode statistics from the matching H0-and-H1 run; "
        "they do not rerun Ripser. Algorithm 2 on the four just-H0 folders has not been generated yet. "
        "Green rows mark the highest F1 in a table when that comparison is meaningful. "
        "Cross-validation is training-set resampling and is not mixed with held-out test scores.",
        sty["note"],
    ))

    story.append(P("Coverage", sty["h1"]))
    story.append(make_table(
        ["Dataset", "Process / family", "Experiment", "Artefacts"],
        coverage_rows(payload),
        [48*mm, 85*mm, 45*mm, 85*mm],
    ))

    for folder, _label in DATASETS:
        block = payload[folder]
        story.append(PageBreak())
        story.append(P(block["label"], sty["h1"]))
        story.append(P(
            "Headline numbers below take, for each process and snapshot size, the classifier "
            "with the highest held-out F1. Full model-by-model tables follow.",
            sty["body"],
        ))
        story.append(P("Headline held-out F1", sty["h2"]))
        story.append(make_table(
            ["Process", "Snapshot / setting", "Best model", "Accuracy", "Precision", "Recall", "F1", "Status"],
            headline_rows(block),
            [78*mm, 32*mm, 32*mm, 22*mm, 22*mm, 20*mm, 18*mm, 28*mm],
            highlight_f1=True,
        ))

        story.append(P("Tabular ML baselines", sty["h2"]))
        for exp, item in block["baselines"].items():
            story.append(P(item["description"], sty["body"]))
            story.extend(classifier_detail_table(item["test"], sty))
            story.extend(cv_table(item["cv"], sty))
            story.append(Spacer(1, 4*mm))

        for slug, proc in block["processes"].items():
            story.append(KeepTogether([
                P(proc["display"], sty["h2"]),
                P(proc["experiments"]["1_PH_Default_Parameters"]["description"] + ".", sty["body"]),
            ]))
            story.extend(classifier_detail_table(proc["experiments"]["1_PH_Default_Parameters"]["test"], sty))
            story.extend(cv_table(proc["experiments"]["1_PH_Default_Parameters"]["cv"], sty))
            tuned = proc["experiments"].get("2_PH_Tuned_Parameters")
            if tuned:
                story.append(P(tuned["description"] + ".", sty["body"]))
                story.extend(classifier_detail_table(tuned["test"], sty))
                story.extend(cv_table(tuned["cv"], sty))
            story.extend(algorithm2_table(proc["algorithm2"], sty))
            story.extend(audit_table(proc["audit"], sty))
            story.extend(revised_table(proc["revised"], sty))
            story.append(Spacer(1, 3*mm))
    return story


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = collect()
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=landscape(A4),
        leftMargin=12*mm,
        rightMargin=12*mm,
        topMargin=18*mm,
        bottomMargin=14*mm,
        title="Compiled TDA results — Statlog and Default of Credit Card Client",
        author="Enhancing Loan Default Prediction Using TDA",
    )
    doc.build(build_story(payload), onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
