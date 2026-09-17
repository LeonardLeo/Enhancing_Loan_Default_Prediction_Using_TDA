# Compile the supervisor HTML briefing from live Default of Credit Card Client artefacts.
from __future__ import annotations

import html
import json
import math
import os
import re
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from utils import (  # noqa: E402
    CANDIDATE_POINTS_PER_SNAPSHOT,
    N_SNAPSHOTS_GRID,
    N_TRAIN_POOL,
    TDA_PROCESS_REGISTRY,
    win_long_path,
)

OUT_DIR = ROOT / "6_Results" / "Compiled_Reports"
OUT_HTML = OUT_DIR / "TDA_Results_Briefing.html"
DATASET = "Default_Of_Credit_Card_Client_Data"
DATASET_LABEL = "Default of Credit Card Client"
TITLE = "Experiment Results from Current TDA Research"

MODELS = ("svm", "knn", "xgb", "logistic", "random_forest")
MODEL_LABEL = {
    "svm": "SVM",
    "knn": "KNN",
    "xgb": "XGBoost",
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
}

# Locked supervisor headings (not folder names, not display_name()).
PROCESS_BRIEFING = (
    {
        "slug": "Late_Split_And_Undersample_H0_And_H1",
        "heading": "Late split, undersampling, H0 and H1 — default-parameter classifiers on barcode statistics",
        "short": "Late, undersample, H0 and H1",
    },
    {
        "slug": "Late_Split_And_Undersample_H0",
        "heading": "Late split, undersampling, H0 only — default-parameter classifiers on barcode statistics",
        "short": "Late, undersample, H0",
    },
    {
        "slug": "Late_Split_No_Undersample_H0_And_H1",
        "heading": "Late split, no undersampling, H0 and H1 — default-parameter classifiers on barcode statistics",
        "short": "Late, no undersample, H0 and H1",
    },
    {
        "slug": "Late_Split_No_Undersample_H0",
        "heading": "Late split, no undersampling, H0 only — default-parameter classifiers on barcode statistics",
        "short": "Late, no undersample, H0",
    },
    {
        "slug": "Early_Split_And_Undersample_H0_And_H1",
        "heading": "Early split, undersampling, H0 and H1 — default-parameter classifiers on barcode statistics",
        "short": "Early, undersample, H0 and H1",
    },
    {
        "slug": "Early_Split_And_Undersample_H0",
        "heading": "Early split, undersampling, H0 only — default-parameter classifiers on barcode statistics",
        "short": "Early, undersample, H0",
    },
    {
        "slug": "Early_Split_No_Undersample_H0_And_H1",
        "heading": "Early split, no undersampling, H0 and H1 — default-parameter classifiers on barcode statistics",
        "short": "Early, no undersample, H0 and H1",
    },
    {
        "slug": "Early_Split_No_Undersample_H0",
        "heading": "Early split, no undersampling, H0 only — default-parameter classifiers on barcode statistics",
        "short": "Early, no undersample, H0",
    },
)

SSS_PROTOCOL_ORDER = (
    "Historical_Late_Split_Balanced_TDA",
    "Early_Split_TDA",
    "No_Undersampling",
    "Early_Split_TDA_And_No_Undersampling",
)
SSS_PROTOCOL_SHORT = {
    "Historical_Late_Split_Balanced_TDA": "Late split and undersample",
    "Early_Split_TDA": "Early split and undersample",
    "No_Undersampling": "Late split, no undersample",
    "Early_Split_TDA_And_No_Undersampling": "Early split, no undersample",
}

NAVY = "#1B365D"
STEEL = "#4A6FA5"
GOLD = "#C4A35A"
INK = "#1B1B1B"
PROTOCOL_COLORS = {
    "Historical_Late_Split_Balanced_TDA": NAVY,
    "Early_Split_TDA": STEEL,
    "No_Undersampling": GOLD,
    "Early_Split_TDA_And_No_Undersampling": "#8B3A3A",
}
CLOUD_SIZE_COLORS = {
    15: "#0072B2",
    30: "#D55E00",
    45: "#009E73",
    60: "#CC79A7",
    90: "#56B4E9",
    120: "#E69F00",
    180: "#6A3D9A",
    240: "#8B3A3A",
    330: "#1B365D",
}
MODEL_COLORS = {
    "svm": NAVY,
    "knn": STEEL,
    "xgb": GOLD,
    "logistic": "#0072B2",
    "random_forest": "#8B3A3A",
}
HEATMAP_X = {
    "svm": "SVM",
    "knn": "KNN",
    "xgb": "XGBoost",
    "logistic": "Logistic",
    "random_forest": "Random forest",
}
HEATMAP_Y = {
    "Late_Split_And_Undersample_H0_And_H1": "Late, undersample · H0 and H1",
    "Late_Split_And_Undersample_H0": "Late, undersample · H0 only",
    "Late_Split_No_Undersample_H0_And_H1": "Late, no undersample · H0 and H1",
    "Late_Split_No_Undersample_H0": "Late, no undersample · H0 only",
    "Early_Split_And_Undersample_H0_And_H1": "Early, undersample · H0 and H1",
    "Early_Split_And_Undersample_H0": "Early, undersample · H0 only",
    "Early_Split_No_Undersample_H0_And_H1": "Early, no undersample · H0 and H1",
    "Early_Split_No_Undersample_H0": "Early, no undersample · H0 only",
}


def openable(path: Path) -> str:
    return str(win_long_path(path))


def exists(path: Path) -> bool:
    return os.path.isfile(openable(path))


def load_pkl(path: Path):
    return joblib.load(openable(path))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(openable(path))


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def fmt_display(value, digits: int = 3) -> str:
    if is_missing(value):
        return "not generated"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "not generated"
    if abs(number - round(number)) < 1e-12 and abs(number) >= 1:
        return f"{int(round(number)):,}"
    return f"{number:.{digits}f}"


def td(value, digits: int = 3, kind: str = "num") -> str:
    if is_missing(value):
        return "<td class='miss'>not generated</td>"
    if kind == "int":
        number = int(value)
        return f"<td data-value='{number}'>{number:,}</td>"
    if kind == "text":
        return f"<td>{esc(value)}</td>"
    number = float(value)
    if not math.isfinite(number):
        return "<td class='miss'>not generated</td>"
    shown = fmt_display(number, digits)
    return f"<td data-value='{number:.12g}'>{shown}</td>"


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


def walk_model_results(obj, setting_hint: str | None = None) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(obj, dict) or not obj:
        return rows
    looks_like_models = any(
        isinstance(value, dict) and any(k in value for k in ("accuracy", "f1_score", "f1"))
        for value in obj.values()
    )
    if looks_like_models:
        for model, stats in obj.items():
            if not isinstance(stats, dict):
                continue
            if "accuracy" not in stats and "f1_score" not in stats and "f1" not in stats:
                continue
            row = metric_tuple(stats)
            row.update({"setting": setting_hint or "Original features", "model": str(model)})
            rows.append(row)
        return rows
    for key, inner in obj.items():
        if isinstance(inner, dict):
            rows.extend(walk_model_results(inner, snapshot_label(key)))
    return rows


def fig_html(fig: go.Figure, fig_id: str, **layout_kwargs) -> str:
    showlegend = layout_kwargs.get("showlegend", True)
    title = layout_kwargs.get("title", None)
    if title in ("", None):
        layout_kwargs.pop("title", None)
        title = None
    layout = dict(
        font=dict(family="Georgia, 'Times New Roman', serif", size=13, color=INK),
        paper_bgcolor="white",
        plot_bgcolor="#F7F8FA",
        margin=dict(l=72, r=36, t=84 if title else 28, b=110 if showlegend else 52),
        hoverlabel=dict(namelength=-1),
        showlegend=showlegend,
    )
    if title:
        layout["title"] = dict(text=title, x=0, xanchor="left", pad=dict(b=10))
    if showlegend:
        layout["legend"] = dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            x=0,
            xanchor="left",
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#d9d4c8",
            borderwidth=1,
            font=dict(size=11),
            tracegroupgap=8,
        )
    layout.update(layout_kwargs)
    lazy = bool(layout.pop("lazy", False))
    fig.update_layout(**layout)
    if fig.layout.annotations:
        fig.update_annotations(font=dict(size=12, color=NAVY))
    config = {"displaylogo": False, "responsive": True}
    if lazy:
        return (
            f'<div class="plotly-lazy" id="{esc(fig_id)}"></div>'
            f'<script type="application/json" id="{esc(fig_id)}-spec">{fig.to_json()}</script>'
        )
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=fig_id,
        config=config,
    )


def html_table(
    headers: list[str],
    body_rows: list[list[str]],
    caption: str = "",
    *,
    table_id: str = "",
    row_attrs: list[dict] | None = None,
) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body_bits = []
    for idx, cells in enumerate(body_rows):
        extra = ""
        if row_attrs and idx < len(row_attrs):
            extra = "".join(
                f' data-{esc(key)}="{esc(value)}"' for key, value in row_attrs[idx].items()
            )
        body_bits.append("<tr" + extra + ">" + "".join(cells) + "</tr>")
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    id_attr = f' id="{esc(table_id)}"' if table_id else ""
    return (
        f"<div class='table-wrap'><table{id_attr}>{cap}"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_bits)}</tbody></table></div>"
    )


def html_swatch_legend(items: list[tuple[str, str]], *, for_view: str | None = None) -> str:
    chips = "".join(
        f'<span class="swatch"><i style="background:{esc(color)}"></i>{esc(label)}</span>'
        for label, color in items
    )
    for_attr = f' data-for="{esc(for_view)}"' if for_view else ""
    return f'<div class="swatch-legend"{for_attr}>{chips}</div>'


def chart_card(
    title: str,
    figure: str,
    *,
    view: str | None = None,
    key: str | None = None,
    order: int = 0,
    model: str | None = None,
    protocol: str | None = None,
) -> str:
    attrs = [f'data-order="{order}"']
    if view is not None:
        attrs.append(f'data-view="{esc(view)}"')
    if key is not None:
        attrs.append(f'data-key="{esc(key)}"')
    if model is not None:
        attrs.append(f'data-model="{esc(model)}"')
    if protocol is not None:
        attrs.append(f'data-protocol="{esc(protocol)}"')
    return (
        f'<article class="chart-card" {" ".join(attrs)}>'
        f'<p class="chart-title">{esc(title)}</p>{figure}</article>'
    )


def fold(title: str, inner: str, *, opened: bool = False, n_rows: int | None = None) -> str:
    if not inner:
        return inner
    label = title if n_rows is None else f"{title} ({n_rows} rows)"
    open_attr = " open" if opened else ""
    return f"<details class='table-fold'{open_attr}><summary>{esc(label)}</summary>{inner}</details>"


def six_part(research: str, run: str, formula: str, numbers: str, why: str, limits: str) -> str:
    return f"""
<div class="six">
  <h3>What was being researched</h3>
  <div class="prose">{research}</div>
  <h3>What was run</h3>
  <div class="prose">{run}</div>
  <h3>Formula and worked Default of Credit Card Client numbers</h3>
  <div class="prose">{formula}</div>
  <h3>What the numbers were</h3>
  <div class="prose">{numbers}</div>
  <h3>Why they look like that</h3>
  <div class="prose">{why}</div>
  <h3>What the result does not prove</h3>
  <div class="prose">{limits}</div>
</div>
"""


def path_line(*parts: str) -> str:
    joined = "/".join(parts)
    return f"<p class='path'>Artefact: <code>{esc(joined)}</code></p>"


def spec_of(slug: str) -> dict:
    return dict(TDA_PROCESS_REGISTRY[slug])


def rel_results(*parts: str) -> Path:
    return ROOT.joinpath("6_Results", *parts)


# ---------------------------------------------------------------------------
# Artefact loaders
# ---------------------------------------------------------------------------
def load_tabular_default() -> list[dict]:
    path = rel_results("Default_Parameters", "1_ML_Default_Parameters", DATASET, "model_results.pkl")
    if not exists(path):
        return []
    return walk_model_results(load_pkl(path))


def load_process_classifiers(slug: str, experiment: str) -> tuple[list[dict], str]:
    rel = f"6_Results/{slug}/{experiment}/{DATASET}/model_results.pkl"
    path = rel_results(slug, experiment, DATASET, "model_results.pkl")
    if not exists(path):
        return [], rel
    return walk_model_results(load_pkl(path)), rel


def load_process_csv(slug: str, experiment: str, filename: str) -> tuple[pd.DataFrame, str]:
    rel = f"6_Results/{slug}/{experiment}/{DATASET}/{filename}"
    path = rel_results(slug, experiment, DATASET, filename)
    if not exists(path):
        return pd.DataFrame(), rel
    return load_csv(path), rel


def load_id_bundle() -> dict:
    folder = rel_results("Statistics", "1_Intrinsic_Dimension_Estimation", DATASET)
    csv_path = folder / "intrinsic_dimension_estimates.csv"
    pkl_path = folder / "intrinsic_dimension_estimates.pkl"
    bundle = {
        "csv": load_csv(csv_path) if exists(csv_path) else pd.DataFrame(),
        "pkl": load_pkl(pkl_path) if exists(pkl_path) else {},
        "csv_rel": "6_Results/Statistics/1_Intrinsic_Dimension_Estimation/"
        f"{DATASET}/intrinsic_dimension_estimates.csv",
        "pkl_rel": "6_Results/Statistics/1_Intrinsic_Dimension_Estimation/"
        f"{DATASET}/intrinsic_dimension_estimates.pkl",
    }
    return bundle


def load_sss(folder: str) -> pd.DataFrame:
    path = rel_results("Snapshot_Sample_Size", folder, "all_summary.csv")
    if not exists(path):
        return pd.DataFrame()
    frame = load_csv(path)
    if "folder_name" in frame.columns:
        frame = frame[frame["folder_name"] == DATASET].copy()
    return frame


def audit_l5_row(frame: pd.DataFrame, l_rule: str, class_name: str) -> pd.Series | None:
    if frame.empty:
        return None
    subset = frame[
        (frame.get("l_rule") == l_rule)
        & (frame.get("snapshot_size_percent_of_class") == 5.0)
        & (frame.get("class") == class_name)
    ]
    if "split" in subset.columns:
        preferred = subset[subset["split"].isin(["full_table", "train"])]
        if not preferred.empty:
            subset = preferred
    if subset.empty:
        return None
    return subset.iloc[0]


def metric_lookup(rows: list[dict], setting: str, model: str, field: str):
    for row in rows:
        if row.get("setting") == setting and row.get("model") == model:
            return row.get(field)
    return None


def best_f1(rows: list[dict], setting: str):
    values = [row.get("f1") for row in rows if row.get("setting") == setting and not is_missing(row.get("f1"))]
    return max(values) if values else None


def model_table(rows: list[dict], settings: tuple[str, ...] | None = None) -> str:
    if not rows:
        return "<p class='miss'>not generated</p>"
    used_settings = settings or tuple(dict.fromkeys(row["setting"] for row in rows))
    headers = ["Snapshot size", "Model", "Accuracy", "Precision", "Recall", "F1"]
    body = []
    for setting in used_settings:
        for model in MODELS:
            row = next((r for r in rows if r["setting"] == setting and r["model"] == model), None)
            if row is None:
                body.append(
                    [
                        td(setting, kind="text"),
                        td(MODEL_LABEL.get(model, model), kind="text"),
                        "<td class='miss'>not generated</td>",
                        "<td class='miss'>not generated</td>",
                        "<td class='miss'>not generated</td>",
                        "<td class='miss'>not generated</td>",
                    ]
                )
                continue
            body.append(
                [
                    td(setting, kind="text"),
                    td(MODEL_LABEL.get(model, model), kind="text"),
                    td(row.get("accuracy")),
                    td(row.get("precision")),
                    td(row.get("recall")),
                    td(row.get("f1")),
                ]
            )
    return html_table(headers, body)


def f1_grid_table(rows_map: dict[str, list[dict]]) -> str:
    headers = ["Process"]
    for setting in ("L5", "L15"):
        for model in MODELS:
            headers.append(f"{MODEL_LABEL[model]} · {setting}")
    body = []
    for item in PROCESS_BRIEFING:
        cells = [td(item["short"], kind="text")]
        for setting in ("L5", "L15"):
            for model in MODELS:
                cells.append(td(metric_lookup(rows_map.get(item["slug"], []), setting, model, "f1")))
        body.append(cells)
    return html_table(
        headers,
        body,
        caption="Every library-default model, both snapshot sizes. F1 on barcode rows.",
    )


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------
def _finish_single_chart(fig: go.Figure, fig_id: str, *, y_title: str, x_title: str, height: int = 440) -> str:
    fig.update_yaxes(title_text=y_title, range=[0, 1.08], title_standoff=10, tickfont=dict(size=12))
    fig.update_xaxes(title_text=x_title, tickfont=dict(size=12), title_standoff=8)
    return fig_html(
        fig,
        fig_id,
        showlegend=False,
        height=height,
        margin=dict(l=76, r=28, t=24, b=56),
        title="",
    )


def plot_tabular(rows: list[dict]) -> str:
    fig = go.Figure()
    fig.add_bar(
        x=[MODEL_LABEL[m] for m in MODELS],
        y=[metric_lookup(rows, "Original features", m, "f1") for m in MODELS],
        marker_color=NAVY,
        name="F1",
        hovertemplate="%{x}<br>F1=%{y:.3f}<extra></extra>",
    )
    fig.update_layout(yaxis_title="F1", yaxis=dict(range=[0, 1.05]), bargap=0.35)
    return fig_html(
        fig,
        "fig-tabular-f1",
        title="Tabular baseline F1 by model (library-default classifiers)",
        showlegend=False,
        height=420,
        margin=dict(l=72, r=36, t=88, b=64),
    )


def plot_process_f1(rows: list[dict], fig_id: str, heading: str) -> str:
    if not rows:
        return "<p class='miss'>not generated</p>"
    settings = [s for s in ("L5", "L15") if any(r.get("setting") == s for r in rows)]
    if not settings:
        settings = list(dict.fromkeys(r.get("setting") for r in rows if r.get("setting")))
    colors = {"L5": NAVY, "L15": STEEL}
    fig = go.Figure()
    legend_items = []
    for setting in settings:
        color = colors.get(setting, NAVY)
        label = "L5 (5% of class)" if setting == "L5" else "L15 (15% of class)" if setting == "L15" else setting
        legend_items.append((label, color))
        fig.add_bar(
            name=label,
            x=[MODEL_LABEL[m] for m in MODELS],
            y=[metric_lookup(rows, setting, m, "f1") for m in MODELS],
            marker_color=color,
            hovertemplate=f"%{{x}}<br>{esc(label)} F1=%{{y:.3f}}<extra></extra>",
        )
    fig.update_layout(
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        yaxis_title="F1",
        yaxis=dict(range=[0, 1.08]),
    )
    return html_swatch_legend(legend_items) + fig_html(
        fig,
        fig_id,
        title=heading,
        showlegend=False,
        height=440,
        margin=dict(l=72, r=36, t=84, b=64),
    )


def plot_eight_process_heatmap(process_rows: dict[str, list[dict]]) -> str:
    y_labels = [HEATMAP_Y[item["slug"]] for item in PROCESS_BRIEFING]
    x_labels = [HEATMAP_X[m] for m in MODELS]
    panels = []
    for setting in ("L5", "L15"):
        z = []
        text = []
        for item in PROCESS_BRIEFING:
            row_z = []
            row_t = []
            for model in MODELS:
                value = metric_lookup(process_rows.get(item["slug"], []), setting, model, "f1")
                row_z.append(value if not is_missing(value) else None)
                row_t.append("" if is_missing(value) else f"{float(value):.3f}")
            z.append(row_z)
            text.append(row_t)
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 13, "color": "#102a43"},
                coloraxis="coloraxis",
                hovertemplate="%{y}<br>%{x}<br>F1=%{z:.3f}<extra></extra>",
                xgap=4,
                ygap=4,
            )
        )
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), ticks="")
        fig.update_xaxes(tickfont=dict(size=12), side="bottom", ticks="")
        html_fig = fig_html(
            fig,
            f"fig-eight-heatmap-{setting.lower()}",
            title=f"{setting} F1 for every process and every model",
            coloraxis=dict(
                colorscale="Teal",
                cmin=0,
                cmax=1,
                colorbar=dict(title="F1", len=0.78, y=0.47, yanchor="middle", thickness=16, x=1.02),
            ),
            showlegend=False,
            height=620,
            margin=dict(l=248, r=96, t=80, b=72),
        )
        hidden = " hidden" if setting != "L5" else ""
        panels.append(
            f'<div class="heatmap-panel" data-setting="{setting}"{hidden}>{html_fig}</div>'
        )
    return f"""
<div class="heatmap-explorer" id="heatmap-explorer">
  <div class="joint-controls">
    <label>Snapshot size
      <select id="heatmap-setting">
        <option value="L5" selected>L5 (5% of class)</option>
        <option value="L15">L15 (15% of class)</option>
      </select>
    </label>
  </div>
  <p class="joint-help">Both snapshot sizes and all five models are in this section. Switch L5 / L15 here; the folded table under the chart lists every process × model cell.</p>
  {''.join(panels)}
</div>
"""


def plot_id(bundle: dict) -> str:
    pkl = bundle.get("pkl") or {}
    suite = pkl.get("suite") or {}
    estimators = [
        ("Two-NN", "handcoded_two_nn", "intrinsic_dim_two_nn"),
        ("Levina–Bickel (k = 10)", "handcoded_levina_bickel", "intrinsic_dim_levina_bickel"),
        ("skdim TwoNN", "skdim", None),
    ]
    labels = []
    before = []
    after = []
    for label, key, field in estimators:
        labels.append(label)
        before_pack = (suite.get("before_pca") or {}).get(key) or {}
        after_pack = (suite.get("after_pca") or {}).get(key) or {}
        if key == "skdim":
            before.append((before_pack.get("estimators") or {}).get("TwoNN"))
            after.append((after_pack.get("estimators") or {}).get("TwoNN"))
        else:
            before.append(before_pack.get(field))
            after.append(after_pack.get(field))
    row = bundle["csv"].iloc[0] if not bundle["csv"].empty else None
    if row is not None and is_missing(before[2]):
        before[2] = row.get("skdim_TwoNN_before_pca")
        after[2] = row.get("skdim_TwoNN_after_pca")
    fig = go.Figure()
    fig.add_bar(name="Before PCA", x=labels, y=before, marker_color=NAVY)
    fig.add_bar(name="After Exp 3 PCA (7 components)", x=labels, y=after, marker_color=STEEL)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    legend = html_swatch_legend(
        [("Before PCA", NAVY), ("After Exp 3 PCA (7 components)", STEEL)]
    )
    return legend + fig_html(
        fig,
        "fig-id",
        title="Intrinsic dimension before PCA and after the Exp 3 PCA",
        yaxis_title="Estimated intrinsic dimension",
        showlegend=False,
        height=440,
        margin=dict(l=72, r=36, t=84, b=64),
    )


def _sss_protocol_label(protocol: str) -> str:
    return SSS_PROTOCOL_SHORT.get(protocol, protocol)


def _cloud_color(size) -> str:
    return CLOUD_SIZE_COLORS.get(int(size), NAVY)


def _add_sss_line(fig: go.Figure, subset: pd.DataFrame, *, name: str, color: str, x_col: str, x_title: str, meta=None) -> None:
    if subset.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=subset[x_col],
            y=subset["f1_mean"],
            mode="lines+markers",
            name=name,
            meta=meta,
            showlegend=False,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=8),
            customdata=subset["reuse_ratio"].to_numpy(),
            hovertemplate=(
                f"{esc(name)}<br>{x_title}=%{{x}}<br>F1 mean=%{{y:.3f}}"
                "<br>reuse ratio=%{customdata:.3f}<extra></extra>"
            ),
        )
    )


def plot_sss_by_model(frame: pd.DataFrame, model: str, x_col: str, x_title: str, fig_id: str) -> str:
    fig = go.Figure()
    protocols = [p for p in SSS_PROTOCOL_ORDER if p in set(frame.get("protocol", pd.Series(dtype=str)))]
    for protocol in protocols:
        subset = frame[(frame["model"] == model) & (frame["protocol"] == protocol)].sort_values(x_col)
        _add_sss_line(
            fig,
            subset,
            name=_sss_protocol_label(protocol),
            color=PROTOCOL_COLORS.get(protocol, NAVY),
            x_col=x_col,
            x_title=x_title,
        )
    return _finish_single_chart(fig, fig_id, y_title="F1 mean", x_title=x_title)


def plot_sss_by_protocol(frame: pd.DataFrame, protocol: str, x_col: str, x_title: str, fig_id: str) -> str:
    fig = go.Figure()
    models = [m for m in MODELS if m in set(frame.get("model", pd.Series(dtype=str)))]
    for model in models:
        subset = frame[(frame["model"] == model) & (frame["protocol"] == protocol)].sort_values(x_col)
        _add_sss_line(
            fig,
            subset,
            name=MODEL_LABEL.get(model, model),
            color=MODEL_COLORS.get(model, NAVY),
            x_col=x_col,
            x_title=x_title,
        )
    return _finish_single_chart(fig, fig_id, y_title="F1 mean", x_title=x_title)


def _joint_grid_matrices(
    frame: pd.DataFrame, model: str, protocol: str
) -> dict | None:
    subset = frame[(frame["model"] == model) & (frame["protocol"] == protocol)]
    if subset.empty:
        return None
    xs = [int(v) for v in sorted(subset["n_snapshots"].dropna().unique())]
    ys = [int(v) for v in sorted(subset["points_per_snapshot"].dropna().unique())]
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    reuse: list[list[float | None]] = []
    for pts in ys:
        row_z: list[float | None] = []
        row_t: list[str] = []
        row_r: list[float | None] = []
        for n_snap in xs:
            cell = subset[
                (subset["n_snapshots"] == n_snap) & (subset["points_per_snapshot"] == pts)
            ]
            if cell.empty:
                row_z.append(None)
                row_t.append("")
                row_r.append(None)
            else:
                f1 = float(cell.iloc[0]["f1_mean"])
                row_z.append(f1)
                row_t.append(f"{f1:.2f}")
                row_r.append(float(cell.iloc[0]["reuse_ratio"]))
        z.append(row_z)
        text.append(row_t)
        reuse.append(row_r)
    return {"xs": xs, "ys": ys, "z": z, "text": text, "reuse": reuse}


def plot_joint_surface(frame: pd.DataFrame, model: str, protocol: str, fig_id: str) -> str:
    grid = _joint_grid_matrices(frame, model, protocol)
    if grid is None:
        return "<p class='miss'>not generated</p>"
    xs = np.asarray(grid["xs"], dtype=float)
    ys = np.asarray(grid["ys"], dtype=float)
    z = np.asarray(
        [[np.nan if v is None else v for v in row] for row in grid["z"]],
        dtype=float,
    )
    xx, yy = np.meshgrid(xs, ys)
    fig = go.Figure()
    fig.add_surface(
        x=xs,
        y=ys,
        z=z,
        colorscale="Teal",
        cmin=0,
        cmax=1,
        opacity=0.93,
        connectgaps=False,
        colorbar=dict(title="F1", len=0.62, y=0.48, yanchor="middle", thickness=14, x=1.04),
        lighting=dict(ambient=0.64, diffuse=0.78, specular=0.22, roughness=0.42, fresnel=0.12),
        lightposition=dict(x=80, y=40, z=2),
        contours=dict(
            x=dict(show=False),
            y=dict(show=False),
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor=GOLD,
                project_z=False,
                width=2,
            ),
        ),
        hovertemplate=(
            "Number of snapshots=%{x}<br>Points per snapshot=%{y}<br>"
            "F1 mean=%{z:.3f}<extra></extra>"
        ),
    )
    fig.add_scatter3d(
        x=xx.flatten(),
        y=yy.flatten(),
        z=z.flatten(),
        mode="markers",
        marker=dict(
            size=3.6,
            color=z.flatten(),
            colorscale="Teal",
            cmin=0,
            cmax=1,
            showscale=False,
            line=dict(width=0),
        ),
        showlegend=False,
        hovertemplate=(
            "Number of snapshots=%{x}<br>Points per snapshot=%{y}<br>"
            "F1 mean=%{z:.3f}<extra></extra>"
        ),
    )
    axis_shared = dict(
        backgroundcolor="#F7F8FA",
        gridcolor="#d9d4c8",
        showbackground=True,
        zeroline=False,
        title_font=dict(size=13, color=NAVY),
        tickfont=dict(size=11, color=INK),
    )
    return fig_html(
        fig,
        fig_id,
        title="",
        showlegend=False,
        height=560,
        margin=dict(l=8, r=80, t=16, b=8),
        lazy=True,
        scene=dict(
            xaxis=dict(
                title="Number of snapshots",
                tickvals=list(xs),
                **axis_shared,
            ),
            yaxis=dict(
                title="Points per snapshot",
                tickvals=list(ys),
                **axis_shared,
            ),
            zaxis=dict(
                title="F1 mean",
                range=[0, 1.02],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                **axis_shared,
            ),
            camera=dict(
                eye=dict(x=-1.62, y=-1.82, z=1.18),
                center=dict(x=0, y=0, z=-0.08),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.45, y=1.6, z=1.05),
        ),
    )


def plot_joint_heatmap(frame: pd.DataFrame, model: str, protocol: str, fig_id: str) -> str:
    grid = _joint_grid_matrices(frame, model, protocol)
    if grid is None:
        return "<p class='miss'>not generated</p>"
    xs = grid["xs"]
    ys = grid["ys"]
    x_labels = [str(v) for v in xs]
    y_labels = [str(v) for v in ys]
    fig = go.Figure(
        go.Heatmap(
            z=grid["z"],
            x=x_labels,
            y=y_labels,
            text=grid["text"],
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#102a43"},
            customdata=grid["reuse"],
            coloraxis="coloraxis",
            hovertemplate=(
                "Number of snapshots=%{x}<br>Points per snapshot=%{y}<br>"
                "F1 mean=%{z:.3f}<br>reuse ratio=%{customdata:.3f}<extra></extra>"
            ),
            xgap=3,
            ygap=3,
        )
    )
    fig.update_xaxes(
        title_text="Number of snapshots",
        type="category",
        categoryorder="array",
        categoryarray=x_labels,
        tickfont=dict(size=12),
        ticks="",
        title_standoff=8,
    )
    fig.update_yaxes(
        title_text="Points per snapshot",
        type="category",
        categoryorder="array",
        categoryarray=y_labels,
        tickfont=dict(size=12),
        ticks="",
        title_standoff=10,
    )
    return fig_html(
        fig,
        fig_id,
        title="",
        coloraxis=dict(
            colorscale="Teal",
            cmin=0,
            cmax=1,
            colorbar=dict(title="F1", len=0.78, y=0.47, yanchor="middle", thickness=14, x=1.02),
        ),
        showlegend=False,
        height=500,
        margin=dict(l=108, r=88, t=28, b=64),
    )


def _select_options(pairs: list[tuple[str, str]]) -> str:
    return "".join(f'<option value="{esc(value)}">{esc(label)}</option>' for value, label in pairs)


def explorer_shell(
    explorer_id: str,
    *,
    help_text: str,
    model_opts: str,
    protocol_opts: str,
    cards: str,
    protocol_legend: str,
    model_legend: str,
    size_opts: str | None = None,
    layout: str | None = None,
) -> str:
    size_block = ""
    if size_opts is not None:
        size_block = f"""
    <label class="exp-size-label">Points per snapshot
      <select class="exp-size">{size_opts}</select>
    </label>"""
    layout_name = layout or ("pair-grid" if size_opts is not None else "")
    layout_attr = f' data-layout="{esc(layout_name)}"' if layout_name else ""
    return f"""
<div class="chart-explorer" id="{esc(explorer_id)}"{layout_attr}>
  <div class="joint-controls">
    <label>Compare
      <select class="exp-view">
        <option value="model" selected>four protocols for one model</option>
        <option value="protocol">five models for one protocol</option>
      </select>
    </label>
    <label class="exp-model-label">Model
      <select class="exp-model">{model_opts}</select>
    </label>
    <label class="exp-protocol-label" hidden>Protocol
      <select class="exp-protocol">{protocol_opts}</select>
    </label>{size_block}
  </div>
  <p class="joint-help">{help_text}</p>
  {protocol_legend}
  {model_legend}
  <div class="chart-stage panel-grid cols-1"></div>
  <div class="chart-bank" hidden>{cards}</div>
</div>
"""


def sss_line_explorer(frame: pd.DataFrame, x_col: str, x_title: str, explorer_id: str) -> str:
    if frame.empty:
        return "<p class='miss'>not generated</p>"
    protocols = [p for p in SSS_PROTOCOL_ORDER if p in set(frame.get("protocol", pd.Series(dtype=str)))]
    models = [m for m in MODELS if m in set(frame.get("model", pd.Series(dtype=str)))]
    cards = []
    for model in models:
        cards.append(
            chart_card(
                MODEL_LABEL.get(model, model),
                plot_sss_by_model(frame, model, x_col, x_title, f"{explorer_id}-{model}"),
                view="model",
                key=model,
            )
        )
    for order, protocol in enumerate(protocols):
        cards.append(
            chart_card(
                _sss_protocol_label(protocol),
                plot_sss_by_protocol(
                    frame, protocol, x_col, x_title, f"{explorer_id}-p-{protocol.lower()}"
                ),
                view="protocol",
                key=protocol,
                order=order,
            )
        )
    return explorer_shell(
        explorer_id,
        help_text=(
            "All five models and all four protocols are in the dropdowns and in the table. "
            "The chart shows one comparison at a time so axis labels and lines stay readable."
        ),
        model_opts=_select_options([(m, MODEL_LABEL[m]) for m in models]),
        protocol_opts=_select_options([(p, _sss_protocol_label(p)) for p in protocols]),
        cards="".join(cards),
        protocol_legend=html_swatch_legend(
            [(_sss_protocol_label(p), PROTOCOL_COLORS[p]) for p in protocols],
            for_view="model",
        ),
        model_legend=html_swatch_legend(
            [(MODEL_LABEL[m], MODEL_COLORS[m]) for m in models],
            for_view="protocol",
        ),
    )


def joint_grid_explorer(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='miss'>not generated</p>"
    protocols = [p for p in SSS_PROTOCOL_ORDER if p in set(frame.get("protocol", pd.Series(dtype=str)))]
    models = [m for m in MODELS if m in set(frame.get("model", pd.Series(dtype=str)))]
    cards = []
    for model_idx, model in enumerate(models):
        for proto_idx, protocol in enumerate(protocols):
            fig_id = f"fig-joint-{model}-{protocol.lower()}"
            cards.append(
                chart_card(
                    f"{MODEL_LABEL.get(model, model)} · {_sss_protocol_label(protocol)}",
                    '<p class="chart-kicker">3D surface — drag to rotate. Height is F1. '
                    "The floor is number of snapshots × points per snapshot; the surface lifts as either knob grows.</p>"
                    + plot_joint_surface(frame, model, protocol, f"{fig_id}-3d")
                    + '<p class="chart-kicker">The same cells as a heatmap.</p>'
                    + plot_joint_heatmap(frame, model, protocol, f"{fig_id}-heat"),
                    order=model_idx * 10 + proto_idx,
                    model=model,
                    protocol=protocol,
                )
            )
    return explorer_shell(
        "joint-explorer",
        help_text=(
            "Each protocol has a 3D F1 surface and the matching heatmap. Drag the surface to rotate it: "
            "height is F1, the floor axes are number of snapshots and points per snapshot. "
            "The heatmap under it is the same cells, read left to right and bottom to top. "
            "Compare four protocols for one model, or five models for one protocol."
        ),
        model_opts=_select_options([(m, MODEL_LABEL[m]) for m in models]),
        protocol_opts=_select_options([(p, _sss_protocol_label(p)) for p in protocols]),
        cards="".join(cards),
        protocol_legend="",
        model_legend="",
        layout="pair-grid",
    )


def _revised_t_values(process_data: dict) -> list[int]:
    values: set[int] = set()
    for item in PROCESS_BRIEFING:
        frame = process_data[item["slug"]]["revised"]
        if frame.empty or "t" not in frame.columns:
            continue
        values.update(int(v) for v in frame["t"].dropna().unique())
    return sorted(values)


def _revised_f1(frame: pd.DataFrame, model: str, t_pts: int):
    if frame.empty:
        return None
    subset = frame[(frame["model"] == model) & (frame["t"].astype(int) == int(t_pts))]
    if subset.empty:
        return None
    return subset.iloc[0].get("f1")


def plot_revised_heatmap(process_data: dict) -> str:
    t_values = _revised_t_values(process_data)
    if not t_values:
        return "<p class='miss'>not generated</p>"
    y_labels = [HEATMAP_Y[item["slug"]] for item in PROCESS_BRIEFING]
    x_labels = [HEATMAP_X[m] for m in MODELS]
    panels = []
    for t_pts in t_values:
        z = []
        text = []
        for item in PROCESS_BRIEFING:
            frame = process_data[item["slug"]]["revised"]
            row_z = []
            row_t = []
            for model in MODELS:
                value = _revised_f1(frame, model, t_pts)
                row_z.append(value if not is_missing(value) else None)
                row_t.append("" if is_missing(value) else f"{float(value):.3f}")
            z.append(row_z)
            text.append(row_t)
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 13, "color": "#102a43"},
                coloraxis="coloraxis",
                hovertemplate="%{y}<br>%{x}<br>F1=%{z:.3f}<extra></extra>",
                xgap=4,
                ygap=4,
            )
        )
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), ticks="")
        fig.update_xaxes(tickfont=dict(size=12), side="bottom", ticks="")
        html_fig = fig_html(
            fig,
            f"fig-revised-heatmap-{t_pts}",
            title=f"60 / 15 protocol F1 at {t_pts} points per snapshot",
            coloraxis=dict(
                colorscale="Teal",
                cmin=0,
                cmax=1,
                colorbar=dict(title="F1", len=0.78, y=0.47, yanchor="middle", thickness=16, x=1.02),
            ),
            showlegend=False,
            height=620,
            margin=dict(l=248, r=96, t=80, b=72),
        )
        hidden = " hidden" if t_pts != t_values[0] else ""
        panels.append(
            f'<div class="heatmap-panel" data-setting="{t_pts}"{hidden}>{html_fig}</div>'
        )
    options = "".join(
        f'<option value="{t_pts}"{" selected" if t_pts == t_values[0] else ""}>'
        f"{t_pts} points per snapshot</option>"
        for t_pts in t_values
    )
    return f"""
<div class="heatmap-explorer" id="revised-heatmap-explorer">
  <div class="joint-controls">
    <label>Points per snapshot
      <select id="revised-heatmap-setting">{options}</select>
    </label>
  </div>
  <p class="joint-help">Training snapshots per class stay at 60 and test snapshots per class stay at 15.
  Switch points per snapshot here. The folded table under the chart lists every process × model cell.</p>
  {''.join(panels)}
</div>
"""


def plot_revised_process_f1(frame: pd.DataFrame, fig_id: str, heading: str) -> str:
    if frame.empty:
        return "<p class='miss'>not generated</p>"
    t_values = sorted(int(v) for v in frame["t"].dropna().unique())
    t_colors = {29: NAVY, 58: STEEL, 88: GOLD}
    fig = go.Figure()
    legend_items = []
    for t_pts in t_values:
        color = t_colors.get(t_pts, _cloud_color(t_pts))
        label = f"{t_pts} points per snapshot"
        legend_items.append((label, color))
        fig.add_bar(
            name=label,
            x=[MODEL_LABEL[m] for m in MODELS],
            y=[_revised_f1(frame, m, t_pts) for m in MODELS],
            marker_color=color,
            hovertemplate=f"%{{x}}<br>{esc(label)} F1=%{{y:.3f}}<extra></extra>",
        )
    fig.update_layout(
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        yaxis_title="F1",
        yaxis=dict(range=[0, 1.08]),
    )
    return html_swatch_legend(legend_items) + fig_html(
        fig,
        fig_id,
        title=heading,
        showlegend=False,
        height=440,
        margin=dict(l=72, r=36, t=84, b=64),
    )


def revised_f1_grid(process_data: dict) -> str:
    t_values = _revised_t_values(process_data)
    if not t_values:
        return "<p class='miss'>not generated</p>"
    headers = ["Process"]
    for t_pts in t_values:
        for model in MODELS:
            headers.append(f"{MODEL_LABEL[model]} · {t_pts} pts")
    body = []
    for item in PROCESS_BRIEFING:
        frame = process_data[item["slug"]]["revised"]
        cells = [td(item["short"], kind="text")]
        for t_pts in t_values:
            for model in MODELS:
                cells.append(td(_revised_f1(frame, model, t_pts)))
        body.append(cells)
    return html_table(
        headers,
        body,
        caption="Every library-default model, every scored points-per-snapshot value. F1 on 60 / 15 barcode rows.",
    )


# ---------------------------------------------------------------------------
# Worked numbers
# ---------------------------------------------------------------------------
def historical_l5_formula(late_audit: pd.DataFrame, late_no_us_audit: pd.DataFrame) -> str:
    us = audit_l5_row(late_audit, "historical_l500", "class1")
    maj = audit_l5_row(late_no_us_audit, "historical_l500", "class2")
    mino = audit_l5_row(late_no_us_audit, "historical_l500", "class1")
    if us is None or maj is None or mino is None:
        return "<p class='miss'>Sampling-ratio audit artefact missing for the historical L5 substitution.</p>"
    n1 = int(us["minority_class_count"])
    n2 = int(maj["majority_class_count"])
    pts_min = int(mino["points_per_snapshot"])
    pts_maj = int(maj["points_per_snapshot"])
    n_snap = int(us["n_snapshots"])
    reuse = float(us["reuse_ratio"])
    percent = float(us["snapshot_size_percent_of_class"])
    floor_min = int(math.floor(n1 * percent / 100.0))
    floor_maj = int(math.floor(n2 * percent / 100.0))
    recomputed_reuse = (pts_min * n_snap) / n1
    return f"""
<p>Historical 500-snapshot clouds used a percent-of-class rule:</p>
<div class="formula">
points per snapshot = floor(class count × snapshot size percent / 100)
</div>
<p>After undersampling, both classes have the minority class count
<strong data-value="{n1}">{n1:,}</strong>. Snapshot size 5% is</p>
<div class="formula">
points per snapshot = floor({n1:,} × {percent:.0f} / 100) = {floor_min:,}
</div>
<p>The audit artefact stores <strong data-value="{pts_min}">{pts_min:,}</strong> points per snapshot
for that row. Both classes then send {pts_min:,}-point clouds into Ripser, so the H0 and H1 diagrams
are built from equal cloud sizes.</p>
<p>Without undersampling the majority class count stays
<strong data-value="{n2}">{n2:,}</strong>. The same 5% rule becomes</p>
<div class="formula">
points per snapshot (minority) = floor({n1:,} × {percent:.0f} / 100) = {floor_min:,}
<br>points per snapshot (majority) = floor({n2:,} × {percent:.0f} / 100) = {floor_maj:,}
</div>
<p>The audit artefact stores minority clouds of <strong data-value="{pts_min}">{pts_min:,}</strong>
points and majority clouds of <strong data-value="{pts_maj}">{pts_maj:,}</strong> points.
Ripser is then comparing a {pts_min:,}-point cloud to a {pts_maj:,}-point cloud.
Persistence counts and barcode summaries scale with cloud size, so a class difference in barcode
space can be larger-cloud versus smaller-cloud rather than default geometry versus non-default
geometry.</p>
<p>Reuse on the historical undersampled 5% run:</p>
<div class="formula">
reuse ratio = (points per snapshot × number of snapshots) / minority class count
<br>= ({pts_min:,} × {n_snap:,}) / {n1:,}
<br>= {recomputed_reuse:.12g}
</div>
<p>The stored reuse ratio is <strong data-value="{reuse:.12g}">{reuse:.3f}</strong>.
A typical minority customer is drawn about {reuse:.0f} times. The 500-snapshot library is not
500 independent people.</p>
"""


def two_nn_formula(bundle: dict) -> str:
    pkl = bundle.get("pkl") or {}
    suite = pkl.get("suite") or {}
    before = (suite.get("before_pca") or {}).get("handcoded_two_nn") or {}
    after = (suite.get("after_pca") or {}).get("handcoded_two_nn") or {}
    lb_before = (suite.get("before_pca") or {}).get("handcoded_levina_bickel") or {}
    lb_after = (suite.get("after_pca") or {}).get("handcoded_levina_bickel") or {}
    row = bundle["csv"].iloc[0] if not bundle["csv"].empty else pd.Series(dtype=float)
    d_before = before.get("intrinsic_dim_two_nn", row.get("two_nn_before_pca"))
    d_after = after.get("intrinsic_dim_two_nn", row.get("two_nn_after_pca"))
    mu_before = before.get("mean_mu")
    mu_after = after.get("mean_mu")
    n_before = before.get("n_points_used")
    n_after = after.get("n_points_used")
    mean_log_before = (1.0 / float(d_before)) if not is_missing(d_before) and float(d_before) else None
    mean_log_after = (1.0 / float(d_after)) if not is_missing(d_after) and float(d_after) else None
    k = lb_before.get("k", 10)
    pca_n = row.get("pca_components_exp3", suite.get("pca_components_used_in_TDA"))
    var = row.get("variance_retained_exp3_pca", suite.get("variance_retained_pca"))
    n90 = row.get("n_components_for_90pct")
    var90 = row.get("variance_at_90pct_n")
    mu_note_before = (
        f"<strong data-value='{float(mu_before):.12g}'>{float(mu_before):.3f}</strong>"
        if not is_missing(mu_before)
        else "<em>mean of the neighbour-distance ratio was not stored</em>"
    )
    mu_note_after = (
        f"<strong data-value='{float(mu_after):.12g}'>{float(mu_after):.3f}</strong>"
        if not is_missing(mu_after)
        else "<em>mean of the neighbour-distance ratio was not stored</em>"
    )
    return f"""
<p>Two-NN (Facco et al.), as implemented in <code>utils.estimate_intrinsic_dimension_two_nn</code>:</p>
<div class="formula">
For each point i,
neighbour-distance ratio = (distance to 2nd nearest neighbour) / (distance to nearest neighbour)
<br>ratios of 1 or less are dropped.
<br>estimated intrinsic dimension = 1 / mean of log(neighbour-distance ratio)
</div>
<p>The pickle stores the mean of the ratio itself as <code>mean_mu</code>, not the mean of the log.
The mean of the log is inverted from the stored dimension:</p>
<div class="formula">
Before PCA: estimated dimension = {fmt_display(d_before, 4)}
<br>so mean of log(ratio) = 1 / {fmt_display(d_before, 4)} = {fmt_display(mean_log_before, 4)}
<br>mean of the ratio (stored) = {mu_note_before}
<br>points used = {fmt_display(n_before, 0)}
<br><br>After the Exp 3 PCA: estimated dimension = {fmt_display(d_after, 4)}
<br>so mean of log(ratio) = 1 / {fmt_display(d_after, 4)} = {fmt_display(mean_log_after, 4)}
<br>mean of the ratio (stored) = {mu_note_after}
<br>points used = {fmt_display(n_after, 0)}
</div>
<p>Levina–Bickel uses neighbourhood size k = <strong data-value="{int(float(k))}">{int(float(k))}</strong>.
For each point, a local dimension is</p>
<div class="formula">
local dimension = (1 / (k − 1)) × sum of log(distance to the k-th neighbour / distance to the j-th neighbour)
for j = 1 … k − 1
<br>estimated dimension = mean of those local dimensions
</div>
<p>Stored Levina–Bickel: before PCA {fmt_display(lb_before.get('intrinsic_dim_levina_bickel', row.get('levina_bickel_before_pca')), 4)},
after PCA {fmt_display(lb_after.get('intrinsic_dim_levina_bickel', row.get('levina_bickel_after_pca')), 4)}.</p>
<p>PCA rank actually used in the historical TDA table:
<strong data-value="{int(float(pca_n))}">{int(float(pca_n))}</strong> components, variance retained
<strong data-value="{float(var):.12g}">{float(var):.3f}</strong>.
Smallest rank that keeps at least 90% of variance:
<strong data-value="{int(float(n90))}">{int(float(n90))}</strong> components at
<strong data-value="{float(var90):.12g}">{float(var90):.3f}</strong>
(<code>utils.n_components_for_target_variance</code>).</p>
"""


def permutation_formula() -> str:
    return """
<p>Class difference on barcode-statistic vectors uses the Robinson and Turner permutation
construction, implemented as a vector-summary proxy in
<code>utils.joint_loss_fpq_feature_vectors</code> and
<code>utils.permutation_test_of_class_difference</code>. The stored flag
<code>barcode_vector_proxy</code> is True on these runs.</p>
<div class="formula">
Within-group term for a class = (sum of Minkowski distances with exponent p, each raised to q,
excluding self-distances) / (2 × class count × (class count − 1))
<br>F<sub>p,q</sub> = within-group term (class 1) + within-group term (class 2)
</div>
<p>Labels of the two barcode-row groups are shuffled. Number of permutations B = 200
(the observed assignment is counted as one of the 200). The p-value is</p>
<div class="formula">
p-value = z / B
<br>z = 1 + (number of shuffled assignments whose F<sub>p,q</sub> is less than or equal to the observed F)
<br>smallest printable p-value = 1 / 200 = 0.005
</div>
<p>The 100 versus 100 counts in the stored results are barcode rows per class, not customers.
A p-value of 0.005 means every shuffled copy produced a larger F than the observed one,
at this resolution of B. The barcode rows are not independent customers, so the number
does not by itself establish a customer-level class difference.</p>
"""


def reuse_worked_for_process(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='miss'>not generated</p>"
    bits = []
    for percent, label in ((5.0, "5%"), (15.0, "15%")):
        for class_name, class_label in (("class1", "minority / class 1"), ("class2", "majority / class 2")):
            row = None
            subset = frame[
                (frame.get("l_rule") == "historical_l500")
                & (frame.get("snapshot_size_percent_of_class") == percent)
                & (frame.get("class") == class_name)
            ]
            if "split" in subset.columns:
                preferred = subset[subset["split"].isin(["full_table", "train"])]
                if not preferred.empty:
                    subset = preferred
            if not subset.empty:
                row = subset.iloc[0]
            if row is None:
                continue
            n_class = int(row["n_class"])
            pts = int(row["points_per_snapshot"])
            n_snap = int(row["n_snapshots"])
            reuse = float(row["reuse_ratio"])
            recomputed = (pts * n_snap) / n_class if n_class else float("nan")
            bits.append(
                f"<div class='formula'>{label}, {class_label}: "
                f"points per snapshot = {pts:,}, class count = {n_class:,}, "
                f"number of snapshots = {n_snap:,}<br>"
                f"reuse ratio = ({pts:,} × {n_snap:,}) / {n_class:,} = {recomputed:.12g} "
                f"(stored {reuse:.12g})</div>"
            )
    return "".join(bits) if bits else "<p class='miss'>not generated</p>"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def site_nav() -> str:
    process_opts = "".join(
        f'<option value="#{esc(item["slug"])}">{esc(item["short"])}</option>'
        for item in PROCESS_BRIEFING
    )
    revised_opts = "".join(
        f'<option value="#revised-{esc(item["slug"])}">60/15 — {esc(item["short"])}</option>'
        for item in PROCESS_BRIEFING
    )
    return f"""
<nav class="site-nav" id="site-nav" aria-label="Briefing sections">
  <div class="site-nav-inner">
    <div class="nav-row">
      <a class="nav-home" href="#top">TDA briefing</a>
      <ul class="nav-links">
        <li><a href="#tabular">Tabular</a></li>
        <li><a href="#eight">Eight processes</a></li>
        <li><a href="#statistics">Statistics</a></li>
        <li><a href="#sss">Snapshot size</a></li>
        <li><a href="#revised">Revised 60/15</a></li>
        <li><a href="#app-a">App. A</a></li>
        <li><a href="#app-b">App. B</a></li>
        <li><a href="#app-c">App. C</a></li>
      </ul>
    </div>
    <div class="nav-row">
      <label class="nav-jump"><span>Jump to</span>
        <select id="nav-jump-select">
          <option value="#top">Title</option>
          <option value="#briefing-covers">What this briefing covers</option>
          <option value="#tabular">Tabular baseline</option>
          <option value="#eight">Eight TDA processes</option>
          {process_opts}
          <option value="#statistics">Statistics: intrinsic dimension</option>
          <option value="#sss">Snapshot sample size</option>
          <option value="#sss-count">Effect of the number of snapshots</option>
          <option value="#sss-points">Effect of points per snapshot</option>
          <option value="#sss-joint">Joint grid</option>
          <option value="#revised">Revised snapshot protocol (60 / 15)</option>
          {revised_opts}
          <option value="#app-a">Appendix A — Tuned classifiers</option>
          <option value="#app-b">Appendix B — Sampling-ratio audit</option>
          <option value="#app-c">Appendix C — Permutation test</option>
        </select>
      </label>
    </div>
  </div>
</nav>
"""


def section_intro() -> str:
    return f"""
<header id="top">
  <p class="kicker">{esc(DATASET_LABEL)} · supervisor briefing</p>
  <h1>{esc(TITLE)}</h1>
  <p class="lede">Live artefacts only. Generated {date.today().isoformat()} from
  <code>6_Results/</code>. This briefing loads Default of Credit Card Client files and no other dataset.
  Use the bar at the top of the page to jump between sections.</p>
</header>
<section id="briefing-covers">
  <h2>What this briefing covers</h2>
  <p>The live research asks whether Vietoris–Rips barcode statistics, computed on snapshots of
  customers in the processed feature space, change default-prediction numbers relative to
  classifiers trained on the original processed table — and how three protocol knobs move those
  numbers: split timing (late vs early), undersampling vs no undersampling, and homology
  (H0 only vs H0 and H1).</p>
  <p>Two further programmes sit beside that grid. Statistics estimates intrinsic dimension on the
  processed table before PCA and after the Exp 3 PCA (7 components). Snapshot sample size holds
  homology at H0 and H1 and varies <em>number of snapshots</em> and <em>points per snapshot</em>
  on four split/undersample protocols, not on eight homology variants.</p>
  <p>The revised 60 / 15 snapshot protocol is a main result: the same eight processes, scored after
  the snapshot library is cut from 500 snapshots per class to 60 training and 15 test snapshots.
  Tuned barcode classifiers, the sampling-ratio audit, and the permutation test of class difference
  are grouped in appendices A–C. They are the same eight processes, not a ninth grid.</p>
</section>
"""


def section_tabular(rows: list[dict]) -> str:
    chart = plot_tabular(rows) if rows else "<p class='miss'>not generated</p>"
    best = best_f1(rows, "Original features")
    return f"""
<section id="tabular">
  <h2>Tabular baseline: library-default classifiers on the original processed table</h2>
  {path_line("6_Results", "Default_Parameters", "1_ML_Default_Parameters", DATASET, "model_results.pkl")}
  {six_part(
      "<p>What F1, accuracy, precision, and recall do library-default SVM, KNN, XGBoost, logistic regression, "
      "and random forest reach when they are trained on the processed customer table, with no barcode features.</p>",
      "<p>Experiment <code>Default_Parameters/1_ML_Default_Parameters</code>. One train/test cut on customers. "
      "Classifiers keep scikit-learn / XGBoost library defaults (no grid search). Metrics are held-out test metrics "
      "on customers, not barcode rows.</p>",
      "<p>No snapshot formula applies here. Each row is one customer. F1 is the binary F1 of the default class "
      "on that customer test set. These figures are the customer-level reference; they are not on the same "
      "sample unit as the TDA tables that follow.</p>",
      chart + model_table(rows, ("Original features",)),
      f"<p>Best F1 among the five library-default models is {fmt_display(best)}. "
      "The table is a processed tabular credit-default problem with the usual class imbalance of this dataset. "
      "Nothing topological has been added yet.</p>",
      "<p>This does not bound what a tuned tabular model could do, and it does not measure barcode-row "
      "separability. Appendix A retunes the barcode classifiers, not this table.</p>",
  )}
</section>
"""


def process_run_text(slug: str) -> str:
    spec = spec_of(slug)
    split = spec["split_timing"]
    undersample = bool(spec["undersample"])
    homology = spec["homology"]
    if split == "late":
        split_txt = (
            "MinMax and PCA were fitted on the full processed table. Snapshots were then drawn, "
            "Ripser produced barcode statistics, and the classifier used an 80/20 cut on <em>barcode rows</em>."
        )
    else:
        split_txt = (
            "Customers were cut 80/20 first. MinMax and PCA were fitted on the training customers only. "
            "Snapshots and Ripser were built inside that split. The classifier is evaluated on hold-out "
            "<em>customers'</em> snapshots, not on a random cut of barcode rows from the full table."
        )
    if undersample:
        us_txt = "Each class pool was undersampled to the minority class count before snapshots were drawn."
    else:
        us_txt = (
            "No undersampling. Each class keeps its own count, so the percent-of-class rule yields "
            "unequal cloud sizes (see the worked numbers)."
        )
    if homology == "H0":
        hom_txt = (
            "H0 only: this process does not run Ripser. It slices the H0 barcode-statistic columns "
            "from the matching H0-and-H1 tables, then trains the same five library-default classifiers "
            "on that 12-column slice."
        )
    else:
        hom_txt = (
            "H0 and H1: Ripser is run once (homology through H1). Each snapshot becomes 24 barcode "
            "statistics (12 H0 + 12 H1). Those rows are the classifier features."
        )
    return (
        f"<p>{split_txt} {us_txt} {hom_txt}</p>"
        "<p>Library-default SVM, KNN, XGBoost, logistic regression, and random forest. "
        "Historical snapshot sizes are 5% and 15% of the class (L5, L15). Number of snapshots per class "
        "in this experiment is 500.</p>"
    )


def process_why_text(slug: str, rows: list[dict], audit: pd.DataFrame) -> str:
    spec = spec_of(slug)
    l5_best = best_f1(rows, "L5")
    l15_best = best_f1(rows, "L15")
    bits = [
        f"<p>Best L5 F1 among the five models is {fmt_display(l5_best)}; best L15 F1 is {fmt_display(l15_best)}.</p>"
    ]
    if spec["split_timing"] == "late":
        bits.append(
            "<p>On a late split the scaler and PCA have already seen every customer. The 80/20 cut is on "
            "barcode rows that remix those customers. High F1 here is barcode-row separability after that "
            "geometry has been estimated on the full table, plus whatever reuse the 500-snapshot library "
            "introduces. It is not a customer hold-out score, and it is not comparable to the tabular "
            "baseline F1 without stating the change of sample unit.</p>"
        )
    else:
        bits.append(
            "<p>On an early split the test customers were not used to fit MinMax or PCA. F1 is still computed "
            "on barcode rows of snapshots drawn from those held-out customers, so the sample unit remains "
            "the snapshot, not the customer. The numbers typically sit below the late-split barcode-row F1 "
            "because the test geometry is no longer in the scaler/PCA fit.</p>"
        )
    if not spec["undersample"]:
        bits.append(
            "<p>Without undersampling, L5 clouds are unequal. Persistence summaries can separate classes "
            "because one cloud has more points, even if the underlying shapes were similar.</p>"
        )
    else:
        bits.append(
            "<p>Undersampling equalises cloud size at each percent, so L5 vs L15 is a change of points per "
            "snapshot (and of reuse) rather than a change of class balance in the cloud.</p>"
        )
    if spec["homology"] == "H0":
        bits.append(
            "<p>H0-only F1 is the same snapshots and the same split with the H1 columns removed. "
            "A gap versus H0 and H1 is a feature-slice gap, not evidence of a second persistence computation.</p>"
        )
    bits.append(reuse_worked_for_process(audit))
    return "".join(bits)


def process_limits_text(slug: str) -> str:
    spec = spec_of(slug)
    extra = []
    if spec["split_timing"] == "late":
        extra.append(
            "Near-1.00 F1 on this process is not evidence that TDA predicts default for unseen customers."
        )
    extra.append("These F1 values must not be averaged with Snapshot Sample Size F1 (different snapshot counts, different points per snapshot, different repeats).")
    extra.append("They must not be read as the tabular baseline on a different feature set: the rows are snapshots.")
    extra.append("The same combination scored on the 60 / 15 library is in the revised snapshot protocol section, not here.")
    return "<p>" + " ".join(extra) + "</p>"


def section_eight(process_data: dict) -> str:
    rows_map = {slug: process_data[slug]["default_rows"] for slug in process_data}
    heatmap = plot_eight_process_heatmap(rows_map)
    full_grid = fold(
        "Full F1 grid — all five models at L5 and L15",
        f1_grid_table(rows_map),
        n_rows=len(PROCESS_BRIEFING),
    )
    late_audit = process_data["Late_Split_And_Undersample_H0_And_H1"]["audit"]
    late_no = process_data["Late_Split_No_Undersample_H0_And_H1"]["audit"]
    blocks = []
    for item in PROCESS_BRIEFING:
        slug = item["slug"]
        rows = process_data[slug]["default_rows"]
        audit = process_data[slug]["audit"]
        rel = process_data[slug]["default_rel"]
        settings = tuple(dict.fromkeys(r["setting"] for r in rows)) or ("L5", "L15")
        blocks.append(
            f"""
<section id="{esc(slug)}">
  <h2>{esc(item['heading'])}</h2>
  {path_line(*rel.split('/'))}
  {six_part(
      "<p>Library-default classifiers on barcode statistics for this combination of split timing, "
      "undersampling, and homology.</p>",
      process_run_text(slug),
      historical_l5_formula(late_audit, late_no) if slug.startswith("Late_")
      else "<p>The percent-of-class and reuse substitutions for this process use its own sampling-ratio audit "
           "(train-pool class counts after the early split). They are not the late-split 6,630-row pool.</p>"
           + reuse_worked_for_process(audit),
      plot_process_f1(
          rows,
          f"fig-proc-{slug}",
          f"Held-out F1 by model — {item['short']}",
      )
      + fold(
          "Full held-out metrics for all five models (L5 and L15)",
          model_table(rows, settings),
          n_rows=len(rows),
      ),
      process_why_text(slug, rows, audit),
      process_limits_text(slug),
  )}
</section>
"""
        )
    return f"""
<section id="eight">
  <h2>Eight TDA processes: split timing × undersampling × homology (H0 vs H0 and H1)</h2>
  <p>The eight live processes are one 2 × 2 × 2 design. H0 processes share Ripser output with the
  matching H0-and-H1 folder; they are a column slice, not a second Ripser run. Folder names sit under
  each heading, not in the heading. Each combination below has its own grouped-bar chart of all five
  models at L5 and L15.</p>
  {six_part(
      "<p>How F1 on barcode statistics moves when split timing, undersampling, and homology (H0 vs H0 and H1) "
      "are crossed, at library-default classifiers and the historical 5% / 15% snapshot sizes.</p>",
      "<p>Five library-default classifiers on each process. L5 and L15. 500 snapshots per class. "
      "Default of Credit Card Client only. Seeds and PCA rank follow the process scripts under "
      "<code>5_Experiments/{process}/1_PH_Default_Parameters/</code>.</p>",
      historical_l5_formula(late_audit, late_no),
      heatmap
      + full_grid
      + "<p>The heatmap is every process × model cell at one snapshot size. Open a process heading below "
      "for that combination’s grouped bars and the full accuracy, precision, recall, and F1 table.</p>",
      "<p>Late-split F1 can sit near 1.00 while early-split F1 does not, because the late pipeline "
      "fits MinMax and PCA on the full processed table and then splits barcode rows that remix those "
      "customers. No-undersample F1 can also move because L5 clouds on the late full table have unequal "
      "point counts (see the worked substitution above), not because homology changed. H0 vs H0 and H1 "
      "is a 12-column vs 24-column feature slice of the same barcodes.</p>",
      "<p>The grid does not identify a production scoring rule. It does not put tabular customer F1 "
      "and barcode-row F1 on one scale. It does not include tuned models (Appendix A), reuse diagnostics "
      "(Appendix B), a permutation test (Appendix C), or the 60 / 15 protocol in the revised snapshot section.</p>",
  )}
</section>
{''.join(blocks)}
"""


def section_statistics(bundle: dict) -> str:
    row = bundle["csv"].iloc[0] if not bundle["csv"].empty else pd.Series(dtype=float)
    pkl = bundle.get("pkl") or {}
    suite = pkl.get("suite") or {}
    before = (suite.get("before_pca") or {}).get("handcoded_two_nn") or {}
    after = (suite.get("after_pca") or {}).get("handcoded_two_nn") or {}
    chart = plot_id(bundle)
    headers = ["Estimator", "Before PCA", "After Exp 3 PCA (7 components)"]
    body = [
        [
            td("Two-NN (hand-coded)", kind="text"),
            td(before.get("intrinsic_dim_two_nn", row.get("two_nn_before_pca")), 4),
            td(after.get("intrinsic_dim_two_nn", row.get("two_nn_after_pca")), 4),
        ],
        [
            td("mean of neighbour-distance ratio (stored mean_mu)", kind="text"),
            td(before.get("mean_mu"), 4),
            td(after.get("mean_mu"), 4),
        ],
        [
            td("Levina–Bickel (k = 10)", kind="text"),
            td(
                (suite.get("before_pca") or {}).get("handcoded_levina_bickel", {}).get(
                    "intrinsic_dim_levina_bickel", row.get("levina_bickel_before_pca")
                ),
                4,
            ),
            td(
                (suite.get("after_pca") or {}).get("handcoded_levina_bickel", {}).get(
                    "intrinsic_dim_levina_bickel", row.get("levina_bickel_after_pca")
                ),
                4,
            ),
        ],
        [
            td("skdim TwoNN", kind="text"),
            td(row.get("skdim_TwoNN_before_pca"), 4),
            td(row.get("skdim_TwoNN_after_pca"), 4),
        ],
        [
            td("skdim MLE", kind="text"),
            td(row.get("skdim_MLE_before_pca"), 4),
            td(row.get("skdim_MLE_after_pca"), 4),
        ],
        [
            td("skdim MiND_ML", kind="text"),
            td(row.get("skdim_MiND_ML_before_pca"), 4),
            td(row.get("skdim_MiND_ML_after_pca"), 4),
        ],
        [
            td("skdim lPCA", kind="text"),
            td(row.get("skdim_lPCA_before_pca"), 4),
            td(row.get("skdim_lPCA_after_pca"), 4),
        ],
    ]
    extra = html_table(
        ["Quantity", "Value"],
        [
            [td("Processed feature count", kind="text"), td(row.get("n_features"), kind="int")],
            [td("PCA components used in historical TDA (Exp 3)", kind="text"), td(row.get("pca_components_exp3"), kind="int")],
            [td("Variance retained at those 7 components", kind="text"), td(row.get("variance_retained_exp3_pca"), 4)],
            [td("Smallest rank for 90% variance", kind="text"), td(row.get("n_components_for_90pct"), kind="int")],
            [td("Variance at that 90% rank", kind="text"), td(row.get("variance_at_90pct_n"), 4)],
        ],
    )
    return f"""
<section id="statistics">
  <h2>Statistics: intrinsic dimension before PCA and after the Exp 3 PCA (7 components)</h2>
  {path_line(bundle['csv_rel'])}
  {path_line(bundle['pkl_rel'])}
  {six_part(
      "<p>What intrinsic dimension do Two-NN, Levina–Bickel, and the skdim suite assign to the processed "
      "Default of Credit Card Client table, before PCA and after the 7-component PCA actually used in the "
      "historical TDA pipeline.</p>",
      "<p>Protocol-independent. Experiment <code>Statistics/1_Intrinsic_Dimension_Estimation</code>. "
      "No snapshots and no classifiers. Hand-coded Two-NN and Levina–Bickel (k = 10) plus skdim TwoNN, "
      "MLE, MiND_ML, and lPCA. The same estimators are applied to the scaled table and to the Exp 3 PCA image.</p>",
      two_nn_formula(bundle),
      chart
      + fold("Intrinsic-dimension estimators (all stored values)", html_table(headers, body) + extra),
      "<p>Two-NN falls from the mid-3s before PCA to the high-2s after 7 components. That is the dimension "
      "of the PCA image, not a claim that default risk lives on a 2-dimensional manifold of customers. "
      "Levina–Bickel on this table is far below Two-NN; the two estimators are not interchangeable on these "
      "points. lPCA returning 6 is the linear rank near the 90% cutoff, not a persistence result.</p>",
      "<p>These estimates are not F1, not a snapshot-size recommendation by themselves, and not a test that "
      "H1 carries extra default signal. The revised snapshot protocol uses an intrinsic-dimension value inside "
      "a snapshot-count formula; that is a different experiment.</p>",
  )}
</section>
"""


def sss_summary_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='miss'>not generated</p>"
    headers = [
        "Protocol",
        "Model",
        "Points per snapshot",
        "Number of snapshots",
        "Reuse ratio",
        "F1 mean",
        "F1 95% CI low",
        "F1 95% CI high",
        "Repeats",
    ]
    model_rank = {model: idx for idx, model in enumerate(MODELS)}
    work = frame.copy()
    work["_model_rank"] = work["model"].map(lambda m: model_rank.get(m, 99))
    work["_protocol_rank"] = work["protocol"].map(
        lambda p: SSS_PROTOCOL_ORDER.index(p) if p in SSS_PROTOCOL_ORDER else 99
    )
    work = work.sort_values(
        ["_protocol_rank", "_model_rank", "points_per_snapshot", "n_snapshots"]
    )
    body = []
    for _, row in work.iterrows():
        body.append(
            [
                td(_sss_protocol_label(row["protocol"]), kind="text"),
                td(MODEL_LABEL.get(str(row["model"]), row["model"]), kind="text"),
                td(row["points_per_snapshot"], kind="int"),
                td(row["n_snapshots"], kind="int"),
                td(row["reuse_ratio"]),
                td(row["f1_mean"]),
                td(row["f1_ci95_low"]),
                td(row["f1_ci95_high"]),
                td(row["n_repeats"], kind="int"),
            ]
        )
    return html_table(
        headers,
        body,
        caption="All five models × four protocols. F1 mean and 95% interval over the stored repeats.",
    )


def section_sss(count_df: pd.DataFrame, points_df: pd.DataFrame, joint_df: pd.DataFrame) -> str:
    default_pts = (
        int(count_df["points_per_snapshot"].dropna().unique()[0])
        if not count_df.empty and count_df["points_per_snapshot"].nunique()
        else None
    )
    n_snap_points = (
        int(points_df["n_snapshots"].dropna().unique()[0])
        if not points_df.empty and points_df["n_snapshots"].nunique()
        else N_TRAIN_POOL
    )
    held_pts = default_pts if default_pts is not None else 330
    count_heading = (
        f"Effect of the number of snapshots, with points per snapshot held fixed at {held_pts}"
    )
    points_heading = (
        f"Effect of points per snapshot, with the number of snapshots held at {n_snap_points}"
    )
    count_vals = ", ".join(str(int(v)) for v in sorted(count_df["n_snapshots"].dropna().unique())) if not count_df.empty else "not generated"
    pts_vals = ", ".join(str(int(v)) for v in sorted(points_df["points_per_snapshot"].dropna().unique())) if not points_df.empty else "not generated"
    reuse_ex = ""
    if not count_df.empty:
        sample = count_df.iloc[0]
        n1 = int(sample["minority_class_count"])
        pts = int(sample["points_per_snapshot"])
        n_snap = int(sample["n_snapshots"])
        reuse = float(sample["reuse_ratio"])
        recomputed = (pts * n_snap) / n1
        reuse_ex = f"""
<div class="formula">
reuse ratio = (points per snapshot × number of snapshots) / minority class count
<br>First row of the count sweep: ({pts:,} × {n_snap:,}) / {n1:,} = {recomputed:.12g}
<br>stored reuse ratio = {reuse:.12g}
</div>
<p>The count sweep holds points per snapshot at {held_pts}. That is the study default
(the historical L5 floor on the full table is 331; this study locked 330). Number of snapshots
moves through {count_vals}. Each F1 mean is over {int(sample['n_repeats'])} repeats.</p>
"""
    points_formula = ""
    if not points_df.empty:
        sample = points_df.iloc[0]
        unique_n = sorted(points_df["n_snapshots"].dropna().unique())
        assert_note = (
            f"Every row in this artefact has number of snapshots = {int(unique_n[0])}."
            if len(unique_n) == 1
            else f"Unexpected snapshot counts in the artefact: {unique_n}."
        )
        n1 = int(sample["minority_class_count"])
        pts = int(sample["points_per_snapshot"])
        n_snap = int(sample["n_snapshots"])
        recomputed = (pts * n_snap) / n1
        points_formula = f"""
<p>{esc(assert_note)} That 180 is <code>utils.N_TRAIN_POOL</code>, equal to the maximum of
<code>utils.N_SNAPSHOTS_GRID</code> {tuple(int(x) for x in N_SNAPSHOTS_GRID)}. It is the experiment’s
control, not a heading invented for this briefing. Candidate points per snapshot in code:
{tuple(int(x) for x in CANDIDATE_POINTS_PER_SNAPSHOT)}. Values present in the artefact: {pts_vals}.</p>
<div class="formula">
reuse ratio = (points per snapshot × number of snapshots) / minority class count
<br>First row: ({pts:,} × {n_snap:,}) / {n1:,} = {recomputed:.12g}
<br>stored reuse ratio = {float(sample['reuse_ratio']):.12g}
</div>
"""
    joint_formula = ""
    if not joint_df.empty:
        n_pts = sorted(joint_df["points_per_snapshot"].dropna().unique())
        n_snaps = sorted(joint_df["n_snapshots"].dropna().unique())
        sample = joint_df.iloc[0]
        joint_formula = f"""
<p>Both knobs move together in this study. The heatmap x-axis is number of snapshots
({', '.join(str(int(v)) for v in n_snaps)}). The heatmap y-axis is points per snapshot
({', '.join(str(int(v)) for v in n_pts)}), with the smallest cloud at the bottom.
Repeats per cell: {int(sample['n_repeats'])}. The reuse formula is unchanged:</p>
<div class="formula">
reuse ratio = (points per snapshot × number of snapshots) / minority class count
</div>
<p>Where reuse exceeds 1, those cells remix the
same training customers. They are not additional people.</p>
"""
    return f"""
<section id="sss">
  <h2>Snapshot sample size</h2>
  <p>Four protocols (late/early × undersample/no undersample), always the H0-and-H1
  barcode construction. These F1 means are not the historical 500-snapshot process F1
  and must not be averaged with them. Test snapshot count is 15. Training prefixes are nested inside
  a pool of {N_TRAIN_POOL} training snapshots. The three studies below hold one knob, then the other,
  then both.</p>
</section>

<section id="sss-count">
  <h2>{esc(count_heading)}</h2>
  {path_line("6_Results", "Snapshot_Sample_Size", "1_Snapshot_Count_Sweep", "all_summary.csv")}
  {six_part(
      "<p>How barcode-classifier F1 moves as the number of training snapshots grows, while every cloud "
      f"keeps the study’s default point count ({held_pts}).</p>",
      f"<p>Protocols: late split and undersample; early split and undersample; late split, no undersample; "
      f"early split, no undersample. Models: SVM, logistic regression, KNN, XGBoost, random forest. "
      f"Number of snapshots ∈ {{{count_vals}}}. Points per snapshot fixed at {held_pts}. "
      f"{int(count_df['n_repeats'].iloc[0]) if not count_df.empty else 10} repeats. Nested prefixes from "
      f"one shuffled pool of {N_TRAIN_POOL} training snapshots.</p>",
      reuse_ex,
      sss_line_explorer(
          count_df,
          "n_snapshots",
          "Number of snapshots",
          "sss-count-explorer",
      )
      + fold(
          "Full snapshot-count sweep table — all five models and four protocols",
          sss_summary_table(count_df),
          n_rows=len(count_df),
      ),
      "<p>Curves that rise then flatten are extra snapshots of the same point count. Variation that stays "
      "wide at 180 snapshots is repeat-to-repeat snapshot noise at this cloud size, not a second dataset. "
      "Late versus early still differs because of when MinMax and PCA are fitted. No-undersample protocols "
      "still draw unequal class clouds if the shared pool used percent-derived sizes; on this study the "
      "point count is locked, so both classes use the same points per snapshot.</p>",
      "<p>This sweep does not choose an optimal production snapshot count. It does not re-run H0-only. "
      "It does not restore the historical 500-snapshot library.</p>",
  )}
</section>

<section id="sss-points">
  <h2>{esc(points_heading)}</h2>
  {path_line("6_Results", "Snapshot_Sample_Size", "2_Points_Per_Snapshot_Sweep", "all_summary.csv")}
  {six_part(
      "<p>How barcode-classifier F1 moves as each cloud gains or loses points, while the training library "
      f"stays at {n_snap_points} snapshots.</p>",
      f"<p>Same four protocols and five models. Number of snapshots held at {n_snap_points}. "
      f"Points per snapshot ∈ {{{pts_vals}}}. Repeats as stored in the artefact. "
      "180 is the study control <code>N_TRAIN_POOL</code>.</p>",
      points_formula,
      sss_line_explorer(
          points_df,
          "points_per_snapshot",
          "Points per snapshot",
          "sss-points-explorer",
      )
      + fold(
          "Full points-per-snapshot sweep table — all five models and four protocols",
          sss_summary_table(points_df),
          n_rows=len(points_df),
      ),
      "<p>Small clouds (15–45 points) are sparse Vietoris–Rips inputs; F1 is typically lower and more "
      "variable. Moving toward 330 adds persistence structure and also raises reuse for a fixed 180 "
      "snapshots. The chart’s x-axis is points per snapshot, not percent of class.</p>",
      "<p>Holding 180 snapshots does not claim 180 is optimal. It isolates the cloud-size factor. "
      "Do not compare a 15-point, 180-snapshot F1 to a historical L5 331-point, 500-snapshot F1 as if "
      "only one knob changed.</p>",
  )}
</section>

<section id="sss-joint">
  <h2>Joint grid: number of snapshots × points per snapshot</h2>
  {path_line("6_Results", "Snapshot_Sample_Size", "3_Snapshot_Count_Across_Cloud_Sizes", "all_summary.csv")}
  {six_part(
        "<p>How F1 moves as the number of snapshots and the points per snapshot both grow, "
        "on the rectangle of those two knobs rather than on either axis alone.</p>",
        "<p>Same four protocols, five models, nested snapshot prefixes, and ten repeats as the other two "
        "sweeps. Folder <code>3_Snapshot_Count_Across_Cloud_Sizes</code> is this joint grid.</p>",
        joint_formula,
        joint_grid_explorer(joint_df)
        + fold(
            "Full joint-grid table — all five models, four protocols, every cloud size and snapshot count",
            sss_summary_table(joint_df),
            n_rows=len(joint_df),
        ),
        "<p>The 3D surface is F1 as height over the two knobs. Drag it: a steep climb along points per "
        "snapshot is extra structure in each cloud; a climb along number of snapshots is extra barcode rows "
        "at the same cloud size. A plateau is F1 that has already stopped moving. The heatmap under the "
        "surface is the same cells, so a darkening toward the top-right matches a surface that lifts toward "
        "large clouds and large snapshot counts. The two axes are not interchangeable: 30 snapshots of "
        "180-point clouds are not 180 snapshots of 30-point clouds. A high ridge at large cloud size and "
        "large snapshot count can still be extra reuse of the same people.</p>",
      "<p>The joint grid is the two factors above crossed. It still uses the four-protocol H0-and-H1 "
      "sample-size construction, not the eight-process homology split. The next section scores a 60 / 15 "
      "library on all eight processes instead of sweeping snapshot count.</p>",
  )}
</section>
"""


def appendix_a(process_data: dict) -> str:
    tuned_map = {item["slug"]: process_data[item["slug"]]["tuned_rows"] for item in PROCESS_BRIEFING}
    tables = []
    for item in PROCESS_BRIEFING:
        rows = process_data[item["slug"]]["tuned_rows"]
        rel = process_data[item["slug"]]["tuned_rel"]
        settings = tuple(dict.fromkeys(r["setting"] for r in rows)) or ("L5", "L15")
        tables.append(
            fold(
                item["heading"].replace("default-parameter", "tuned"),
                path_line(*rel.split("/")) + model_table(rows, settings),
                n_rows=len(rows),
            )
        )
    return f"""
<section id="app-a">
  <h2>Appendix A — Tuned classifiers on barcode statistics</h2>
  {six_part(
      "<p>What happens to the eight-process barcode F1 when the same five classifiers are retuned "
      "(grid search on each process’s barcode tables), still at historical L5 / L15 and 500 snapshots.</p>",
      "<p>Experiment <code>2_PH_Tuned_Parameters</code> inside each of the eight process folders. "
      "H0 processes still slice columns from the matching H0-and-H1 barcodes; they do not retune Ripser.</p>",
      "<p>No new snapshot formula. The percent-of-class and reuse substitutions are those of Appendix B "
      "and the eight-process section. Tuning changes classifier hyperparameters, not points per snapshot.</p>",
      fold(
          "Full tuned F1 grid — all five models at L5 and L15",
          f1_grid_table(tuned_map),
          n_rows=len(PROCESS_BRIEFING),
      )
      + "<p>Each process below folds accuracy, precision, recall, and F1 for all five models.</p>"
      + "".join(tables),
      "<p>A tuned F1 above the library-default F1 on the same process is a classifier-capacity change on "
      "the same barcode rows. It is not additional homology and not a cleaner split. Late-split tuned F1 "
      "can remain near 1.00 for the same full-table PCA and reuse reasons as the default-parameter tables.</p>",
      "<p>Tuning does not convert barcode-row F1 into customer-level F1. It does not undo a late split. "
      "It is not the Snapshot Sample Size study and it is not the 60 / 15 protocol.</p>",
  )}
</section>
"""


def appendix_b(process_data: dict) -> str:
    late = process_data["Late_Split_And_Undersample_H0_And_H1"]["audit"]
    late_no = process_data["Late_Split_No_Undersample_H0_And_H1"]["audit"]
    tables = []
    for item in PROCESS_BRIEFING:
        frame = process_data[item["slug"]]["audit"]
        rel = process_data[item["slug"]]["audit_rel"]
        if frame.empty:
            tables.append(fold(item["short"], "<p class='miss'>not generated</p>"))
            continue
        show = frame.copy()
        keep = [
            c
            for c in (
                "split",
                "class",
                "l_rule",
                "snapshot_size_percent_of_class",
                "minority_class_count",
                "majority_class_count",
                "n_class",
                "points_per_snapshot",
                "n_snapshots",
                "reuse_ratio",
                "undersample",
            )
            if c in show.columns
        ]
        show = show[keep].drop_duplicates()
        headers = [c.replace("_", " ") for c in keep]
        body = []
        for _, row in show.iterrows():
            cells = []
            for col in keep:
                if col in {"minority_class_count", "majority_class_count", "n_class", "points_per_snapshot", "n_snapshots"}:
                    cells.append(td(row[col], kind="int"))
                elif col in {"reuse_ratio", "snapshot_size_percent_of_class"}:
                    cells.append(td(row[col], 3 if col == "reuse_ratio" else 1))
                else:
                    cells.append(td(row[col], kind="text"))
            body.append(cells)
        tables.append(
            fold(item["short"], path_line(*rel.split("/")) + html_table(headers, body), n_rows=len(body))
        )
    return f"""
<section id="app-b">
  <h2>Appendix B — Sampling-ratio audit</h2>
  {six_part(
      "<p>With the historical percent-of-class rule and 500 snapshots, how many times is a typical customer "
      "redrawn, and what snapshot count would bring reuse down to about 1 at the same points per snapshot.</p>",
      "<p>Experiment <code>6_Sampling_Ratio_Audit</code> on all eight processes. No classifier is trained. "
      "Rows compare the historical 500-snapshot rule with <code>revised_ceil_n_over_t</code> "
      "(ceiling of class count / points per snapshot). Early-split audits use the train pool after the "
      "customer cut; late-split audits use the full processed table.</p>",
      historical_l5_formula(late, late_no)
      + "<p>Early-split class counts are smaller because the customer split happens first. Substitutions "
      "for each process are in the tables below, not the 6,630-row late pool.</p>",
      "".join(tables),
      "<p>Reuse near 25 at L5 and near 75 at L15 on 500 snapshots means the barcode table is a thick remix "
      "of the same people. Ceiling-based snapshot counts (21 at late L5, 7 at late L15 on the undersampled "
      "full table) sit near reuse 1 and are the audit’s comparator, not the experiment that was scored in "
      "the eight-process F1 tables.</p>",
      "<p>The audit does not score F1. A reuse ratio below 1 does not imply that persistence features will "
      "separate defaults. Unequal no-undersample clouds remain unequal even when reuse is reduced.</p>",
  )}
</section>
"""


def appendix_c(process_data: dict) -> str:
    tables = []
    for item in PROCESS_BRIEFING:
        frame = process_data[item["slug"]]["perm"]
        rel = process_data[item["slug"]]["perm_rel"]
        if frame.empty:
            tables.append(fold(item["short"], "<p class='miss'>not generated</p>"))
            continue
        keep = [
            c
            for c in ("source", "p", "q", "observed_F_pq", "p_value", "n1", "n2", "null_mean", "barcode_vector_proxy")
            if c in frame.columns
        ]
        headers = [c.replace("_", " ") for c in keep]
        body = []
        for _, row in frame.iterrows():
            cells = []
            for col in keep:
                if col in {"source", "barcode_vector_proxy"}:
                    cells.append(td(row[col], kind="text"))
                elif col in {"p", "q", "n1", "n2"}:
                    cells.append(td(row[col], kind="int" if col in {"n1", "n2"} else "num"))
                else:
                    cells.append(td(row[col], 4 if col == "observed_F_pq" else 3))
            body.append(cells)
        tables.append(
            fold(item["short"], path_line(*rel.split("/")) + html_table(headers, body), n_rows=len(body))
        )
    return f"""
<section id="app-c">
  <h2>Appendix C — Permutation test of class difference</h2>
  {six_part(
      "<p>Whether the two classes’ barcode-statistic vectors differ by the Robinson–Turner F<sub>p,q</sub> "
      "statistic, using a permutation test on those vectors.</p>",
      "<p>Experiment <code>8_Permutation_Test_Of_Class_Difference</code> on all eight processes. "
      "Pairs (p, q) ∈ {{(2,2), (1,1), (2,1)}}. B = 200 permutations, random_state = 42. "
      "Inputs are barcode-statistic rows (n1 = n2 = 100 in the stored results), not raw customers. "
      "H0 processes permute the H0 column slice.</p>",
      permutation_formula(),
      "".join(tables),
      "<p>Many rows print p-value 0.005 because that is 1/B. The observed F is smaller than every shuffled F "
      "at this resolution. That is a statement about barcode-vector geometry on heavily reused snapshots, "
      "with a vector-summary proxy rather than full persistence-diagram distances.</p>",
      "<p>p = 0.005 is not a star rating and is not a customer-level discovery. Dependence across barcode rows "
      "(reuse, late PCA, unequal clouds) remains. The test does not rank the five classifiers and does not "
      "replace F1.</p>",
  )}
</section>
"""


def _revised_process_table(frame: pd.DataFrame) -> str:
    headers = [
        "Points per snapshot",
        "Train snapshots / class",
        "Test snapshots / class",
        "Model",
        "Accuracy",
        "Balanced accuracy",
        "F1",
        "Train reuse (default class)",
    ]
    body = []
    work = frame.copy()
    if "t" in work.columns:
        model_rank = {model: idx for idx, model in enumerate(MODELS)}
        work["_model_rank"] = work["model"].map(lambda m: model_rank.get(str(m), 99))
        work = work.sort_values(["t", "_model_rank"])
    for _, row in work.iterrows():
        body.append(
            [
                td(row.get("t"), kind="int"),
                td(row.get("train_l"), kind="int"),
                td(row.get("test_l"), kind="int"),
                td(MODEL_LABEL.get(str(row.get("model")), row.get("model")), kind="text"),
                td(row.get("accuracy")),
                td(row.get("balanced_accuracy")),
                td(row.get("f1")),
                td(row.get("reuse_train_default")),
            ]
        )
    return html_table(headers, body)


def section_revised(process_data: dict) -> str:
    t_vals = _revised_t_values(process_data)
    t_txt = ", ".join(str(v) for v in t_vals) if t_vals else "not generated"
    sample_formula = ""
    blocks = []
    for item in PROCESS_BRIEFING:
        frame = process_data[item["slug"]]["revised"]
        rel = process_data[item["slug"]]["revised_rel"]
        if frame.empty:
            blocks.append(
                f"""
<section id="revised-{esc(item['slug'])}">
  <h2>{esc(item['short'])} — 60 training / 15 test snapshots</h2>
  <p class='miss'>not generated</p>
</section>
"""
            )
            continue
        if not sample_formula:
            row = frame.iloc[0]
            t_pts = int(row["t"])
            train_l = int(row["train_l"])
            test_l = int(float(row["test_l"]))
            b_used = float(row["b_used"])
            formula_l = float(row["formula_l"])
            n_train = int(row["n_train_snapshots"])
            n_test = int(row["n_test_snapshots"])
            reuse = float(row.get("reuse_train_default")) if not is_missing(row.get("reuse_train_default")) else float("nan")
            log_t = math.log(t_pts)
            recomputed = (t_pts / log_t) ** (2.0 / b_used)
            sample_formula = f"""
<p>The scored protocol is 60 training snapshots and 15 test snapshots per class, at a fixed
points-per-snapshot value. First stored row of {esc(item['short'])}:</p>
<div class="formula">
points per snapshot = {t_pts:,}
<br>training snapshots per class = {train_l:,}
<br>test snapshots per class = {test_l:,}
<br>barcode rows used to train = {n_train:,} (both classes)
<br>barcode rows used to test = {n_test:,}
<br>train reuse (default class) = {reuse:.12g}
</div>
<p>A separate formula count from <code>utils.formula_l_from_t_b</code> (natural log) is stored
and is <em>not</em> the 60 / 15 schedule that was scored:</p>
<div class="formula">
suggested number of snapshots ≈ (points per snapshot / log(points per snapshot))<sup>(2 / intrinsic dimension)</sup>
<br>= ({t_pts:,} / log({t_pts:,}))<sup>(2 / {b_used:.12g})</sup>
<br>= ({t_pts:,} / {log_t:.12g})<sup>{(2.0 / b_used):.12g}</sup>
<br>= {recomputed:.12g}
</div>
<p>Stored formula count = {formula_l:.12g}. Intrinsic dimension used in that formula = {b_used:.12g}.
The experiment that was scored remains 60 / 15, not this suggested count.</p>
"""
        local_t = ", ".join(str(int(v)) for v in sorted(frame["t"].dropna().unique()))
        blocks.append(
            f"""
<section id="revised-{esc(item['slug'])}">
  <h2>{esc(item['short'])} — 60 training / 15 test snapshots</h2>
  {path_line(*rel.split('/'))}
  <p>The 60 / 15 library on this combination of split timing, undersampling, and homology.
  Bars are F1 at each scored points-per-snapshot value for this process. The 500-snapshot L5 / L15
  chart for the same combination is in the eight-process section.</p>
  {plot_revised_process_f1(
      frame,
      f"fig-revised-{item['slug']}",
      f"F1 by model at {local_t} points per snapshot — {item['short']}",
  )}
  {fold(
      "Full 60 / 15 metrics for all five models",
      _revised_process_table(frame),
      n_rows=len(frame),
  )}
</section>
"""
        )
    heatmap = plot_revised_heatmap(process_data)
    full_grid = fold(
        "Full 60 / 15 F1 grid — all five models at every scored cloud size",
        revised_f1_grid(process_data),
        n_rows=len(PROCESS_BRIEFING),
    )
    return f"""
<section id="revised">
  <h2>Revised snapshot protocol: 60 training / 15 test snapshots</h2>
  <p>The eight-process F1 tables used 500 snapshots per class. The sampling-ratio audit shows that
  this library remixes the same people many times. This section is the reduced-library result on the
  same eight combinations of split timing, undersampling, and homology: 60 training snapshots and
  15 test snapshots per class, at fixed points per snapshot rather than a percent-of-class rule.</p>
  {six_part(
      "<p>What classifier metrics look like when the snapshot library is 60 training and 15 test "
      "snapshots per class, at fixed points per snapshot, instead of the historical 500-snapshot "
      "percent-of-class library.</p>",
      f"<p>Experiment <code>9_Revised_Snapshot_Protocol</code> on all eight processes. The scored mode is "
      f"<code>default_60_15</code>. Five library-default classifiers. Points per snapshot present in the "
      f"artefacts: {t_txt} (not every process scored every cloud size). Early vs late and undersample vs not "
      "still follow the process folder. H0 processes still slice the H0 columns from the matching H0-and-H1 barcodes.</p>",
      sample_formula,
      heatmap
      + full_grid
      + f"<p>The heatmap reports F1 for every process and every model at one points-per-snapshot value at a time. "
      f"Each process heading below has its own grouped-bar chart across {t_txt} points per snapshot, "
      "and a folded table with accuracy, balanced accuracy, F1, and train reuse.</p>",
      "<p>F1 here is computed on far fewer barcode rows (30 test rows when both classes contribute 15). "
      "A swing of a few mistakes moves F1 by tenths. Train reuse is well below the historical "
      "24.96 figure because 60 snapshots replace 500. These scores are not the eight-process 500-snapshot F1 "
      "and are not the Snapshot Sample Size means.</p>",
      "<p>60 / 15 is the protocol that was scored, not the formula count. The formula count is a suggested "
      "library size from points per snapshot and an intrinsic-dimension estimate; it was not the train/test "
      "schedule. This section does not claim a best production snapshot rule. Tuned classifiers, the "
      "sampling-ratio audit, and the permutation test follow in appendices A–C.</p>",
  )}
</section>
{''.join(blocks)}
"""


CSS = f"""
:root {{
  --navy: {NAVY};
  --steel: {STEEL};
  --gold: {GOLD};
  --ink: {INK};
  --paper: #f4f1ea;
  --card: #ffffff;
  --line: #d9d4c8;
  --miss: #8B3A3A;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; scroll-padding-top: 7.2rem; }}
body {{
  margin: 0;
  max-width: none;
  padding: 0;
  background: var(--paper);
  color: var(--ink);
  font: 17px/1.55 Georgia, "Times New Roman", serif;
}}
.page {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 20px 28px 80px;
  position: relative;
  z-index: 0;
}}
.site-nav {{
  position: sticky;
  top: 0;
  z-index: 2147483647;
  background: {NAVY};
  color: #fff;
  border-bottom: 1px solid {NAVY};
  box-shadow: 0 2px 10px rgba(27,54,93,0.18);
  max-width: none;
  margin: 0;
  padding: 0;
}}
.site-nav-inner {{
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-width: 1180px;
  margin: 0 auto;
  padding: 8px 16px 10px;
}}
.nav-row {{
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
  background: {NAVY};
}}
.site-nav a {{ color: #fff; text-decoration: none; white-space: nowrap; }}
.site-nav a:hover, .site-nav a:focus {{ text-decoration: underline; }}
.nav-home {{ font-weight: 650; font-size: 0.95rem; flex: 0 0 auto; }}
.nav-links {{
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
  list-style: none;
  margin: 0;
  padding: 0;
  flex: 1 1 auto;
}}
.nav-links a {{ font-size: 0.88rem; }}
.nav-jump {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  width: 100%;
  min-width: 0;
  white-space: nowrap;
}}
.nav-jump span {{ flex: 0 0 auto; }}
.nav-jump select {{
  flex: 1 1 auto;
  min-width: 0;
  width: 100%;
  font: 14px/1.3 Georgia, "Times New Roman", serif;
  padding: 6px 8px;
}}
.joint-controls select {{
  max-width: 420px;
  font: 14px/1.3 Georgia, "Times New Roman", serif;
  padding: 4px 6px;
}}
header, section {{
  background: var(--card);
  border: 1px solid var(--line);
  padding: 28px 32px;
  margin: 0 0 22px;
}}
section {{ scroll-margin-top: 7.2rem; }}
header {{ scroll-margin-top: 7.2rem; }}
.kicker {{ letter-spacing: 0.08em; text-transform: uppercase; color: var(--steel); font-size: 0.78rem; }}
h1, h2, h3 {{ color: var(--navy); font-weight: 650; line-height: 1.25; }}
h1 {{ font-size: 2.1rem; margin: 0.2em 0 0.4em; }}
h2 {{ font-size: 1.45rem; margin-top: 0; }}
h3 {{ font-size: 1.05rem; }}
.lede {{ color: #333; }}
.path, .path code {{ font-size: 0.86rem; color: #444; }}
code {{ font-family: Consolas, "Courier New", monospace; font-size: 0.9em; }}
.formula {{
  background: #f7f8fa;
  border-left: 4px solid var(--navy);
  padding: 12px 16px;
  margin: 12px 0;
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.92rem;
  white-space: pre-wrap;
}}
.table-wrap {{ overflow-x: auto; margin: 16px 0 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: var(--navy); color: #fff; font-weight: 600; }}
caption {{ caption-side: top; text-align: left; color: #444; padding: 0 0 8px; font-style: italic; }}
tr:nth-child(even) td {{ background: #f3f6fb; }}
td.miss {{ color: var(--miss); font-style: italic; }}
.six h3 {{ margin-bottom: 0.35em; }}
.prose p {{ margin: 0.45em 0 0.8em; }}
a {{ color: var(--navy); }}
footer {{ color: #555; font-size: 0.9rem; padding: 8px 4px; }}
.table-fold {{
  border: 1px solid var(--line);
  background: #fbfaf7;
  margin: 12px 0 18px;
  padding: 0 12px 8px;
}}
.table-fold > summary {{
  cursor: pointer;
  font-weight: 650;
  color: var(--navy);
  padding: 10px 4px;
  list-style: none;
}}
.table-fold > summary::-webkit-details-marker {{ display: none; }}
.table-fold > summary::before {{ content: "▸ "; }}
.table-fold[open] > summary::before {{ content: "▾ "; }}
.joint-controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: center;
  margin: 8px 0 12px;
  padding: 12px 14px;
  background: #f7f8fa;
  border: 1px solid var(--line);
}}
.joint-controls label {{ display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }}
.joint-help {{ font-size: 0.95rem; color: #333; }}
.swatch-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px 16px;
  margin: 0 0 12px;
}}
.swatch {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.88rem;
}}
.swatch i {{
  width: 14px;
  height: 14px;
  border-radius: 2px;
  display: inline-block;
  flex: 0 0 auto;
}}
.panel-grid {{
  display: grid;
  gap: 18px 20px;
}}
.panel-grid.cols-1 {{ grid-template-columns: 1fr; }}
.panel-grid.cols-2 {{ grid-template-columns: 1fr 1fr; }}
.chart-card {{
  min-width: 0;
  border: 1px solid var(--line);
  background: #fff;
  padding: 8px 8px 4px;
}}
.chart-card .chart-title {{
  margin: 0 0 2px;
  font-size: 0.98rem;
  color: var(--navy);
  font-weight: 650;
}}
.chart-kicker {{
  margin: 10px 4px 4px;
  font-size: 0.92rem;
  color: #333;
}}
.plotly-lazy {{
  width: 100%;
  min-height: 560px;
}}
.heatmap-panel, .chart-stage {{ min-height: 200px; }}
.plotly-graph-div, .js-plotly-plot {{
  width: 100%;
  position: relative;
  z-index: 0;
}}
[hidden] {{ display: none !important; }}
@media (max-width: 900px) {{
  .page {{ padding-left: 14px; padding-right: 14px; }}
  .panel-grid.cols-2 {{ grid-template-columns: 1fr; }}
}}
@media print {{
  body {{ background: #fff; }}
  .page {{ max-width: none; padding: 12px; }}
  .site-nav {{ position: static; background: #fff; color: var(--navy); box-shadow: none; }}
  .site-nav a {{ color: var(--navy); }}
  header, section {{ break-inside: avoid; border: none; padding: 12px 0; }}
  .table-fold {{ border: none; }}
  .table-fold:not([open]) {{ display: block; }}
}}
"""

PAGE_JS = """
<script>
(function () {
  function goTo(hash) {
    if (!hash) return;
    var target = document.querySelector(hash);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    history.replaceState(null, "", hash);
    var jump = document.getElementById("nav-jump-select");
    if (jump) jump.value = hash;
  }
  var jump = document.getElementById("nav-jump-select");
  if (jump) {
    jump.addEventListener("change", function () { goTo(this.value); });
  }
  document.querySelectorAll(".site-nav a[href^='#']").forEach(function (link) {
    link.addEventListener("click", function (event) {
      event.preventDefault();
      goTo(this.getAttribute("href"));
    });
  });
  function isHidden(el) {
    return !!(el && el.closest("[hidden]"));
  }
  function mountLazyPlots(root) {
    if (!window.Plotly) return;
    (root || document).querySelectorAll(".plotly-lazy").forEach(function (el) {
      if (isHidden(el)) return;
      if (el.classList.contains("js-plotly-plot")) {
        Plotly.Plots.resize(el);
        return;
      }
      var specNode = document.getElementById(el.id + "-spec");
      if (!specNode) return;
      var spec = JSON.parse(specNode.textContent);
      Plotly.newPlot(el, spec.data, spec.layout, {displaylogo: false, responsive: true}).then(function () {
        Plotly.Plots.resize(el);
      });
    });
  }
  function resizeVisiblePlots(root) {
    if (!window.Plotly) return;
    (root || document).querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      if (isHidden(gd)) return;
      Plotly.Plots.resize(gd);
    });
  }
  function applySizeFilter(explorer) {
    var sizeSel = explorer.querySelector(".exp-size");
    if (!sizeSel || !window.Plotly) return;
    var size = sizeSel.value;
    explorer.querySelectorAll(".chart-stage .js-plotly-plot").forEach(function (gd) {
      if (!gd.data) return;
      var vis = gd.data.map(function (tr) {
        if (size === "all" || tr.meta == null || tr.meta === "") return true;
        return String(tr.meta) === String(size);
      });
      vis.forEach(function (show, i) {
        Plotly.restyle(gd, {visible: show}, [i]);
      });
    });
  }
  function syncExplorer(explorer) {
    var viewSel = explorer.querySelector(".exp-view");
    if (!viewSel) return;
    var view = viewSel.value;
    var modelSel = explorer.querySelector(".exp-model");
    var protSel = explorer.querySelector(".exp-protocol");
    var modelLab = explorer.querySelector(".exp-model-label");
    var protLab = explorer.querySelector(".exp-protocol-label");
    if (modelLab) modelLab.hidden = view !== "model";
    if (protLab) protLab.hidden = view !== "protocol";
    explorer.querySelectorAll(".swatch-legend[data-for]").forEach(function (el) {
      el.hidden = el.getAttribute("data-for") !== view;
    });
    var bank = explorer.querySelector(".chart-bank");
    var stage = explorer.querySelector(".chart-stage");
    if (!bank || !stage) return;
    while (stage.firstChild) bank.appendChild(stage.firstChild);
    var layout = explorer.getAttribute("data-layout");
    var cards;
    if (layout === "pair-grid") {
      var sel = view === "model"
        ? '.chart-card[data-model="' + modelSel.value + '"]'
        : '.chart-card[data-protocol="' + protSel.value + '"]';
      cards = [].slice.call(bank.querySelectorAll(sel));
    } else {
      var key = view === "model" ? modelSel.value : protSel.value;
      cards = [].slice.call(bank.querySelectorAll('.chart-card[data-view="' + view + '"][data-key="' + key + '"]'));
    }
    cards.sort(function (a, b) {
      return Number(a.getAttribute("data-order") || 0) - Number(b.getAttribute("data-order") || 0);
    });
    stage.className = "chart-stage panel-grid cols-1";
    cards.forEach(function (card) { stage.appendChild(card); });
    window.requestAnimationFrame(function () {
      mountLazyPlots(explorer);
      window.requestAnimationFrame(function () {
        resizeVisiblePlots(explorer);
        applySizeFilter(explorer);
      });
    });
  }
  function bindExplorer(explorer) {
    ["change"].forEach(function () {
      explorer.querySelectorAll(".exp-view, .exp-model, .exp-protocol, .exp-size").forEach(function (el) {
        el.addEventListener("change", function () { syncExplorer(explorer); });
      });
    });
    syncExplorer(explorer);
  }
  function syncNamedHeatmap(explorer) {
    var sel = explorer.querySelector("select");
    if (!sel) return;
    var setting = sel.value;
    explorer.querySelectorAll(".heatmap-panel").forEach(function (el) {
      if (el.getAttribute("data-setting") === setting) el.removeAttribute("hidden");
      else el.setAttribute("hidden", "");
    });
    window.requestAnimationFrame(function () { resizeVisiblePlots(explorer); });
  }
  document.querySelectorAll(".heatmap-explorer").forEach(function (explorer) {
    var sel = explorer.querySelector("select");
    if (sel) sel.addEventListener("change", function () { syncNamedHeatmap(explorer); });
    syncNamedHeatmap(explorer);
  });
  document.querySelectorAll(".chart-explorer").forEach(bindExplorer);
  window.addEventListener("resize", function () {
    resizeVisiblePlots(document);
    document.querySelectorAll(".chart-explorer").forEach(applySizeFilter);
  });
  window.addEventListener("load", function () {
    document.querySelectorAll(".heatmap-explorer").forEach(syncNamedHeatmap);
    document.querySelectorAll(".chart-explorer").forEach(function (explorer) {
      syncExplorer(explorer);
    });
    if (location.hash) {
      window.requestAnimationFrame(function () { goTo(location.hash); });
    }
  });
})();
</script>
"""


def collect() -> dict:
    process_data = {}
    missing = []
    for item in PROCESS_BRIEFING:
        slug = item["slug"]
        default_rows, default_rel = load_process_classifiers(slug, "1_PH_Default_Parameters")
        tuned_rows, tuned_rel = load_process_classifiers(slug, "2_PH_Tuned_Parameters")
        audit, audit_rel = load_process_csv(slug, "6_Sampling_Ratio_Audit", "sampling_ratio_audit.csv")
        perm, perm_rel = load_process_csv(
            slug, "8_Permutation_Test_Of_Class_Difference", "permutation_test_results.csv"
        )
        revised, revised_rel = load_process_csv(slug, "9_Revised_Snapshot_Protocol", "ml_results.csv")
        process_data[slug] = {
            "default_rows": default_rows,
            "default_rel": default_rel,
            "tuned_rows": tuned_rows,
            "tuned_rel": tuned_rel,
            "audit": audit,
            "audit_rel": audit_rel,
            "perm": perm,
            "perm_rel": perm_rel,
            "revised": revised,
            "revised_rel": revised_rel,
        }
        if not default_rows:
            missing.append(default_rel)
        if not tuned_rows:
            missing.append(tuned_rel)
        if audit.empty:
            missing.append(audit_rel)
        if perm.empty:
            missing.append(perm_rel)
        if revised.empty:
            missing.append(revised_rel)
    tabular = load_tabular_default()
    if not tabular:
        missing.append(
            f"6_Results/Default_Parameters/1_ML_Default_Parameters/{DATASET}/model_results.pkl"
        )
    id_bundle = load_id_bundle()
    if id_bundle["csv"].empty:
        missing.append(id_bundle["csv_rel"])
    sss = {
        "count": load_sss("1_Snapshot_Count_Sweep"),
        "points": load_sss("2_Points_Per_Snapshot_Sweep"),
        "joint": load_sss("3_Snapshot_Count_Across_Cloud_Sizes"),
    }
    for key, folder in (
        ("count", "1_Snapshot_Count_Sweep"),
        ("points", "2_Points_Per_Snapshot_Sweep"),
        ("joint", "3_Snapshot_Count_Across_Cloud_Sizes"),
    ):
        if sss[key].empty:
            missing.append(f"6_Results/Snapshot_Sample_Size/{folder}/all_summary.csv")
    return {
        "tabular": tabular,
        "process": process_data,
        "id": id_bundle,
        "sss": sss,
        "missing": missing,
    }


def validate_sss(sss: dict) -> list[str]:
    warnings = []
    count = sss["count"]
    points = sss["points"]
    if not count.empty:
        pts = sorted(count["points_per_snapshot"].dropna().unique())
        if len(pts) != 1:
            warnings.append(
                f"Snapshot count sweep expected one points-per-snapshot value, found {pts}."
            )
    if not points.empty:
        snaps = sorted(points["n_snapshots"].dropna().unique())
        if snaps != [N_TRAIN_POOL]:
            warnings.append(
                f"Points-per-snapshot sweep expected n_snapshots={N_TRAIN_POOL}, found {snaps}."
            )
    return warnings


def build_html(data: dict) -> str:
    warnings = data["missing"] + validate_sss(data["sss"])
    banner = ""
    if warnings:
        items = "".join(f"<li><code>{esc(w)}</code></li>" for w in warnings)
        banner = f"<section><h2>Missing or unexpected artefacts</h2><ul>{items}</ul></section>"
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>',
        f"<title>{esc(TITLE)}</title>",
        f"<style>{CSS}</style>",
        f"<script>{get_plotlyjs()}</script>",
        "</head>",
        "<body>",
        site_nav(),
        '<main class="page">',
        section_intro(),
        banner,
        section_tabular(data["tabular"]),
        section_eight(data["process"]),
        section_statistics(data["id"]),
        section_sss(data["sss"]["count"], data["sss"]["points"], data["sss"]["joint"]),
        section_revised(data["process"]),
        appendix_a(data["process"]),
        appendix_b(data["process"]),
        appendix_c(data["process"]),
        "<footer>Compiler: <code>6_Results/Run_Queue/_compile_results_briefing.py</code>. "
        "Paper figures remain the matplotlib <code>visualize_results.py</code> pipeline. "
        "The table-only PDF compiler is a separate artefact.</footer>",
        "</main>",
        PAGE_JS,
        "</body></html>",
    ]
    return "\n".join(parts)


def main() -> None:
    data = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_text = build_html(data)
    OUT_HTML.write_text(html_text, encoding="utf-8")
    manifest = {
        "title": TITLE,
        "dataset": DATASET,
        "output": str(OUT_HTML.relative_to(ROOT)).replace("\\", "/"),
        "bytes": OUT_HTML.stat().st_size,
        "missing": data["missing"],
        "sss_warnings": validate_sss(data["sss"]),
    }
    manifest_path = OUT_DIR / "TDA_Results_Briefing.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_HTML}")
    print(f"Wrote {manifest_path}")
    if data["missing"]:
        print("Missing artefacts:")
        for row in data["missing"]:
            print(" ", row)


if __name__ == "__main__":
    main()
