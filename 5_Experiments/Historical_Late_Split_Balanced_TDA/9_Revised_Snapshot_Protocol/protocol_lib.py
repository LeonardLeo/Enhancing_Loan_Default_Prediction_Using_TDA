# -*- coding: utf-8 -*-
"""
Revised snapshot protocol (Experiment 9 in each TDA arm; historically Exp 28).

Snapshot rules (this experiment only — not used by experiments 1–8):
  - Fixed absolute t (points per snapshot), same t for train and test.
  - Default split snapshot counts: train l=60, test l=15.
  - Zaniar sweep (3 points each): train {60,80,100}, test {15,22,30}.
  - Full-data (non-split) on DCCCD: l in {60,75,90}.
  - Intrinsic dimension b guides t and the email formula for l.
  - Formula concern and reuse-ratio concern are SEPARATE.
  - Overlap: pairwise snapshot overlap + reuse ratios + significance tests.

Protocol knobs (first-class, not stubs) are chosen by the arm:
  - split_timing = "early" | "late"
  - undersample  = True | False
"""

from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.stats import mannwhitneyu
def win_long_path(path) -> Path:
    """Windows path that can be created/opened beyond MAX_PATH (260)."""
    raw = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        if raw.startswith("\\\\"):
            raw = "\\\\?\\UNC\\" + raw[2:]
        else:
            raw = "\\\\?\\" + raw
    return Path(raw)


from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state

# Optional package — preferred ID estimators
try:
    import skdim

    HAS_SKDIM = True
except Exception:  # pragma: no cover
    HAS_SKDIM = False


# -----------------------------------------------------------------------------
# Canonical sweep grids (meeting answers)
# -----------------------------------------------------------------------------
DEFAULT_TRAIN_L = 60
DEFAULT_TEST_L = 15
ZANIAR_TRAIN_L = (60, 80, 100)  # 3 points in 60–100
ZANIAR_TEST_L = (15, 22, 30)  # 3 points in 15–30
DCCCD_FULL_L = (60, 75, 90)  # 3 points in 60–90 for the bigger dataset
TARGET_REUSE = 1.0
TARGET_T_OVER_CLASS = 0.20


# =============================================================================
# Concern A — email formula:  l ~ (t / log t)^{2/b}
# =============================================================================
def formula_l_from_t_b(t: int, b: float, log_base: str = "e") -> float:
    """
    Formal sample-complexity-style rule from the email:

        l ≈ (t / log(t))^{2/b}

    where:
      t = points per snapshot
      b = intrinsic dimension estimate
      l = suggested number of snapshots

    This concern answers: "how many snapshots are theoretically needed for
    topological summaries at this (t, b)?" — NOT whether points are reused.
    """
    t = int(t)
    if t < 3:
        raise ValueError("t must be >= 3 for log(t) to be meaningful")
    if b is None or not np.isfinite(b) or b <= 0:
        raise ValueError(f"b must be a positive finite intrinsic dimension, got {b}")
    if log_base == "e":
        log_t = math.log(t)
    elif log_base == "10":
        log_t = math.log10(t)
    else:
        raise ValueError("log_base must be 'e' or '10'")
    if log_t <= 0:
        raise ValueError("log(t) must be positive")
    return float((t / log_t) ** (2.0 / float(b)))


def formula_t_candidates_for_target_l(
    target_l: int,
    b: float,
    t_min: int = 10,
    t_max: int = 200,
) -> List[Dict[str, float]]:
    """Find integer t where formula_l_from_t_b(t,b) is closest to target_l."""
    rows = []
    for t in range(t_min, t_max + 1):
        l_hat = formula_l_from_t_b(t, b)
        rows.append(
            {
                "t": t,
                "b": float(b),
                "l_formula": l_hat,
                "abs_diff_to_target_l": abs(l_hat - target_l),
                "target_l": target_l,
            }
        )
    rows.sort(key=lambda r: r["abs_diff_to_target_l"])
    return rows


# =============================================================================
# Concern B — reuse-ratio / sampling constraints
# =============================================================================
def reuse_ratio(t: int, l: int, n_class: int) -> float:
    """
    Expected number of times a typical point from a class appears across
    l independent without-replacement draws of size t from a pool of size n:

        R = (t * l) / n_class

    Target: R ≲ 1 (near or below one).
    """
    if n_class <= 0:
        return float("nan")
    return float(t * l) / float(n_class)


def max_t_for_reuse(
    n_class: int,
    l: int,
    max_reuse: float = TARGET_REUSE,
) -> int:
    """Largest integer t with (t*l)/n_class <= max_reuse and t <= n_class."""
    if n_class < 2 or l < 1:
        return 0
    return int(max(2, min(n_class, math.floor(max_reuse * n_class / l))))


def max_l_for_reuse(
    n_class: int,
    t: int,
    max_reuse: float = TARGET_REUSE,
) -> int:
    """Largest integer l with (t*l)/n_class <= max_reuse."""
    if n_class < 2 or t < 1:
        return 0
    return int(max(1, math.floor(max_reuse * n_class / t)))


def t_over_class(t: int, n_class: int) -> float:
    if n_class <= 0:
        return float("nan")
    return float(t) / float(n_class)


def audit_reuse_constraints(
    n_pos: int,
    n_neg: int,
    t: int,
    l: int,
    max_reuse: float = TARGET_REUSE,
    max_t_frac: float = TARGET_T_OVER_CLASS,
) -> Dict[str, Any]:
    """
    Concern B audit. Independent of the email formula.

    Checks:
      1) t / n_c < max_t_frac for each class c
      2) (t * l) / n_c ≲ max_reuse for each class c
    The binding class is the minority (smaller n_c).
    """
    n_min = min(n_pos, n_neg)
    n_maj = max(n_pos, n_neg)
    r_pos = reuse_ratio(t, l, n_pos)
    r_neg = reuse_ratio(t, l, n_neg)
    return {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_minority": int(n_min),
        "n_majority": int(n_maj),
        "t": int(t),
        "l": int(l),
        "t_over_n_pos": t_over_class(t, n_pos),
        "t_over_n_neg": t_over_class(t, n_neg),
        "reuse_pos": r_pos,
        "reuse_neg": r_neg,
        "reuse_binding": max(r_pos, r_neg),
        "ok_t_fraction": (t_over_class(t, n_min) < max_t_frac),
        "ok_reuse": (r_pos <= max_reuse and r_neg <= max_reuse),
        "max_t_reuse_ok": max_t_for_reuse(n_min, l, max_reuse),
        "max_l_reuse_ok": max_l_for_reuse(n_min, t, max_reuse),
        "max_reuse_target": max_reuse,
        "max_t_frac_target": max_t_frac,
    }


def recommend_t_l_separated(
    n_pos: int,
    n_neg: int,
    b: float,
    train_l_target: int = DEFAULT_TRAIN_L,
    test_l_target: int = DEFAULT_TEST_L,
    t_candidates: Optional[Sequence[int]] = None,
    t_min_practical: int = 10,
) -> Dict[str, Any]:
    """
    Produce recommendations while keeping Concern A and Concern B separate.

    When minority pools are too small for (t_min_practical, train_l=60) under
    reuse ≤ 1, we *adapt train_l / test_l downward* and document that Concern B
    overrode the meeting default (formula Concern A still reported separately).
    """
    n_min = min(n_pos, n_neg)
    adapted_train_l = int(train_l_target)
    adapted_test_l = int(test_l_target)
    adaptation_notes = []

    # Ensure a practical t exists under reuse at the requested l; else lower l.
    t_at_train = max_t_for_reuse(n_min, adapted_train_l, TARGET_REUSE)
    if t_at_train < t_min_practical:
        # Need l <= n_min / t_min_practical
        adapted_train_l = max(1, int(math.floor(TARGET_REUSE * n_min / t_min_practical)))
        adapted_train_l = min(adapted_train_l, train_l_target)
        adaptation_notes.append(
            f"train_l reduced {train_l_target}→{adapted_train_l} so t≥{t_min_practical} "
            f"keeps reuse≤{TARGET_REUSE} on n_min={n_min}"
        )
        t_at_train = max_t_for_reuse(n_min, adapted_train_l, TARGET_REUSE)

    t_at_test = max_t_for_reuse(n_min, adapted_test_l, TARGET_REUSE)
    if t_at_test < t_min_practical:
        adapted_test_l = max(1, int(math.floor(TARGET_REUSE * n_min / t_min_practical)))
        adapted_test_l = min(adapted_test_l, test_l_target)
        adaptation_notes.append(
            f"test_l reduced {test_l_target}→{adapted_test_l} so t≥{t_min_practical} "
            f"keeps reuse≤{TARGET_REUSE} on n_min={n_min}"
        )
        t_at_test = max_t_for_reuse(n_min, adapted_test_l, TARGET_REUSE)

    # Absolute floor: need t>=3 for formula and PH
    t_hi = max(3, min(t_at_train, n_min))
    if t_candidates is None:
        mid = max(3, t_hi // 2)
        lo = max(3, t_hi // 4)
        extras = [t for t in (10, 20, 40, 60, 80) if t <= t_hi]
        t_candidates = sorted(set([lo, mid, t_hi, *extras]))
        t_candidates = [t for t in t_candidates if 3 <= t <= n_min]

    formula_rows = []
    for t in t_candidates:
        if not np.isfinite(b) or b <= 0:
            l_f = float("nan")
        else:
            l_f = formula_l_from_t_b(t, b)
        formula_rows.append(
            {
                "t": t,
                "b": float(b) if np.isfinite(b) else None,
                "l_formula": l_f,
                "vs_train_default_60": (l_f - train_l_target) if np.isfinite(l_f) else None,
                "vs_test_default_15": (l_f - test_l_target) if np.isfinite(l_f) else None,
            }
        )

    reuse_rows = []
    for t in t_candidates:
        reuse_rows.append(
            {
                "t": t,
                **audit_reuse_constraints(n_pos, n_neg, t, adapted_train_l),
                "split": "train",
            }
        )
        reuse_rows.append(
            {
                "t": t,
                **audit_reuse_constraints(n_pos, n_neg, t, adapted_test_l),
                "split": "test",
            }
        )

    feasible_train = [
        r for r in reuse_rows if r["split"] == "train" and r["ok_reuse"] and r["ok_t_fraction"]
    ]
    if feasible_train:
        chosen_t = max(r["t"] for r in feasible_train)
    else:
        chosen_t = max(3, t_hi)

    chosen_audit_train = audit_reuse_constraints(n_pos, n_neg, chosen_t, adapted_train_l)
    chosen_audit_test = audit_reuse_constraints(n_pos, n_neg, chosen_t, adapted_test_l)
    if np.isfinite(b) and b > 0 and chosen_t >= 3:
        l_formula_at_chosen = formula_l_from_t_b(chosen_t, b)
    else:
        l_formula_at_chosen = float("nan")

    return {
        "concern_A_formula": {
            "definition": "l ≈ (t / log t)^{2/b}",
            "rows": formula_rows,
            "at_chosen_t": {
                "t": chosen_t,
                "b": float(b) if np.isfinite(b) else None,
                "l_formula": l_formula_at_chosen,
                "interpretation": (
                    "Formula-suggested snapshot count at chosen t. "
                    "Compare to meeting defaults (train 60 / test 15) separately "
                    "from reuse feasibility."
                ),
            },
        },
        "concern_B_reuse": {
            "definition": "R=(t*l)/n_class ≲ 1 and t/n_class < 0.20",
            "rows": reuse_rows,
            "max_t_at_train_l": max_t_for_reuse(n_min, adapted_train_l),
            "max_t_at_test_l": max_t_for_reuse(n_min, adapted_test_l),
            "adapted_train_l": adapted_train_l,
            "adapted_test_l": adapted_test_l,
            "adaptation_notes": adaptation_notes,
        },
        "chosen_joint": {
            "t": int(chosen_t),
            "train_l": int(adapted_train_l),
            "test_l": int(adapted_test_l),
            "meeting_train_l_requested": int(train_l_target),
            "meeting_test_l_requested": int(test_l_target),
            "same_t_train_test": True,
            "no_undersampling": True,
            "train_reuse_audit": chosen_audit_train,
            "test_reuse_audit": chosen_audit_test,
            "formula_l_at_chosen_t": l_formula_at_chosen,
            "why": (
                "Pick the largest reuse-safe t at the (possibly adapted) train_l "
                "(binding class = minority). Keep the SAME t for test. "
                "Report formula l separately; do not override meeting 60/15 "
                "with the formula alone — only Concern B (reuse) may reduce l."
            ),
            "adaptation_notes": adaptation_notes,
        },
        "per_class_minima": per_class_safe_minima(n_pos, n_neg, t=chosen_t),
    }


def choose_joint_t_train_test_l(
    train_pos: int,
    train_neg: int,
    test_pos: int,
    test_neg: int,
    target_train_l: int = DEFAULT_TRAIN_L,
    target_test_l: int = DEFAULT_TEST_L,
    t_max_cap: int = 120,
    min_train_l: int = 5,
    min_test_l: int = 3,
) -> Dict[str, Any]:
    """
    Jointly choose a single t and (train_l, test_l) under Concern B.

    Preference order:
      1) reuse ≤ 1 on BOTH train and test minority pools
      2) train_l as close as possible to 60, test_l as close as possible to 15
      3) larger t (richer topology) among ties
      4) if impossible, relax test reuse to ≤ 2.0 (documented), never train reuse > 1
    """
    n_tr = min(train_pos, train_neg)
    n_te = min(test_pos, test_neg)
    t_hi = int(min(n_tr, n_te, t_max_cap))
    candidates = []
    for t in range(t_hi, 4, -1):
        max_tr_l = max_l_for_reuse(n_tr, t, 1.0)
        max_te_l = max_l_for_reuse(n_te, t, 1.0)
        train_l = min(target_train_l, max_tr_l)
        test_l = min(target_test_l, max_te_l)
        if train_l >= min_train_l and test_l >= min_test_l:
            score = (train_l / target_train_l) + (test_l / target_test_l) + 0.001 * t
            candidates.append(
                {
                    "t": t,
                    "train_l": train_l,
                    "test_l": test_l,
                    "score": score,
                    "test_reuse_limit": 1.0,
                    "relaxed_test_reuse": False,
                }
            )
    if not candidates:
        # Relax test reuse to 2.0; keep train reuse ≤ 1
        for t in range(t_hi, 4, -1):
            max_tr_l = max_l_for_reuse(n_tr, t, 1.0)
            max_te_l = max_l_for_reuse(n_te, t, 2.0)
            train_l = min(target_train_l, max_tr_l)
            test_l = min(target_test_l, max_te_l)
            if train_l >= min_train_l and test_l >= min_test_l:
                score = (train_l / target_train_l) + (test_l / target_test_l) + 0.001 * t - 0.5
                candidates.append(
                    {
                        "t": t,
                        "train_l": train_l,
                        "test_l": test_l,
                        "score": score,
                        "test_reuse_limit": 2.0,
                        "relaxed_test_reuse": True,
                    }
                )
    if not candidates:
        # Last resort: smallest workable PH setting
        t = max(5, min(t_hi, 10))
        return {
            "t": t,
            "train_l": max(1, max_l_for_reuse(n_tr, t, 1.0)),
            "test_l": max(1, max_l_for_reuse(n_te, t, 2.0)),
            "score": 0.0,
            "test_reuse_limit": 2.0,
            "relaxed_test_reuse": True,
            "fallback": True,
            "n_train_min": n_tr,
            "n_test_min": n_te,
        }
    best = max(candidates, key=lambda r: r["score"])
    best["n_train_min"] = n_tr
    best["n_test_min"] = n_te
    best["fallback"] = False
    return best


def per_class_safe_minima(
    n_pos: int,
    n_neg: int,
    t: int,
    max_reuse: float = TARGET_REUSE,
) -> Dict[str, Any]:
    """
    Safe upper bounds on l (and implied minima discussion) per class at fixed t.
    Conservative experimental l must not exceed the minority-class bound.
    """
    return {
        "t": int(t),
        "pos": {
            "n": int(n_pos),
            "max_l_reuse_le_1": max_l_for_reuse(n_pos, t, max_reuse),
            "reuse_at_l60": reuse_ratio(t, 60, n_pos),
            "reuse_at_l100": reuse_ratio(t, 100, n_pos),
        },
        "neg": {
            "n": int(n_neg),
            "max_l_reuse_le_1": max_l_for_reuse(n_neg, t, max_reuse),
            "reuse_at_l60": reuse_ratio(t, 60, n_neg),
            "reuse_at_l100": reuse_ratio(t, 100, n_neg),
        },
        "conservative_max_l": int(
            min(max_l_for_reuse(n_pos, t, max_reuse), max_l_for_reuse(n_neg, t, max_reuse))
        ),
    }


# =============================================================================
# Intrinsic dimension (skdim + fallbacks)
# =============================================================================
def estimate_intrinsic_dimensions(
    X: np.ndarray,
    n_samples: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Estimate intrinsic dimension b with multiple estimators.

    Primary package: scikit-dimension (skdim) — TwoNN, MLE (Levina–Bickel),
    DANCo (when computationally feasible), lPCA.
    """
    rng = check_random_state(random_state)
    X = np.asarray(X, dtype=float)
    if n_samples is not None and len(X) > n_samples:
        idx = rng.choice(len(X), size=n_samples, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    out: Dict[str, Any] = {
        "n_points_used": int(len(Xs)),
        "n_features": int(Xs.shape[1]),
        "package": "scikit-dimension" if HAS_SKDIM else "fallback_local",
        "estimators": {},
    }

    if HAS_SKDIM:
        try:
            twonn = skdim.id.TwoNN().fit(Xs)
            out["estimators"]["TwoNN"] = float(twonn.dimension_)
        except Exception as exc:  # pragma: no cover
            out["estimators"]["TwoNN"] = f"error:{exc}"
        try:
            mle = skdim.id.MLE(K=20).fit(Xs)
            out["estimators"]["MLE_LevinaBickel"] = float(mle.dimension_)
        except Exception as exc:  # pragma: no cover
            out["estimators"]["MLE_LevinaBickel"] = f"error:{exc}"
        try:
            lpca = skdim.id.lPCA().fit(Xs)
            out["estimators"]["lPCA"] = float(lpca.dimension_)
        except Exception as exc:  # pragma: no cover
            out["estimators"]["lPCA"] = f"error:{exc}"
        if hasattr(skdim.id, "MiND_ML"):
            try:
                mind = skdim.id.MiND_ML().fit(Xs)
                out["estimators"]["MiND_ML"] = float(mind.dimension_)
            except Exception as exc:  # pragma: no cover
                out["estimators"]["MiND_ML"] = f"error:{exc}"
        # DANCo is slower; only on modest samples
        if len(Xs) <= 800:
            try:
                danco = skdim.id.DANCo().fit(Xs)
                out["estimators"]["DANCo"] = float(danco.dimension_)
            except Exception as exc:  # pragma: no cover
                out["estimators"]["DANCo"] = f"error:{exc}"
    else:
        # Minimal Two-NN fallback
        from scipy.spatial.distance import cdist

        D = cdist(Xs, Xs)
        np.fill_diagonal(D, np.inf)
        nn = np.sort(D, axis=1)[:, :2]
        mu = nn[:, 1] / np.maximum(nn[:, 0], 1e-12)
        mu = mu[mu > 1]
        out["estimators"]["TwoNN"] = float(1.0 / np.mean(np.log(mu))) if len(mu) else float("nan")

    numeric = [v for v in out["estimators"].values() if isinstance(v, float) and np.isfinite(v)]
    # Prefer TwoNN as primary b (matches Exp 26 / Facco et al.)
    b_primary = out["estimators"].get("TwoNN")
    if not isinstance(b_primary, float) or not np.isfinite(b_primary):
        b_primary = float(np.median(numeric)) if numeric else float("nan")
    out["b_primary_TwoNN"] = float(b_primary) if np.isfinite(b_primary) else float("nan")
    out["b_median_available"] = float(np.median(numeric)) if numeric else float("nan")
    return out


# =============================================================================
# Fixed-t landmark generation (NO percentage, NO undersampling)
# =============================================================================
def split_classes_no_balance(
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Keep full class pools — do not undersample."""
    data = X.copy()
    data["__y__"] = y.values
    pos = data[data["__y__"] == positive_label].drop(columns=["__y__"]).reset_index(drop=True)
    neg = data[data["__y__"] != positive_label].drop(columns=["__y__"]).reset_index(drop=True)
    return {"default": pos, "non-default": neg}


def undersample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: int = 1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Undersample the majority class to the minority count. First-class mode."""
    data = X.copy()
    data["__y__"] = pd.Series(y).values
    pos = data[data["__y__"] == positive_label]
    neg = data[data["__y__"] != positive_label]
    n = min(len(pos), len(neg))
    if n < 2:
        raise ValueError(f"Cannot undersample: pos={len(pos)} neg={len(neg)}")
    pos = pos.sample(n=n, random_state=random_state)
    neg = neg.sample(n=n, random_state=random_state)
    out = pd.concat([pos, neg], ignore_index=True)
    return out.drop(columns=["__y__"]), out["__y__"].astype(int)


def split_classes_maybe_balance(
    X: pd.DataFrame,
    y: pd.Series,
    undersample: bool,
    positive_label: int = 1,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    if undersample:
        X, y = undersample_xy(X, y, positive_label=positive_label, random_state=random_state)
    return split_classes_no_balance(X, y, positive_label=positive_label)


def late_split_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Late-split first-class mode: impute / scale / PCA on the FULL table,
    then stratified 80/20 on the reduced customers. Leaky by design
    (matches Historical / No_Undersampling PCA timing).
    """
    from sklearn.impute import SimpleImputer

    miss = X.isna().astype(float)
    miss.columns = [f"miss_{c}" for c in X.columns]
    keep = [c for c in miss.columns if miss[c].sum() > 0]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    if keep:
        X_imp = pd.concat([X_imp, miss[keep]], axis=1)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X_imp)
    n_comp = min(n_components, max(1, Xs.shape[0] - 1), Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    cols = [f"PCA_{i}" for i in range(1, n_comp + 1)]
    Xp = pd.DataFrame(pca.fit_transform(Xs), columns=cols, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(
        Xp, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, float(pca.explained_variance_ratio_.sum())


def prepare_protocol_clouds(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    split_timing: str,
    undersample: bool,
    positive_label: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    First-class protocol factory for all four TDA arms.

    early + undersample=False  → original Exp 28 (Early_Split_TDA_And_No_Undersampling)
    early + undersample=True   → Early_Split_TDA
    late  + undersample=True   → Historical_Late_Split_Balanced_TDA
    late  + undersample=False  → No_Undersampling
    """
    split_timing = str(split_timing).strip().lower()
    if split_timing not in {"early", "late"}:
        raise ValueError(f"split_timing must be 'early' or 'late', got {split_timing!r}")

    if split_timing == "early":
        X_train, X_test, y_train, y_test, var = early_split_pca(
            X, y, n_components=n_components, test_size=test_size, random_state=random_state
        )
        pca_fit = "train_only"
        if undersample:
            X_train, y_train = undersample_xy(
                X_train, y_train, positive_label=positive_label, random_state=random_state
            )
            X_test, y_test = undersample_xy(
                X_test, y_test, positive_label=positive_label, random_state=random_state
            )
    else:
        if undersample:
            # Balance on the raw table first so PCA sees the same balanced cloud
            # the landmarks will be drawn from, then full-table PCA, then split.
            X, y = undersample_xy(X, y, positive_label=positive_label, random_state=random_state)
        X_train, X_test, y_train, y_test, var = late_split_pca(
            X, y, n_components=n_components, test_size=test_size, random_state=random_state
        )
        pca_fit = "full_table"

    meta = {
        "split_timing": split_timing,
        "undersample": bool(undersample),
        "pca_fit": pca_fit,
        "variance_retained": float(var),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_pos": int((y_train == positive_label).sum()),
        "train_neg": int((y_train != positive_label).sum()),
        "test_pos": int((y_test == positive_label).sum()),
        "test_neg": int((y_test != positive_label).sum()),
    }
    return X_train, X_test, y_train, y_test, meta


def generate_fixed_t_snapshots(
    class_frames: Dict[str, pd.DataFrame],
    t: int,
    l: int,
    output_root: Path,
    tag: str,
    random_state: int = 42,
    store_index_sets: bool = True,
    undersample: bool = False,
) -> Dict[str, Any]:
    """
    Draw l snapshots of exactly t points (without replacement within a snapshot)
    from each class pool. Same absolute t for every class.

    Saves:
      output_root / {class}_T{t} / landmarks_{i}.csv
      optional index JSONs for overlap analysis
    """
    rng = check_random_state(random_state)
    meta: Dict[str, Any] = {
        "t": int(t),
        "l": int(l),
        "tag": tag,
        "undersample": bool(undersample),
        "no_undersampling": not bool(undersample),
        "classes": {},
        "index_sets": {},
    }
    output_root = win_long_path(Path(output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    for class_name, frame in class_frames.items():
        n = len(frame)
        if t > n:
            raise ValueError(
                f"Cannot draw t={t} from class '{class_name}' with only n={n} rows "
                f"(no undersampling / no replacement within a snapshot)."
            )
        class_dir = win_long_path(output_root / f"{class_name}_T{t}")
        class_dir.mkdir(parents=True, exist_ok=True)
        index_sets = []
        for i in range(l):
            # Independent snapshot seeds derived from master seed
            local_rng = check_random_state(rng.randint(0, 2**31 - 1))
            idx = local_rng.choice(n, size=t, replace=False)
            snap = frame.iloc[idx]
            snap.to_csv(class_dir / f"landmarks_{i}.csv", index=False)
            index_sets.append(sorted(map(int, idx.tolist())))
        meta["classes"][class_name] = {"n_pool": n, "n_snapshots": l, "t": t}
        if store_index_sets:
            meta["index_sets"][class_name] = index_sets
            with open(class_dir / "snapshot_index_sets.json", "w", encoding="utf-8") as f:
                json.dump(index_sets, f)
    with open(output_root / f"snapshot_meta_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in meta.items() if k != "index_sets"}, f, indent=2)
        # store indices separately (can be large)
    with open(output_root / f"snapshot_indices_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta["index_sets"], f)
    return meta


def pairwise_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def pairwise_overlap_fraction(a: Sequence[int], b: Sequence[int], t: int) -> float:
    """|A∩B| / t  (fraction of a snapshot's points shared with another)."""
    if t <= 0:
        return float("nan")
    return len(set(a) & set(b)) / float(t)


def analyze_snapshot_overlap(
    index_sets: List[List[int]],
    t: int,
    n_pool: int,
    n_pair_sample: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Pairwise snapshot overlap summary for one class.
    If C(l,2) is large, subsample pairs.
    """
    rng = check_random_state(random_state)
    l = len(index_sets)
    all_pairs = list(combinations(range(l), 2))
    if len(all_pairs) > n_pair_sample:
        chosen = rng.choice(len(all_pairs), size=n_pair_sample, replace=False)
        pairs = [all_pairs[i] for i in chosen]
        sampled = True
    else:
        pairs = all_pairs
        sampled = False

    jaccards = []
    overlaps = []
    for i, j in pairs:
        jaccards.append(pairwise_jaccard(index_sets[i], index_sets[j]))
        overlaps.append(pairwise_overlap_fraction(index_sets[i], index_sets[j], t))

    # Theoretical expected overlap fraction for two independent samples without replacement:
    # E[|A∩B|]/t = t/n  (approximately, exact hypergeometric mean is t*(t/n) wait)
    # Exact: E[|A∩B|] = t * (t / n) = t^2 / n   => E[frac] = t/n
    expected_overlap_frac = t / n_pool if n_pool else float("nan")
    expected_jaccard = (
        expected_overlap_frac / (2 - expected_overlap_frac)
        if np.isfinite(expected_overlap_frac) and expected_overlap_frac < 2
        else float("nan")
    )

    return {
        "n_snapshots": l,
        "n_pairs_evaluated": len(pairs),
        "pairs_sampled": sampled,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
        "std_jaccard": float(np.std(jaccards)) if jaccards else float("nan"),
        "mean_overlap_frac": float(np.mean(overlaps)) if overlaps else float("nan"),
        "std_overlap_frac": float(np.std(overlaps)) if overlaps else float("nan"),
        "expected_overlap_frac_indep": float(expected_overlap_frac),
        "expected_jaccard_approx": float(expected_jaccard),
        "reuse_ratio_tl_over_n": reuse_ratio(t, l, n_pool),
        "jaccard_values": jaccards,
        "overlap_frac_values": overlaps,
    }


def overlap_significance_tests(
    index_sets: List[List[int]],
    t: int,
    n_pool: int,
    n_permutations: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Formal tests that observed pairwise overlap is consistent with independent
    uniform sampling without replacement (null), vs systematically too high.

    1) One-sided permutation / Monte-Carlo test on mean overlap fraction.
    2) Mann–Whitney U: observed pair overlaps vs null-simulated pair overlaps.
    """
    rng = check_random_state(random_state)
    observed = analyze_snapshot_overlap(
        index_sets, t=t, n_pool=n_pool, n_pair_sample=300, random_state=random_state
    )
    obs_mean = observed["mean_overlap_frac"]

    null_means = []
    null_pair_values = []
    for _ in range(n_permutations):
        sim_sets = [rng.choice(n_pool, size=t, replace=False).tolist() for _ in range(len(index_sets))]
        sim = analyze_snapshot_overlap(
            sim_sets, t=t, n_pool=n_pool, n_pair_sample=200, random_state=rng.randint(0, 2**31 - 1)
        )
        null_means.append(sim["mean_overlap_frac"])
        null_pair_values.extend(sim["overlap_frac_values"])

    # p-value: fraction of null means >= observed (more overlap than chance)
    p_mean = (1 + sum(m >= obs_mean for m in null_means)) / (n_permutations + 1)

    obs_pairs = observed["overlap_frac_values"]
    if obs_pairs and null_pair_values:
        # alternative: observed overlaps stochastically greater than null
        u_stat, p_mw = mannwhitneyu(obs_pairs, null_pair_values, alternative="greater")
    else:
        u_stat, p_mw = float("nan"), float("nan")

    return {
        "observed_mean_overlap_frac": obs_mean,
        "null_mean_of_means": float(np.mean(null_means)),
        "null_std_of_means": float(np.std(null_means)),
        "expected_overlap_frac_theory": observed["expected_overlap_frac_indep"],
        "p_value_mean_overlap_greater_than_null": float(p_mean),
        "mannwhitney_U": float(u_stat) if np.isfinite(u_stat) else float("nan"),
        "mannwhitney_p_greater": float(p_mw) if np.isfinite(p_mw) else float("nan"),
        "n_permutations": n_permutations,
        "interpretation": (
            "Large p-values support the null that snapshots behave like independent "
            "uniform draws. Small p-values suggest excess overlap beyond chance."
        ),
        "summary_without_raw": {
            k: v
            for k, v in observed.items()
            if k not in ("jaccard_values", "overlap_frac_values")
        },
    }


# =============================================================================
# Barcodes + ML
# =============================================================================
def compute_barcode_stats_for_snapshot_dir(
    snapshot_dir: Path,
    label: int,
    dim: int = 2,
) -> pd.DataFrame:
    from utils import compute_barcode_statistics

    rows = []
    files = sorted(snapshot_dir.glob("landmarks_*.csv"))
    for fp in files:
        pts = pd.read_csv(fp).values
        dgms = ripser(pts, maxdim=dim - 1)["dgms"]
        row = []
        for d in range(dim):
            row.extend(compute_barcode_statistics(dgms[d]))
        row.append(label)
        rows.append(row)
    cols = [f"g{i}_{j}" for j in range(dim) for i in range(1, 13)] + ["label"]
    return pd.DataFrame(rows, columns=cols)


def build_barcode_matrix_for_tag(
    landmarks_root: Path,
    t: int,
    label_map: Dict[int, str] = None,
) -> pd.DataFrame:
    if label_map is None:
        label_map = {1: "default", 0: "non-default"}
    frames = []
    for lab, name in label_map.items():
        d = win_long_path(landmarks_root / f"{name}_T{t}")
        if not d.exists():
            raise FileNotFoundError(d)
        frames.append(compute_barcode_stats_for_snapshot_dir(d, label=lab))
    return pd.concat(frames, ignore_index=True)


def fit_simple_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    X_tr = train_df.drop(columns=["label"]).values
    y_tr = train_df["label"].values
    X_te = test_df.drop(columns=["label"]).values
    y_te = test_df["label"].values

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    models = {
        "logistic": LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, class_weight="balanced"
        ),
        "xgb": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "svm": SVC(probability=True, random_state=random_state, class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_te, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
        }
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_te)[:, 1]
            try:
                row["roc_auc"] = float(roc_auc_score(y_te, proba))
            except Exception:
                row["roc_auc"] = float("nan")
            try:
                row["average_precision"] = float(average_precision_score(y_te, proba))
            except Exception:
                row["average_precision"] = float("nan")
        else:
            row["roc_auc"] = float("nan")
            row["average_precision"] = float("nan")
        rows.append(row)
    return rows


def early_split_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Protocol B: split first, then impute / scale / PCA on train only.
    Median imputation + missing indicators keep Polish/PKDD usable without leakage.
    """
    from sklearn.impute import SimpleImputer

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Missing indicators (train-fit columns) then median impute
    miss_tr = X_train.isna().astype(float)
    miss_te = X_test.isna().astype(float)
    miss_tr.columns = [f"miss_{c}" for c in X_train.columns]
    miss_te.columns = [f"miss_{c}" for c in X_test.columns]
    # Drop all-zero indicator columns (no missingness in train)
    keep = [c for c in miss_tr.columns if miss_tr[c].sum() > 0]
    miss_tr = miss_tr[keep]
    miss_te = miss_te[keep]

    imputer = SimpleImputer(strategy="median")
    Xtr_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    Xte_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    if keep:
        Xtr_imp = pd.concat([Xtr_imp, miss_tr], axis=1)
        Xte_imp = pd.concat([Xte_imp, miss_te], axis=1)

    scaler = MinMaxScaler()
    Xtr_s = scaler.fit_transform(Xtr_imp)
    Xte_s = scaler.transform(Xte_imp)

    n_comp = min(n_components, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    cols = [f"PCA_{i}" for i in range(1, n_comp + 1)]
    Xtr_p = pd.DataFrame(pca.fit_transform(Xtr_s), columns=cols, index=X_train.index)
    Xte_p = pd.DataFrame(pca.transform(Xte_s), columns=cols, index=X_test.index)
    return Xtr_p, Xte_p, y_train, y_test, float(pca.explained_variance_ratio_.sum())


def save_json(path: Path, obj: Any) -> None:
    path = win_long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)
