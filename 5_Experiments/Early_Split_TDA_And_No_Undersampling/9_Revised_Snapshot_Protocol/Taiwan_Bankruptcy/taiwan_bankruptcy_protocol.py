# -*- coding: utf-8 -*-
"""
Early_Split_TDA_And_No_Undersampling / 9_Revised_Snapshot_Protocol
Dataset: Taiwanese Bankruptcy Prediction

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : early
- Undersample  : False
- PCA rank     : 10  (historical Exp 3 rank for this table)
- Points per snapshot : fixed absolute count (design-chosen points per snapshot), not a class percent
- Training snapshots  : 60 (default)
- Test snapshots      : 15 (default)
Revised snapshot protocol: fixed points per snapshot (not a class percent),
default 60 training snapshots and 15 test snapshots, overlap reported separately
from the reuse-ratio formula. PCA rank is the same Exp 3 rank as experiments 1–8.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

import run_protocol
from protocol_lib import (
    DEFAULT_TEST_L,
    DEFAULT_TRAIN_L,
    DCCCD_FULL_L,
    ZANIAR_TEST_L,
    ZANIAR_TRAIN_L,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'taiwan_bankruptcy'
PROTOCOL_BUCKET = 'Early_Split_TDA_And_No_Undersampling'
EXPERIMENT = "9_Revised_Snapshot_Protocol"
FOLDER = 'Taiwan_Bankruptcy'

SPLIT_TIMING = 'early'
UNDERSAMPLE = False
PCA_N_COMPONENTS = 10
TRAIN_SNAPSHOTS = DEFAULT_TRAIN_L          # 60
TEST_SNAPSHOTS = DEFAULT_TEST_L            # 15
ZANIAR_TRAIN_SNAPSHOTS = ZANIAR_TRAIN_L    # 60, 80, 100
ZANIAR_TEST_SNAPSHOTS = ZANIAR_TEST_L      # 15, 22, 30
DCCCD_FULL_SNAPSHOTS = DCCCD_FULL_L        # 60, 75, 90 (DCCCD non-split arm only)
RANDOM_STATE = 42

assert run_protocol.PROTOCOL_BUCKET == PROTOCOL_BUCKET
assert run_protocol.SPLIT_TIMING == SPLIT_TIMING
assert run_protocol.UNDERSAMPLE == UNDERSAMPLE

# =============================================================================
# Stage 1 — Design: intrinsic dimension, joint points-per-snapshot, reuse
# =============================================================================
# Points per snapshot is a fixed absolute count, not floor(class * percent / 100).
# Reuse ratio R = (points_per_snapshot * n_snapshots) / class_count.
print("=" * 72)
print(f"{EXPERIMENT} / {FOLDER}")
print(f"split={SPLIT_TIMING}  undersample={UNDERSAMPLE}  PCA={PCA_N_COMPONENTS}")
print(f"default snapshots: train={TRAIN_SNAPSHOTS}  test={TEST_SNAPSHOTS}")
print("=" * 72)

design = run_protocol.design_for_dataset(DATASET_KEY)
chosen_t = int(design["chosen_t"])
eff_train = int(design.get("effective_defaults", {}).get("train_l", TRAIN_SNAPSHOTS))
eff_test = int(design.get("effective_defaults", {}).get("test_l", TEST_SNAPSHOTS))
print(
    f"Chosen points per snapshot={chosen_t}  "
    f"effective train snapshots={eff_train}  test snapshots={eff_test}  "
    f"t_sweep={design['t_sweep']}"
)

# =============================================================================
# Stage 2 — Split ML: independent train/test snapshots, overlap, classifiers
# =============================================================================
# For each points-per-snapshot value in the design sweep:
#   1. split customers according to this arm
#   2. fit PCA as this arm requires
#   3. draw train snapshots (default 60) and test snapshots (default 15)
#   4. Ripser each cloud
#   5. report pairwise snapshot overlap
#   6. train the five classifiers
run_protocol.run_split_ml({DATASET_KEY: design}, [DATASET_KEY], sweep=True)

# =============================================================================
# Stage 3 — Optional full-table (non-split) arm on DCCCD only
# =============================================================================
if DATASET_KEY == "credit_card_default":
    run_protocol.run_full_ml({DATASET_KEY: design}, [DATASET_KEY])
else:
    print("Full-table non-split arm is DCCCD-only; skipped for this dataset.")
