"""
apply_prediction_models.py

- Read master player list (master_list.csv)
- For each position (WR/RB/TE/QB):
    - Load all Bakery-derived .pkl files from:
        data/Bakery/_derived/{pos}
    - For each pkl:
        - Extract trained model + feature list + imputer
        - Apply same invert_cols + imputer transform as training
        - Predict a score for every player at that position
        - Add one column per model (e.g. bakery_WR_wr_model_seed123_subs40)
    - Compute an ensemble score (mean of all model columns)
    - Save one CSV per position with scores appended.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, List, Any

import numpy as np
import pandas as pd

# ========== CONFIG ==========
ROOT = Path("/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty")

MASTER_CSV = ROOT / "data" / "processed" / "master_list.csv"
BAKERY_DIR = ROOT / "data" / "Bakery" / "_derived"

OUT_DIR    = ROOT / "data" / "processed" / "bakery_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Positions you care about and corresponding subfolder names
POSITIONS = ["WR", "RB", "TE", "QB"]  # must match folder names under _derived/{pos}

# Make sure we can import src.utils (for invert_cols)
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils import invert_cols  # same helper used in training


# ========== HELPERS ==========

def find_pos_col(columns: List[str]) -> str | None:
    """Locate position column from typical candidates."""
    lower = {c.lower(): c for c in columns}
    for cand in ["pos", "position"]:
        if cand in lower:
            return lower[cand]
    return None


def extract_model_and_features(obj: Any) -> Tuple[Any, List[str], str, Any]:
    """
    Given a loaded pickle object from your GB tuning script, extract:
      - model: fitted estimator (obj['model'])
      - features: list of column names used by the model (obj['feature_names'])
      - label: a short name to use in the output column (e.g., seed)
      - imputer: the fitted SimpleImputer used at training time (obj['imputer'])
    """
    model = None
    features = None
    imputer = None
    label = "model"

    if isinstance(obj, dict):
        # Model (per your training script: key='model')
        if "model" in obj:
            model = obj["model"]

        # Feature list (per your training script: key='feature_names')
        if "feature_names" in obj and obj["feature_names"] is not None:
            features = list(obj["feature_names"])

        # Imputer (per your training script: key='imputer')
        if "imputer" in obj:
            imputer = obj["imputer"]

        # Label: prefer seed, fall back to model tag / name if present
        if "seed" in obj:
            label = f"seed{obj['seed']}"
        elif "n_subsets" in obj:
            label = f"subs{obj['n_subsets']}"
        elif "position" in obj:
            label = str(obj["position"])
        else:
            # generic fallback
            label = "model"
    else:
        # Object itself is the model (not your usual case, but keep as fallback)
        model = obj
        if hasattr(obj, "feature_names_in_"):
            features = list(obj.feature_names_in_)
        imputer = None

    if model is None:
        raise ValueError("Could not find model object inside pkl (expected dict with key 'model').")

    if features is None:
        raise ValueError("Could not find feature list inside pkl (expected key 'feature_names').")

    return model, features, label, imputer


def predict_for_position(
    df_pos: pd.DataFrame,
    pos: str,
    pos_dir: Path
) -> pd.DataFrame:
    """
    For a position-specific subset of master_list, load all Bakery pkl files
    from pos_dir, run predictions (using invert_cols + imputer), and return
    df with new score columns.
    """
    if not pos_dir.exists():
        print(f"[WARN] No Bakery directory for pos={pos}: {pos_dir}")
        return df_pos

    score_cols: list[str] = []

    for pkl_path in sorted(pos_dir.glob("*.pkl")):
        print(f"[INFO] Loading model from: {pkl_path}")
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {pkl_path.name}: {e}")
            continue

        try:
            model, features, label, imputer = extract_model_and_features(obj)
        except Exception as e:
            print(f"[WARN] Skipping {pkl_path.name}: {e}")
            continue

        # Ensure all required features are present
        missing = [c for c in features if c not in df_pos.columns]
        if missing:
            print(f"[WARN] Skipping {pkl_path.name}: missing features {missing[:10]} (showing up to 10).")
            continue

        X = df_pos[features].copy()

        # Apply same inversion as training
        try:
            X = invert_cols(X)
        except Exception as e:
            print(f"[WARN] invert_cols failed for {pkl_path.name}: {e}")
            continue

        # Apply imputer if available
        if imputer is not None:
            try:
                X_arr = imputer.transform(X)
            except Exception as e:
                print(f"[WARN] Imputer transform failed for {pkl_path.name}: {e}")
                continue
        else:
            # Fallback: raw values
            X_arr = X.values

        # Predict
        try:
            preds = model.predict(X_arr)
        except Exception as e:
            print(f"[WARN] Prediction failed for {pkl_path.name}: {e}")
            continue

        # Use filename stem plus label to name column
        col_name = f"bakery_{pos}_{pkl_path.stem}_{label}"
        if col_name in df_pos.columns:
            col_name = f"{col_name}_dup"

        df_pos[col_name] = preds
        score_cols.append(col_name)
        print(f"[INFO] Added score column: {col_name} (shape={preds.shape})")

    # Ensemble across all model outputs for this position
    if score_cols:
        ensemble_col = f"bakery_{pos}_ensemble"
        df_pos[ensemble_col] = df_pos[score_cols].mean(axis=1)
        print(f"[INFO] Added ensemble column {ensemble_col} using {len(score_cols)} models.")
    else:
        print(f"[WARN] No usable models found for pos={pos}.")

    return df_pos


# ========== MAIN ==========

def main():
    print("Loading master list:", MASTER_CSV)
    master = pd.read_csv(MASTER_CSV)

    pos_col = find_pos_col(list(master.columns))
    if pos_col is None:
        raise ValueError("Could not find a position column (expected 'pos' or 'position').")

    for pos in POSITIONS:
        print(f"\n===== Processing position: {pos} =====")
        mask = master[pos_col].astype(str).str.upper() == pos
        df_pos = master.loc[mask].copy()

        if df_pos.empty:
            print(f"[WARN] No players with pos={pos} in master list. Skipping.")
            continue

        pos_dir = BAKERY_DIR / pos
        df_pos_scored = predict_for_position(df_pos, pos, pos_dir)

        out_path = OUT_DIR / f"bakery_scores_{pos}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_pos_scored.to_csv(out_path, index=False)
        print(f"[INFO] Saved {len(df_pos_scored)} rows with scores for {pos} → {out_path}")

    print("\n[DONE] Bakery model scoring complete.")


if __name__ == "__main__":
    main()
