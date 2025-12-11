# src/models/gradientboosting_tuning.py
# ===============================================================
# Position-aware Grade — wide search over feature subsets & hypers
# - position: RB / WR / TE / QB (affects CSV path + target names)
# - Randomly sample feature combos (+ limited interactions)
# - Model: GradientBoostingRegressor (GB)
# - Must/Ban features & interactions + hierarchy control (strong/weak/none)
# - Interaction-only allowed when hierarchy="none" (parents not forced as bases)
# - Soft limiter for Draft Capital influence
# - SHAP per-player contributions + global importances split by base/interaction
# - NaN-safe: drops all-NaN columns globally + per-subset before imputation
# - Saves per-run leaderboards and overall summary CSV
# ===============================================================
from __future__ import annotations

import json, re, time, warnings, datetime, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prefer the sklearn implementation; if static analysis/environment lacks it, provide
# a lightweight fallback SimpleImputer that supports strategy="median".
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Help static analyzers know the symbol exists without forcing a runtime import.
    from sklearn.impute import SimpleImputer  # type: ignore

try:
    from sklearn.impute import SimpleImputer  # type: ignore
except Exception:
    import numpy as _np
    class SimpleImputer:
        """
        Minimal fallback SimpleImputer supporting only strategy="median".
        Provides fit, transform, and fit_transform that handle 1D/2D inputs
        and replace non-finite values with the per-column median.
        """
        def __init__(self, strategy="median"):
            if strategy != "median":
                raise ValueError("Fallback SimpleImputer only supports strategy='median'")
            self.strategy = strategy
            self.statistics_ = None

        def fit(self, X):
            arr = _np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            stats = []
            for j in range(arr.shape[1]):
                col = arr[:, j].astype(float)
                finite = col[_np.isfinite(col)]
                if finite.size == 0:
                    stats.append(_np.nan)
                else:
                    stats.append(_np.median(finite))
            self.statistics_ = _np.array(stats, dtype=float)
            return self

        def transform(self, X):
            arr = _np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if self.statistics_ is None:
                raise ValueError("SimpleImputer instance is not fitted yet")
            out = arr.astype(float).copy()
            n_cols = self.statistics_.shape[0]
            # If input has fewer columns than statistics (rare), trim statistics; if more, pad with nan
            if out.shape[1] != n_cols:
                if out.shape[1] < n_cols:
                    stats = self.statistics_[: out.shape[1]]
                else:
                    stats = _np.concatenate([self.statistics_, _np.full(out.shape[1] - n_cols, _np.nan)])
            else:
                stats = self.statistics_
            for j in range(out.shape[1]):
                mask = ~_np.isfinite(out[:, j])
                if _np.isfinite(stats[j]):
                    out[mask, j] = stats[j]
            return out

        def fit_transform(self, X):
            return self.fit(X).transform(X)

# Prefer the sklearn implementation; help static analyzers know the symbols exist without forcing a runtime import.
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # for static type checkers / editors
    from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score  # type: ignore

# Perform a dynamic import at runtime to avoid static-analysis errors that report
# "Import 'sklearn.model_selection' could not be resolved from source".
# If sklearn is unavailable, fall back to lightweight stubs that raise clear errors when used.
try:
    import importlib
    _sklearn_ms = importlib.import_module("sklearn.model_selection")
    train_test_split = _sklearn_ms.train_test_split
    KFold = _sklearn_ms.KFold
    RandomizedSearchCV = _sklearn_ms.RandomizedSearchCV
    cross_val_score = _sklearn_ms.cross_val_score
except Exception:
    # If sklearn isn't available at import time (common in some static-analysis setups),
    # provide lightweight stubs that raise clear runtime errors when actually used.
    def train_test_split(*args, **kwargs):
        raise RuntimeError(
            "scikit-learn is required for train_test_split; install scikit-learn or ensure it's available in the runtime environment."
        )

    class KFold:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "scikit-learn is required for KFold; install scikit-learn or ensure it's available in the runtime environment."
            )

    class RandomizedSearchCV:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "scikit-learn is required for RandomizedSearchCV; install scikit-learn or ensure it's available in the runtime environment."
            )

    def cross_val_score(*args, **kwargs):
        raise RuntimeError(
            "scikit-learn is required for cross_val_score; install scikit-learn or ensure it's available in the runtime environment."
        )


try:
    import importlib
    _sklearn_metrics = importlib.import_module("sklearn.metrics")
    r2_score = _sklearn_metrics.r2_score
    mean_absolute_error = _sklearn_metrics.mean_absolute_error
    mean_squared_error = _sklearn_metrics.mean_squared_error
except Exception:
    def r2_score(*args, **kwargs):
        raise RuntimeError(
            "scikit-learn 'metrics.r2_score' is required; install scikit-learn or ensure it's available in the runtime environment."
        )
    def mean_absolute_error(*args, **kwargs):
        raise RuntimeError(
            "scikit-learn 'metrics.mean_absolute_error' is required; install scikit-learn or ensure it's available in the runtime environment."
        )
    def mean_squared_error(*args, **kwargs):
        raise RuntimeError(
            "scikit-learn 'metrics.mean_squared_error' is required; install scikit-learn or ensure it's available in the runtime environment."
        )

try:
    import importlib
    _sklearn_ensemble = importlib.import_module("sklearn.ensemble")
    GradientBoostingRegressor = _sklearn_ensemble.GradientBoostingRegressor
except Exception:
    class GradientBoostingRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "scikit-learn 'ensemble.GradientBoostingRegressor' is required; install scikit-learn or ensure it's available in the runtime environment."
            )

try:
    import importlib
    _sklearn_inspect = importlib.import_module("sklearn.inspection")
    permutation_importance = _sklearn_inspect.permutation_importance
except Exception:
    def permutation_importance(*args, **kwargs):
        raise RuntimeError(
            "scikit-learn 'inspection.permutation_importance' is required; install scikit-learn or ensure it's available in the runtime environment."
        )


try:
    import importlib
    shap = importlib.import_module("shap")
    _HAVE_SHAP = True
except Exception:
    shap = None
    _HAVE_SHAP = False
# ----------------------------- Features -----------------------------

from src.utils import BASE_FEATURES, INTERACTIONS, ALIASES

# ------------------------- POSITION UTILS -------------------------

from src.utils import target_position, default_csv_for_position, default_out_dir


# ---------------------------- HELPERS -----------------------------

from src.utils import find_col, to_num, rmse, invert_cols, limit_draft_capital_series

# ---------------------------- Model Spaces -----------------------------

def model_spaces(random_state: int):

    """
    Return a list of (tag, estimator, param_grid) for all supported models. removed all but Gradient Boosting

    """
    return [
        # Gradient Boosting (trees) - Optimized grid to reduce overkill
        ("GB", GradientBoostingRegressor(random_state=random_state), {
            "n_estimators": [200, 400, 600, 800],  # Reduced from 7 to 4 values, lower max
            "learning_rate": [0.05, 0.1, 0.15],    # Reduced from 6 to 3 values, broader steps
            "max_depth": [2, 3, 4, 5, 6, 7],       # Keep as-is, good range
            "subsample": [0.8, 0.9, 1.0],          # Keep as-is, good coverage  
            "min_samples_leaf": [1, 3, 5],         # Reduced from 6 to 3 values
        }),

    ]

# ---------------------- NaN/Inf sanitizers ----------------------

from src.utils import sanitize_frame_global, sanitize_train_test


# -------- SHAP helpers / saving weights / feature Caps --------------

from src.analysis_utils import compute_shap, save_importances
from src.utils import sample_subset


# ---------------------------- Main Function ----------------------------
def run_seed_for_subsets(
    position: str,
    project_root: Path,
    n_subsets: int,
    *,
    seeds: list[int],
    max_base_feats: int,
    max_interactions: int,
    n_iter_per_model: int,
    cv_folds: int,
    test_size: float = 0.20,
    must_feats: list[str] | None = None,
    ban_feats: list[str] | None = None,
    must_inters: list[str] | None = None,
    ban_inters: list[str] | None = None,
    interaction_hierarchy: str = "weak",
    # feature value limiter (pre-model): compress Draft Capital amplitude
    draft_cap_cap: float | None = None,
    draft_cap_lower_q: float = 0.05,
    draft_cap_upper_q: float = 0.95,
    # importance reporting options
    enable_shap: bool = True,
    enable_permutation_importance: bool = False,
    draft_cap_importance_cap: float | None = None,
    breakout_age_importance_cap: float | None = None,
    draft_age_importance_cap: float | None = None,
    # legacy arg names (kept for notebook compatibility)
    must_use_features: list[str] | None = None,
    banned_features: list[str] | None = None,
    must_use_interactions: list[str] | None = None,
    banned_interactions: list[str] | None = None,
    priority_strength: str | None = None
) -> pd.DataFrame:
    """
    Wide search for a given position and n_subsets, per seed.
    Defensive against:
      - all-NaN columns
      - single-column/shape errors (the 1D/scalar problem)
      - too-large CV folds for small RB samples
      - failed model fits (skips subset instead of crashing)
      - empty leaderboard (surfaces a helpful diagnostic)
    """

    # ---- Normalize legacy names ------------------------------------------------
    if must_feats is None:            must_feats = must_use_features
    if ban_feats is None:             ban_feats = banned_features
    if must_inters is None:           must_inters = must_use_interactions
    if ban_inters is None:            ban_inters = banned_interactions
    if priority_strength is not None: interaction_hierarchy = priority_strength

    must_feats  = list(must_feats  or [])
    ban_feats   = list(ban_feats   or [])
    must_inters = list(must_inters or [])
    ban_inters  = list(ban_inters  or [])

    # ---- SHAP availability -----------------------------------------------------
    try:
        shap_enabled = bool(enable_shap)
        # if your top-level imported _HAVE_SHAP is available, respect it
        if 'shap' in globals():
            pass
    except Exception:
        shap_enabled = False

    pos = position.upper()
    out_dir = default_out_dir(project_root, pos)
    csv_path = default_csv_for_position(project_root, pos)

    # ---- Debug banner ----------------------------------------------------------
    print("\n========== DEBUG PATHS ==========")
    print(f"Position          : {pos}")
    print(f"CSV path          : {csv_path}")
    print(f"Output directory  : {out_dir}")
    print(f"Hierarchy         : {interaction_hierarchy}")
    print(f"Must feats        : {must_feats}")
    print(f"Ban feats         : {ban_feats}")
    print(f"Must inters       : {must_inters}")
    print(f"Ban inters        : {ban_inters}")
    if draft_cap_cap is not None:
        print(f"DraftCap limiter  : cap={draft_cap_cap}, lower_q={draft_cap_lower_q}, upper_q={draft_cap_upper_q}")
    print("=================================\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data -------------------------------------------------------------
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Target + names
    y_col = None
    for cand in target_position(pos):  # e.g., ["WR Grade","WRGrade","WR_Grade"]
        y_col = find_col(df, [cand])
        if y_col:
            break
    if not y_col:
        raise ValueError(f"Could not find target column for {pos} among {target_position(pos)}")
    name_col = find_col(df, ["Player","Player Name","Name"]) or "Player"

    # Map available canonical features to actual CSV columns via ALIASES
    mapped: dict[str, str] = {}
    for feat in BASE_FEATURES:
        col = find_col(df, ALIASES.get(feat, [feat]))
        if col is not None:
            mapped[feat] = col
    if not mapped:
        raise ValueError("No usable base features found in CSV for this position.")

    # ---- Apply must/ban to pool -----------------------------------------------
    ban_feats_set = {f.strip() for f in ban_feats}
    must_feats_list = [f for f in must_feats if f in mapped]
    base_pool = [f for f in mapped.keys() if f not in ban_feats_set]

    # Prepare allowed interactions (parents must be mapped & not banned)
    allowed_inters: dict[str, tuple[str, str]] = {}
    for iname, (a, b) in INTERACTIONS.items():
        if iname in ban_inters:
            continue
        if a in mapped and b in mapped and a not in ban_feats_set and b not in ban_feats_set:
            allowed_inters[iname] = (a, b)
    must_inters_list = [n for n in must_inters if n in allowed_inters]

    # If hierarchy == "none" we allow interaction-only, and block parents
    # from random base sampling unless they're explicitly in must_feats.
    if interaction_hierarchy == "none":
        parents_to_block = set()
        for iname in must_inters_list:
            a, b = allowed_inters[iname]
            parents_to_block.update([a, b])
        base_pool = [f for f in base_pool if (f not in parents_to_block) or (f in must_feats_list)]

    # ---- Build numeric matrices + global sanitize -----------------------------
    X_all_raw = pd.DataFrame({feat: to_num(df[col]) for feat, col in mapped.items()})
    X_all_raw = sanitize_frame_global(X_all_raw, min_non_na_frac=0.0)  # replace inf, drop all-NaN cols
    y_all     = to_num(df[y_col])
    names_all = df[name_col].astype(str).fillna("")

    # keep rows with target
    mask = y_all.notna()
    X_all_raw, y_all, names_all = (
        X_all_raw.loc[mask].reset_index(drop=True),
        y_all.loc[mask].reset_index(drop=True),
        names_all.loc[mask].reset_index(drop=True),
    )

    # drop Draft Capital == 0 and optionally compress its amplitude
    if "Draft Capital" in X_all_raw.columns:
        keep = X_all_raw["Draft Capital"] != 0
        X_all_raw = X_all_raw.loc[keep].reset_index(drop=True)
        y_all     = y_all.loc[keep].reset_index(drop=True)
        names_all = names_all.loc[keep].reset_index(drop=True)
        if draft_cap_cap is not None:
            X_all_raw["Draft Capital"] = limit_draft_capital_series(
                X_all_raw["Draft Capital"],
                cap=draft_cap_cap,
                lower_q=draft_cap_lower_q,
                upper_q=draft_cap_upper_q,
            )

    # basic dataset diagnostics (helps catch RB-specific issues)
    print(f"[{pos}] Rows after filtering: {len(y_all)} | Feature cols: {X_all_raw.shape[1]}")
    if len(y_all) < 10:
        print(f"[{pos}] WARNING: very small sample (n={len(y_all)}). Consider lowering cv_folds.")
    if y_all.nunique(dropna=True) < 2:
        raise ValueError(f"[{pos}] Target has <2 unique values after filtering — cannot train.")

    rows: list[dict] = []

    # ---- Per-seed block --------------------------------------------------------
    for seed in seeds:
        t0 = time.perf_counter()

        pred_path  = out_dir / f"{pos.lower()}_wide_best_preds_seed{seed}_subs{n_subsets}.csv"
        meta_path  = out_dir / f"{pos.lower()}_wide_best_meta_seed{seed}_subs{n_subsets}.json"
        board_path = out_dir / f"{pos.lower()}_wide_leaderboard_seed{seed}_subs{n_subsets}.csv"
        shap_contribs_path = out_dir / f"{pos.lower()}_shap_contribs_seed{seed}_subs{n_subsets}.csv"

        rng = np.random.default_rng(seed)

        # train/test split
        X_tr_raw, X_te_raw, y_tr, y_te, n_tr, n_te = train_test_split(
            X_all_raw, y_all, names_all, test_size=test_size, random_state=seed
        )
        if len(y_tr) < 3:
            raise ValueError(f"[{pos}] After split (seed={seed}), train size={len(y_tr)} is too small to CV.")

        # Cap CV folds to avoid RB failures (KFold requires n_splits <= n_samples)
        cv_splits = min(cv_folds, int(len(y_tr)))
        if cv_splits < 2:
            cv_splits = 2

        leaderboard: list[dict] = []
        baseline_done = False

        for subset_idx in range(n_subsets):
            # ---------- Build the feature subset ----------
            if not baseline_done:
                # Baseline: either the user’s must features, or the best single feature by CV
                if must_feats_list:
                    bases = list(dict.fromkeys(must_feats_list))[:max_base_feats]
                    inters = list(dict.fromkeys(must_inters_list))[:max_interactions]
                    baseline_done = True
                else:
                    # Try every candidate base feature and pick best single by CV on train
                    best_feat, best_cv = None, -1e18
                    for f in [fname for fname in base_pool if fname in X_tr_raw.columns]:
                        # must have at least one non-NaN in train
                        if X_tr_raw[f].notna().sum() == 0:
                            continue
                        # 2D array guard (prevents the scalar/1D error)
                        imp = SimpleImputer(strategy="median")
                        X_single = imp.fit_transform(X_tr_raw[[f]])  # shape = (n,1)
                        if not np.isfinite(X_single).any():
                            continue
                        # CV with capped folds
                        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
                        try:
                            cv = cross_val_score(
                                GradientBoostingRegressor(random_state=seed),
                                X_single, y_tr, scoring="r2", cv=kf
                            ).mean()
                        except Exception:
                            continue
                        if np.isfinite(cv) and cv > best_cv:
                            best_cv, best_feat = cv, f
                    if best_feat is not None:
                        bases, inters = [best_feat], []
                        baseline_done = True
                    else:
                        # If even baseline can't find a usable column, skip this subset
                        continue
            else:
                # Random sample with hierarchy rules
                bases, inters = sample_subset(
                    rng,
                    base_pool=base_pool,
                    max_bases=max_base_feats,
                    allowed_inters=allowed_inters,
                    max_inters=max_interactions,
                    must_feats=must_feats_list,
                    must_inters=must_inters_list,
                    interaction_hierarchy=interaction_hierarchy,
                )

            # ---------- Design matrices ----------
            Xtr_df, Xte_df = X_tr_raw[bases].copy(), X_te_raw[bases].copy()

            # Add interactions (computed from parents in raw matrices)
            for iname in inters:
                a, b = allowed_inters[iname]
                if a in X_tr_raw.columns and b in X_tr_raw.columns:
                    Xtr_df[iname] = X_tr_raw[a].values * X_tr_raw[b].values
                    Xte_df[iname] = X_te_raw[a].values * X_te_raw[b].values

            # Drop columns that are all-NaN in TRAIN (imputer cannot fit them),
            # and keep train/test aligned to the remaining columns.
            Xtr_df, Xte_df = sanitize_train_test(Xtr_df, Xte_df)
            if Xtr_df.shape[1] == 0:
                # everything became invalid for this sample — skip it
                continue

            feat_cols = list(Xtr_df.columns)

            # invert “lower-is-better” columns and impute
            Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)
            imp = SimpleImputer(strategy="median")
            try:
                Xtr = imp.fit_transform(Xtr_df)  # must be (n, k)
                Xte = imp.transform(Xte_df)
            except Exception:
                # any shape/NaN issue — skip the subset
                continue

            # ---------- Model selection on TRAIN (CV) ----------
            best_cv, best_tag, best_est = -1e18, None, None
            for tag, est, grid in model_spaces(seed):
                # Ensure we don't request more random iterations than the grid space allows
                try:
                    grid_size = int(np.prod([len(v) for v in grid.values()])) if grid else 1
                    n_iter = min(n_iter_per_model, max(1, grid_size))
                except Exception:
                    n_iter = n_iter_per_model

                # Cap CV folds again here for safety
                kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
                try:
                    search = RandomizedSearchCV(
                        est, grid, n_iter=n_iter, scoring="r2",
                        cv=kf, random_state=seed, n_jobs=-1
                    )
                    search.fit(Xtr, y_tr)
                    if np.isfinite(search.best_score_) and search.best_score_ > best_cv:
                        best_cv, best_tag, best_est = float(search.best_score_), tag, search.best_estimator_
                except Exception:
                    # model failed on this subset; skip it
                    continue

            # If no model succeeded for this subset, skip (don’t kill the run)
            if best_est is None:
                continue

            # ---------- Test performance ----------
            try:
                best_est.fit(Xtr, y_tr)
                y_pred = best_est.predict(Xte)
            except Exception:
                continue

            leaderboard.append({
                "position": pos,
                "seed": seed,
                "n_subsets": n_subsets,
                "subset_idx": subset_idx,
                "model": best_tag,
                "cvR2_mean": best_cv,
                "TEST_R2": r2_score(y_te, y_pred),
                "TEST_MAE": mean_absolute_error(y_te, y_pred),
                "TEST_RMSE": rmse(y_te, y_pred),
                "n_features": len(feat_cols),
                "bases": "|".join([f for f in feat_cols if f in bases]),
                "interactions": "|".join([f for f in feat_cols if f in inters]),
                "max_pred": float(np.max(y_pred)) if len(y_pred) else np.nan,
            })

        # ---------- Wrap up this seed ----------
        if not leaderboard:
            # Helpful diagnostics when this happens (common for small/dirty RB sets)
            raise RuntimeError(
                f"[{pos}] No valid models/subsets for seed={seed}.\n"
                f"- Train rows: {len(y_tr)} | CV folds attempted: {cv_splits}\n"
                f"- Base feature pool size: {len(base_pool)} | Allowed interactions: {len(allowed_inters)}\n"
                f"- Consider: lowering cv_folds, increasing test_size, loosening bans/musts, "
                f"or verifying RB CSV columns/aliases actually map to numeric values."
            )

        board = pd.DataFrame(leaderboard).sort_values(
            ["TEST_R2","cvR2_mean"], ascending=False
        ).head(15)
        board_path.parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(board_path, index=False)

        # Refit best config on train; evaluate + export artifacts
        best_row = board.iloc[0]
        best_bases = [c for c in best_row["bases"].split("|") if c]
        best_inters = [c for c in best_row["interactions"].split("|") if c]

        Xtr_df = X_tr_raw[best_bases].copy()
        Xte_df = X_te_raw[best_bases].copy()
        for iname in best_inters:
            a, b = INTERACTIONS[iname]
            if a in X_tr_raw.columns and b in X_tr_raw.columns:
                Xtr_df[iname] = X_tr_raw[a].values * X_tr_raw[b].values
                Xte_df[iname] = X_te_raw[a].values * X_te_raw[b].values

        # sanitize again for final fit
        Xtr_df, Xte_df = sanitize_train_test(Xtr_df, Xte_df)
        feat_cols = list(Xtr_df.columns)
        feature_kinds = {f: ("interaction" if f in best_inters else "base") for f in feat_cols}

        Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)
        imp = SimpleImputer(strategy="median")
        Xtr_arr, Xte_arr = imp.fit_transform(Xtr_df), imp.transform(Xte_df)

        # Re-tune within best class to squeeze a bit more
        tag_map = {row[0]: (row[1], row[2]) for row in model_spaces(seed)}
        est, grid = tag_map[best_row["model"]]
        try:
            grid_size = int(np.prod([len(v) for v in grid.values()])) if grid else 1
            n_iter_inner = min(n_iter_per_model, max(1, grid_size))
        except Exception:
            n_iter_inner = n_iter_per_model

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(est, grid, n_iter=n_iter_inner, scoring="r2", cv=kf, random_state=seed, n_jobs=-1)
        search.fit(Xtr_arr, y_tr)
        best_est = search.best_estimator_
        y_best = best_est.predict(Xte_arr)

        # Predictions file
        out_df = pd.DataFrame({
            "Player": n_te.values,
            f"Actual_{pos}_Grade": y_te.values,
            f"Predicted_{pos}_Grade": y_best,
            "Error": y_best - y_te.values
        }).sort_values(f"Actual_{pos}_Grade", ascending=False)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(pred_path, index=False)

        # Save the trained model as pickle file
        model_path = out_dir / f"{pos.lower()}_model_seed{seed}_subs{n_subsets}.pkl"
        model_metadata = {
            'model': best_est,
            'feature_names': feat_cols,
            'feature_kinds': feature_kinds,
            'imputer': imp,
            'best_bases': best_bases,
            'best_interactions': best_inters,
            'test_r2': float(r2_score(y_te, y_best)),
            'test_mae': float(mean_absolute_error(y_te, y_best)),
            'test_rmse': float(rmse(y_te, y_best)),
            'seed': seed,
            'n_subsets': n_subsets,
            'position': pos
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_metadata, f)
        print(f"💾 Saved model: {model_path}")

        # ---------------- SHAP per-player contributions + global importances ----------------
        shap_values = None
        shap_feature_importance = None
        extra_paths = {}
        base_cols = [c for c in feat_cols if feature_kinds[c] == "base"]
        inter_cols = [c for c in feat_cols if feature_kinds[c] == "interaction"]
        if enable_shap:
            try:
                sv, base_value = compute_shap(best_est, Xtr_arr, Xte_arr)
            except Exception:
                sv, base_value = None, 0.0
            if isinstance(sv, np.ndarray) and sv.size:
                shap_values = sv
                # Mean absolute SHAP per feature
                shap_mean_abs = np.mean(np.abs(sv), axis=0)
                feature_importances_dict = {feat: float(val) for feat, val in zip(feat_cols, shap_mean_abs)}
                shap_feature_importance = {
                    'feature_importances': feature_importances_dict,
                    'base_features_importance': {feat: float(feature_importances_dict[feat]) for feat in base_cols} if base_cols else {},
                    'interaction_features_importance': {feat: float(feature_importances_dict[feat]) for feat in inter_cols} if inter_cols else {},
                    'top_5_features': sorted(zip(feat_cols, shap_mean_abs), key=lambda x: x[1], reverse=True)[:5],
                    'base_importance_sum': float(sum(shap_mean_abs[i] for i, feat in enumerate(feat_cols) if feat in base_cols)) if base_cols else 0.0,
                    'interaction_importance_sum': float(sum(shap_mean_abs[i] for i, feat in enumerate(feat_cols) if feat in inter_cols)) if inter_cols else 0.0,
                    'shap_base_value': float(base_value)
                }

                # Save per-player SHAP contributions
                shap_contribs = pd.DataFrame(np.round(sv, 4), columns=feat_cols)
                shap_contribs["Base_SHAP_Sum"] = np.round(shap_contribs[base_cols].sum(axis=1), 4) if base_cols else 0.0
                shap_contribs["Interaction_SHAP_Sum"] = np.round(shap_contribs[inter_cols].sum(axis=1), 4) if inter_cols else 0.0
                shap_contribs.insert(0, "Player", n_te.values)
                shap_contribs.insert(1, "Bias", np.round(base_value, 4))
                shap_contribs[f"Predicted_{pos}_Grade"] = np.round(y_best, 4)
                shap_contribs[f"Actual_{pos}_Grade"] = np.round(y_te.values, 4)
                shap_contribs["Error"] = np.round(shap_contribs[f"Predicted_{pos}_Grade"] - shap_contribs[f"Actual_{pos}_Grade"], 4)
                shap_contribs.to_csv(shap_contribs_path, index=False, float_format="%.4f")
                extra_paths["shap_contribs_csv"] = str(shap_contribs_path)

        # ---------------- Save importances (SHAP/global/permutation) ----------------
        extra_paths.update(
            save_importances(
                out_dir=out_dir,
                pos=pos,
                seed=seed,
                n_subsets=n_subsets,
                feature_names=feat_cols,
                feature_kinds=feature_kinds,
                model=best_est,
                shap_values=shap_values,
                X_te_arr=Xte_arr,
                y_te=y_te.values if isinstance(y_te, pd.Series) else np.asarray(y_te),
                do_permutation=enable_permutation_importance,
                draft_cap_importance_cap=draft_cap_importance_cap,
                breakout_age_importance_cap=breakout_age_importance_cap,
                draft_age_importance_cap=draft_age_importance_cap,
            )
        )

        runtime_sec = time.perf_counter() - t0

        # ---------------- Build and save metadata JSON (after SHAP so it's included) ----------------
        best_meta_json = {
            'position': pos,
            'seed': seed,
            'n_subsets': n_subsets,
            'best_model_tag': best_row["model"],
            'best_hyperparams': search.best_params_ if 'search' in locals() else {},
            'performance': {
                'test_r2': float(r2_score(y_te, y_best)),
                'test_mae': float(mean_absolute_error(y_te, y_best)),
                'test_rmse': float(rmse(y_te, y_best)),
                'cv_r2_mean': float(best_row["cvR2_mean"]),
                'max_prediction': float(np.max(y_best))
            },
            'features': {
                'n_features': len(feat_cols),
                'feature_names': feat_cols,
                'feature_kinds': feature_kinds,
                'best_bases': best_bases,
                'best_interactions': best_inters
            },
            'shap_analysis': shap_feature_importance if shap_feature_importance is not None else {
                'note': 'SHAP analysis not available or disabled'
            },
            'data_info': {
                'train_samples': len(y_tr),
                'test_samples': len(y_te),
                'cv_folds_used': cv_splits,
                'total_samples': len(y_all)
            },
            'training_config': {
                'max_base_feats': max_base_feats,
                'max_interactions': max_interactions,
                'n_iter_per_model': n_iter_per_model,
                'test_size': test_size,
                'interaction_hierarchy': interaction_hierarchy,
                'draft_cap_cap': draft_cap_cap,
                'shap_enabled': enable_shap
            },
            'file_paths': {
                'model_pickle': str(model_path),
                'predictions_csv': str(pred_path),
                'leaderboard_csv': str(board_path),
                **extra_paths
            },
            'timestamp': datetime.datetime.now().isoformat(),
            'runtime_sec': runtime_sec
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(best_meta_json, f, indent=2)
        print(f"📄 Saved metadata with SHAP results: {meta_path}")

        # meta row for this seed
        rows.append({
            "position": pos,
            "seed": seed,
            "n_subsets": n_subsets,
            "best_test_R2": float(r2_score(y_te, y_best)),
            "best_test_MAE": float(mean_absolute_error(y_te, y_best)),
            "best_test_RMSE": float(rmse(y_te, y_best)),
            "runtime_sec": runtime_sec,
            "leaderboard_csv": str(board_path),
            "predictions_csv": str(pred_path),
            "metadata_json": str(meta_path),
            "model_pickle": str(model_path),
            "best_model_tag": best_row["model"],
            "best_bases": "|".join(best_bases),
            "best_interactions": "|".join(best_inters),
        })

        print(f"\n[{pos}] Seed {seed} | n_subsets={n_subsets} → "
              f"R²={rows[-1]['best_test_R2']:.4f} | MAE={rows[-1]['best_test_MAE']:.3f} | "
              f"RMSE={rows[-1]['best_test_RMSE']:.3f} | time={runtime_sec:.1f}s")

    # End seeds
    return pd.DataFrame(rows)

# ---------------------------- CLI ENTRYPOINT ----------------------------
if __name__ == "__main__":
    import argparse
    try:
        _HERE = Path(__file__).resolve()
        PROJECT_ROOT = _HERE.parents[2]
    except NameError:
        PROJECT_ROOT = Path.cwd()

    ap = argparse.ArgumentParser(description="Position-aware wide search (GB) + must/ban controls + hierarchy + DraftCap limiter + SHAP weights + NaN-safe.")
    ap.add_argument("--position", type=str, default="RB", help="RB, WR, TE, or QB")
    ap.add_argument("--seeds", type=int, nargs="*", default=[123,456,789], help="Seeds to run")
    ap.add_argument("--subset-grid", type=int, nargs="*", default=[10,20,40,60], help="Iteration counts (N_SUBSETS) to profile")
    ap.add_argument("--max-base-feats", type=int, default=8)
    ap.add_argument("--max-interactions", type=int, default=3)
    ap.add_argument("--n-iter-per-model", type=int, default=25)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.20)

    ap.add_argument("--must-feats", type=str, default="")
    ap.add_argument("--ban-feats",  type=str, default="")
    ap.add_argument("--must-inters",type=str, default="")
    ap.add_argument("--ban-inters", type=str, default="")
    ap.add_argument("--interaction-hierarchy", choices=["strong","weak","none"], default="weak")

    ap.add_argument("--draft-cap-cap", type=float, default=None)
    ap.add_argument("--draft-cap-lower-q", type=float, default=0.05)
    ap.add_argument("--draft-cap-upper-q", type=float, default=0.95)

    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--perm-importance", action="store_true")

    # Optional: cap reported importances for specific features
    ap.add_argument("--cap-imp-draftcap", type=float, default=None)
    ap.add_argument("--cap-imp-breakoutage", type=float, default=None)
    ap.add_argument("--cap-imp-draftage", type=float, default=None)

    args = ap.parse_args()

    def _csv_to_list(s: str) -> list[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    must_feats  = _csv_to_list(args.must_feats)
    ban_feats   = _csv_to_list(args.ban_feats)
    must_inters = _csv_to_list(args.must_inters)
    ban_inters  = _csv_to_list(args.ban_inters)

    all_results: list[pd.DataFrame] = []

    for n_subsets in args.subset_grid:
        res = run_seed_for_subsets(
            position=args.position,
            project_root=PROJECT_ROOT,
            n_subsets=n_subsets,
            seeds=args.seeds,
            max_base_feats=args.max_base_feats,
            max_interactions=args.max_interactions,
            n_iter_per_model=args.n_iter_per_model,
            cv_folds=args.cv_folds,
            test_size=args.test_size,
            must_feats=must_feats,
            ban_feats=ban_feats,
            must_inters=must_inters,
            ban_inters=ban_inters,
            interaction_hierarchy=args.interaction_hierarchy,
            draft_cap_cap=args.draft_cap_cap,
            draft_cap_lower_q=args.draft_cap_lower_q,
            draft_cap_upper_q=args.draft_cap_upper_q,
            enable_shap=not args.no_shap,
            enable_permutation_importance=args.perm_importance,
            draft_cap_importance_cap=args.cap_imp_draftcap,
            breakout_age_importance_cap=args.cap_imp_breakoutage,
            draft_age_importance_cap=args.cap_imp_draftage,
        )
        all_results.append(res)

    summary = pd.concat(all_results, ignore_index=True)
    out_dir = default_out_dir(PROJECT_ROOT, args.position)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.position.lower()}_runtime_accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)

    agg = (summary.groupby("n_subsets")
                   .agg(best_R2=("best_test_R2","max"),
                        avg_runtime_s=("runtime_sec","mean"),
                        runs=("best_test_R2","count"))
                   .reset_index())
    print("\n=== Accuracy vs Runtime (by n_subsets) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved detailed summary → {summary_path}")
