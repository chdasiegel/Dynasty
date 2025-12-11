# src/models/gradientboosting_custom_rank.py
# ===============================================================
# Position-aware Rank-Guided Grade — wide search over feature subsets & hypers
#
# - position: RB / WR / TE / QB
# - Input CSVs: data/Rankings/ranked_by_position/master_list_with_ranks_{pos}.csv
# - Ranks are used only as supervision labels for training.
#   The model learns a continuous target:
#       Train_Target = -Rank_raw
#   (lower rank → higher target).
# - Final output: raw model prediction on the Train_Target scale.
# - Randomly samples feature combos + interactions per run
# - Model: GradientBoostingRegressor (GB)
# - Must/Ban feature & interaction controls + hierarchy (strong/weak/none)
# - SHAP per-player breakdown + global importances
# - NaN-safe; spits out leaderboards, models, metadata, etc.
# ===============================================================
from __future__ import annotations

import json, re, time, warnings, datetime, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prefer the sklearn SimpleImputer, but keep a small local fallback around
# so the code is still analyzable / importable in thin environments.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklearn.impute import SimpleImputer  # type: ignore

try:
    from sklearn.impute import SimpleImputer  # type: ignore
except Exception:
    import numpy as _np
    class SimpleImputer:
        """
        SimpleImputer that only supports strategy='median'.
        Handles 1D/2D inputs and replaces non-finite values with the
        per-column median.
        """
        def __init__(self, strategy="median"):
            if strategy != "median":
                raise ValueError("SimpleImputer only supports strategy='median'")
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

# ---------------- sklearn model_selection ----------------
if TYPE_CHECKING:
    from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score  # type: ignore

try:
    import importlib
    _sklearn_ms = importlib.import_module("sklearn.model_selection")
    train_test_split = _sklearn_ms.train_test_split
    KFold = _sklearn_ms.KFold
    RandomizedSearchCV = _sklearn_ms.RandomizedSearchCV
    cross_val_score = _sklearn_ms.cross_val_score
except Exception:
    # If sklearn isn't installed, fail loudly when someone actually tries to run
    def train_test_split(*args, **kwargs):
        raise RuntimeError("scikit-learn is required for train_test_split.")
    class KFold:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("scikit-learn is required for KFold.")
    class RandomizedSearchCV:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("scikit-learn is required for RandomizedSearchCV.")
    def cross_val_score(*args, **kwargs):
        raise RuntimeError("scikit-learn is required for cross_val_score.")

# ---------------- sklearn metrics ----------------
try:
    import importlib
    _sklearn_metrics = importlib.import_module("sklearn.metrics")
    r2_score = _sklearn_metrics.r2_score
    mean_absolute_error = _sklearn_metrics.mean_absolute_error
    mean_squared_error = _sklearn_metrics.mean_squared_error
except Exception:
    def r2_score(*args, **kwargs):
        raise RuntimeError("scikit-learn metrics.r2_score is required.")
    def mean_absolute_error(*args, **kwargs):
        raise RuntimeError("scikit-learn metrics.mean_absolute_error is required.")
    def mean_squared_error(*args, **kwargs):
        raise RuntimeError("scikit-learn metrics.mean_squared_error is required.")

# ---------------- sklearn ensemble ----------------
try:
    import importlib
    _sklearn_ensemble = importlib.import_module("sklearn.ensemble")
    GradientBoostingRegressor = _sklearn_ensemble.GradientBoostingRegressor
except Exception:
    class GradientBoostingRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("scikit-learn ensemble.GradientBoostingRegressor is required.")

# ---------------- sklearn inspection ----------------
try:
    import importlib
    _sklearn_inspect = importlib.import_module("sklearn.inspection")
    permutation_importance = _sklearn_inspect.permutation_importance
except Exception:
    def permutation_importance(*args, **kwargs):
        raise RuntimeError("scikit-learn inspection.permutation_importance is required.")

# ---------------- shap ----------------
try:
    import importlib
    shap = importlib.import_module("shap")
    _HAVE_SHAP = True
except Exception:
    shap = None
    _HAVE_SHAP = False

# ----------------------------- Features / utils -----------------------------
from src.utils import BASE_FEATURES, INTERACTIONS, ALIASES
from src.utils import default_out_dir
from src.utils import find_col, to_num, rmse, invert_cols, limit_draft_capital_series
from src.utils import sanitize_frame_global, sanitize_train_test, sample_subset
from src.analysis_utils import compute_shap, save_importances


# ---------------------------- Model Spaces -----------------------------
def model_spaces(random_state: int):
    """
    Define the model(s) and hyperparameter grids to search over.
    Currently its only GradientBoostingRegressor with flexibility to add more
    models if desired.
    """
    return [
        ("GB", GradientBoostingRegressor(random_state=random_state), {
            "n_estimators": [200, 400, 600, 800],
            "learning_rate": [0.05, 0.1, 0.15],
            "max_depth": [2, 3, 4, 5, 6, 7],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 3, 5],
        }),
    ]


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
    draft_cap_cap: float | None = None,
    draft_cap_lower_q: float = 0.05,
    draft_cap_upper_q: float = 0.95,
    enable_shap: bool = True,
    enable_permutation_importance: bool = False,
    draft_cap_importance_cap: float | None = None,
    breakout_age_importance_cap: float | None = None,
    draft_age_importance_cap: float | None = None,
    must_use_features: list[str] | None = None,
    banned_features: list[str] | None = None,
    must_use_interactions: list[str] | None = None,
    banned_interactions: list[str] | None = None,
    priority_strength: str | None = None
) -> pd.DataFrame:
    """
    Primary function for the rank-guided GB search.

    For a given position (RB/WR/TE/QB) and number of random feature subsets,
    this:
      - Loads the ranked-by-position CSV
      - Builds Train_Target = -Rank_raw (higher is better)
      - Samples feature + interaction subsets per loop
      - Tunes GB on each subset via RandomizedSearchCV
      - Tracks a leaderboard on held-out test data
      - Saves the best config + model + SHAP + importances per seed

    Returns a summary DataFrame, one row per (seed, n_subsets) run.
    """

    # ---- Normalize older argument names into the current ones ------------------
    if must_feats is None:            must_feats = must_use_features
    if ban_feats is None:             ban_feats = banned_features
    if must_inters is None:           must_inters = must_use_interactions
    if ban_inters is None:            ban_inters = banned_interactions
    if priority_strength is not None: interaction_hierarchy = priority_strength

    must_feats  = list(must_feats  or [])
    ban_feats   = list(ban_feats   or [])
    must_inters = list(must_inters or [])
    ban_inters  = list(ban_inters  or [])

    # ---- SHAP availability flag (only really matters at runtime) --------------
    try:
        shap_enabled = bool(enable_shap)
        if 'shap' in globals():
            pass
    except Exception:
        shap_enabled = False

    pos = position.upper()
    out_dir = default_out_dir(project_root, pos)

    # ranked-by-position CSV
    ranked_dir = project_root / "data" / "Rankings" / "ranked_by_position"
    csv_path = ranked_dir / f"master_list_with_ranks_{pos}.csv"

    # ---- Quick debug printout --------------------------------------------------
    print("\n========== DEBUG PATHS ==========")
    print(f"Position          : {pos}")
    print(f"CSV path          : {csv_path}")
    print(f"Output directory  : {out_dir}")
    print(f"Hierarchy         : {interaction_hierarchy}")
    print(f"Must feats        : {must_feats}")
    print(f"Ban feats         : {ban_feats}")
    print(f"Must inters       : {must_inters}")
    print(f"Ban inters        : {ban_inters}")
    print(f"n_subsets         : {n_subsets}")
    print(f"max_base_feats    : {max_base_feats}")
    print(f"n_iter_per_model  : {n_iter_per_model}")
    if draft_cap_cap is not None:
        print(f"DraftCap limiter  : cap={draft_cap_cap}, lower_q={draft_cap_lower_q}, upper_q={draft_cap_upper_q}")
    print("=================================\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data -------------------------------------------------------------
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Rank supervision: keep rank as reference, train on its negative
    rank_col = find_col(df, ["rank", "Rank", "RANK"])
    if not rank_col:
        raise ValueError(f"Could not find 'rank' column in {csv_path}")

    df["Rank_raw"] = to_num(df[rank_col])

    if df["Rank_raw"].isna().all():
        raise ValueError(f"All rank values are NaN in {csv_path}")

    # Train target: -rank → higher is better
    df["Train_Target"] = -df["Rank_raw"]
    y_col = "Train_Target"

    # Try the usual suspects for player name, fall back to first column if needed
    name_col = find_col(df, ["Player","Player Name","Name","player_name","player"]) or df.columns[0]

    # Map canonical features to actual CSV columns via ALIASES
    mapped: dict[str, str] = {}
    for feat in BASE_FEATURES:
        col = find_col(df, ALIASES.get(feat, [feat]))
        if col is not None:
            mapped[feat] = col
    if not mapped:
        raise ValueError("No usable base features found in CSV for this position.")

    # ---- Build feature pools w/ must/ban logic --------------------------------
    ban_feats_set = {f.strip() for f in ban_feats}
    must_feats_list = [f for f in must_feats if f in mapped]
    base_pool = [f for f in mapped.keys() if f not in ban_feats_set]

    # Precompute which interactions we’re allowed to use
    allowed_inters: dict[str, tuple[str, str]] = {}
    for iname, (a, b) in INTERACTIONS.items():
        if iname in ban_inters:
            continue
        if a in mapped and b in mapped and a not in ban_feats_set and b not in ban_feats_set:
            allowed_inters[iname] = (a, b)
    must_inters_list = [n for n in must_inters if n in allowed_inters]

    # If hierarchy == "none", we can allow interaction-only features and drop parents
    if interaction_hierarchy == "none":
        parents_to_block = set()
        for iname in must_inters_list:
            a, b = allowed_inters[iname]
            parents_to_block.update([a, b])
        base_pool = [f for f in base_pool if (f not in parents_to_block) or (f in must_feats_list)]

    # ---- Turn everything numeric + global cleaning ----------------------------
    X_all_raw = pd.DataFrame({feat: to_num(df[col]) for feat, col in mapped.items()})
    X_all_raw = sanitize_frame_global(X_all_raw, min_non_na_frac=0.0)

    # Realign pools to whatever survived sanitization
    available_feats = set(X_all_raw.columns)
    base_pool = [f for f in base_pool if f in available_feats]
    must_feats_list = [f for f in must_feats_list if f in available_feats]
    allowed_inters = {
        name: (a, b)
        for name, (a, b) in allowed_inters.items()
        if (a in available_feats) and (b in available_feats)
    }
    must_inters_list = [n for n in must_inters_list if n in allowed_inters]

    y_all     = to_num(df[y_col])
    names_all = df[name_col].astype(str).fillna("")

    # Keep only rows where the target exists
    mask = y_all.notna()
    X_all_raw, y_all, names_all, df = (
        X_all_raw.loc[mask].reset_index(drop=True),
        y_all.loc[mask].reset_index(drop=True),
        names_all.loc[mask].reset_index(drop=True),
        df.loc[mask].reset_index(drop=True),
    )

    # Optionally clip / drop weird Draft Capital values
    if "Draft Capital" in X_all_raw.columns:
        keep = X_all_raw["Draft Capital"] != 0
        X_all_raw = X_all_raw.loc[keep].reset_index(drop=True)
        y_all     = y_all.loc[keep].reset_index(drop=True)
        names_all = names_all.loc[keep].reset_index(drop=True)
        df        = df.loc[keep].reset_index(drop=True)
        if draft_cap_cap is not None:
            X_all_raw["Draft Capital"] = limit_draft_capital_series(
                X_all_raw["Draft Capital"],
                cap=draft_cap_cap,
                lower_q=draft_cap_lower_q,
                upper_q=draft_cap_upper_q,
            )

    # Quick sanity checks
    print(f"[{pos}] Rows after filtering: {len(y_all)} | Feature cols: {X_all_raw.shape[1]}")
    if len(y_all) < 10:
        print(f"[{pos}] WARNING: very small sample (n={len(y_all)}). Consider lowering cv_folds.")
    if y_all.nunique(dropna=True) < 2:
        raise ValueError(f"[{pos}] Target has <2 unique Train_Target values after filtering — cannot train.")

    rows: list[dict] = []

    # ---- Run the full search for each seed ------------------------------------
    for seed in seeds:
        t0 = time.perf_counter()

        # Bake hyperparams into filenames so different runs don't stomp each other
        suffix = f"_seed{seed}_subs{n_subsets}_base{max_base_feats}_iter{n_iter_per_model}"

        pred_path  = out_dir / f"{pos.lower()}_wide_best_preds{suffix}.csv"
        meta_path  = out_dir / f"{pos.lower()}_wide_best_meta{suffix}.json"
        board_path = out_dir / f"{pos.lower()}_wide_leaderboard{suffix}.csv"
        shap_contribs_path = out_dir / f"{pos.lower()}_shap_contribs{suffix}.csv"

        rng = np.random.default_rng(seed)

        # Standard train/test split — keep Rank_raw and names in sync
        X_tr_raw, X_te_raw, y_tr, y_te, n_tr, n_te, ranks_tr, ranks_te = train_test_split(
            X_all_raw,
            y_all,
            names_all,
            df["Rank_raw"],
            test_size=test_size,
            random_state=seed
        )
        if len(y_tr) < 3:
            raise ValueError(f"[{pos}] After split (seed={seed}), train size={len(y_tr)} is too small to CV.")

        cv_splits = min(cv_folds, int(len(y_tr)))
        if cv_splits < 2:
            cv_splits = 2

        leaderboard: list[dict] = []
        baseline_done = False

        for subset_idx in range(n_subsets):
            # ---------- Choose a feature subset ----------
            if not baseline_done:
                if must_feats_list:
                    # If caller forced some features, start with those
                    bases = list(dict.fromkeys(must_feats_list))[:max_base_feats]
                    inters = list(dict.fromkeys(must_inters_list))[:max_interactions]
                    baseline_done = True
                else:
                    # Baseline: "best" single feature by CV R²
                    best_feat, best_cv = None, -1e18
                    for f in [fname for fname in base_pool if fname in X_tr_raw.columns]:
                        if X_tr_raw[f].notna().sum() == 0:
                            continue
                        imp = SimpleImputer(strategy="median")
                        X_single = imp.fit_transform(X_tr_raw[[f]])
                        if not np.isfinite(X_single).any():
                            continue
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
                        # If even that fails, just skip this subset
                        continue
            else:
                # Main random sampling path
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

            # ---------- Assemble design matrices ----------
            Xtr_df, Xte_df = X_tr_raw[bases].copy(), X_te_raw[bases].copy()

            for iname in inters:
                a, b = allowed_inters[iname]
                if a in X_tr_raw.columns and b in X_tr_raw.columns:
                    Xtr_df[iname] = X_tr_raw[a].values * X_tr_raw[b].values
                    Xte_df[iname] = X_te_raw[a].values * X_te_raw[b].values

            Xtr_df, Xte_df = sanitize_train_test(Xtr_df, Xte_df)
            if Xtr_df.shape[1] == 0:
                continue

            feat_cols = list(Xtr_df.columns)

            Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)
            imp = SimpleImputer(strategy="median")
            try:
                Xtr = imp.fit_transform(Xtr_df)
                Xte = imp.transform(Xte_df)
            except Exception:
                continue

            # ---------- Model selection on TRAIN (CV) ----------
            best_cv, best_tag, best_est = -1e18, None, None
            for tag, est, grid in model_spaces(seed):
                try:
                    grid_size = int(np.prod([len(v) for v in grid.values()])) if grid else 1
                    n_iter = min(n_iter_per_model, max(1, grid_size))
                except Exception:
                    n_iter = n_iter_per_model

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
                    continue

            if best_est is None:
                continue

            # ---------- Evaluate on held-out test ----------
            try:
                best_est.fit(Xtr, y_tr)
                y_pred = best_est.predict(Xte)
            except Exception:
                continue

            leaderboard.append({
                "position": pos,
                "seed": seed,
                "n_subsets": n_subsets,
                "max_base_feats": max_base_feats,
                "n_iter_per_model": n_iter_per_model,
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

        # ---------- Finish this seed --------------------------------------------
        if not leaderboard:
            raise RuntimeError(
                f"[{pos}] No valid models/subsets for seed={seed}.\n"
                f"- Train rows: {len(y_tr)} | CV folds attempted: {cv_splits}\n"
                f"- Base feature pool size: {len(base_pool)} | Allowed interactions: {len(allowed_inters)}\n"
            )

        board = (pd.DataFrame(leaderboard)
                   .sort_values(["TEST_R2","cvR2_mean"], ascending=False)
                   .head(15))
        board_path.parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(board_path, index=False)

        # Refit best config on train; collect artifacts
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

        Xtr_df, Xte_df = sanitize_train_test(Xtr_df, Xte_df)
        feat_cols = list(Xtr_df.columns)
        feature_kinds = {f: ("interaction" if f in best_inters else "base") for f in feat_cols}

        Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)
        imp = SimpleImputer(strategy="median")
        Xtr_arr, Xte_arr = imp.fit_transform(Xtr_df), imp.transform(Xte_df)

        # One more round of tuning restricted to the winning model family
        tag_map = {row[0]: (row[1], row[2]) for row in model_spaces(seed)}
        est, grid = tag_map[best_row["model"]]
        try:
            grid_size = int(np.prod([len(v) for v in grid.values()])) if grid else 1
            n_iter_inner = min(n_iter_per_model, max(1, grid_size))
        except Exception:
            n_iter_inner = n_iter_per_model

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(
            est, grid, n_iter=n_iter_inner, scoring="r2",
            cv=kf, random_state=seed, n_jobs=-1
        )
        search.fit(Xtr_arr, y_tr)
        best_est = search.best_estimator_
        y_best = best_est.predict(Xte_arr)  # still on Train_Target scale

        # ---------- Keep raw outputs on the Train_Target scale ------------------
        out_df = pd.DataFrame({
            "Player": n_te.values,
            f"{pos}_Rank": ranks_te.values,
            "Train_Target_raw": y_te.values,
            f"Model_Output_raw_{pos}": y_best,
            "Error_Target": y_best - y_te.values
        }).sort_values(f"Model_Output_raw_{pos}", ascending=False)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(pred_path, index=False)

        # Save model + metadata bundle to reload this later
        model_path = out_dir / f"{pos.lower()}_model{suffix}.pkl"
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
            'max_base_feats': max_base_feats,
            'n_iter_per_model': n_iter_per_model,
            'position': pos
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_metadata, f)
        print(f"💾 Saved model: {model_path}")

        # ---------------- SHAP per-player contributions + global importances ----
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

                shap_contribs = pd.DataFrame(np.round(sv, 4), columns=feat_cols)
                shap_contribs["Base_SHAP_Sum"] = np.round(shap_contribs[base_cols].sum(axis=1), 4) if base_cols else 0.0
                shap_contribs["Interaction_SHAP_Sum"] = np.round(shap_contribs[inter_cols].sum(axis=1), 4) if inter_cols else 0.0
                shap_contribs.insert(0, "Player", n_te.values)
                shap_contribs.insert(1, "Bias", np.round(base_value, 4))
                shap_contribs[f"Model_Output_raw_{pos}"] = np.round(y_best, 4)
                shap_contribs["Train_Target_raw"] = np.round(y_te.values, 4)
                shap_contribs["Error_Target"] = np.round(
                    shap_contribs[f"Model_Output_raw_{pos}"] - shap_contribs["Train_Target_raw"], 4
                )
                shap_contribs.to_csv(shap_contribs_path, index=False, float_format="%.4f")
                extra_paths["shap_contribs_csv"] = str(shap_contribs_path)

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

        best_meta_json = {
            'position': pos,
            'seed': seed,
            'n_subsets': n_subsets,
            'max_base_feats': max_base_feats,
            'n_iter_per_model': n_iter_per_model,
            'best_model_tag': best_row["model"],
            'best_hyperparams': search.best_params_ if 'search' in locals() else {},
            'performance': {
                'test_r2': float(r2_score(y_te, y_best)),
                'test_mae': float(mean_absolute_error(y_te, y_best)),
                'test_rmse': float(rmse(y_te, y_best)),
                'cv_r2_mean': float(best_row["cvR2_mean"]),
                'max_prediction_raw': float(np.max(y_best)),
                'min_prediction_raw': float(np.min(y_best)),
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

        rows.append({
            "position": pos,
            "seed": seed,
            "n_subsets": n_subsets,
            "max_base_feats": max_base_feats,
            "n_iter_per_model": n_iter_per_model,
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

        print(
            f"\n[{pos}] Seed {seed} | n_subsets={n_subsets} | "
            f"max_base_feats={max_base_feats} | n_iter_per_model={n_iter_per_model} → "
            f"R²={rows[-1]['best_test_R2']:.4f} | MAE={rows[-1]['best_test_MAE']:.3f} | "
            f"RMSE={rows[-1]['best_test_RMSE']:.3f} | time={runtime_sec:.1f}s"
        )

    return pd.DataFrame(rows)


# ---------------------------- CLI ENTRYPOINT ----------------------------
if __name__ == "__main__":
    import argparse
    try:
        _HERE = Path(__file__).resolve()
        PROJECT_ROOT = _HERE.parents[2]
    except NameError:
        PROJECT_ROOT = Path.cwd()

    ap = argparse.ArgumentParser(
        description="Position-aware wide search (GB) using ranks as supervision."
    )
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

    ap.add_argument("--cap-imp-draftcap", type=float, default=None)
    ap.add_argument("--cap-imp-breakoutage", type=float, default=None)
    ap.add_argument("--cap-imp-draftage", type=float, default=None)

    args = ap.parse_args()

    def _csv_to_list(s: str) -> list[str]:
        """
        Quick helper to turn 'a,b,c' into ['a','b','c'], trimming whitespace.
        """
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

    # Group by n_subsets + hyperparameters so I can see the tradeoffs at a glance
    agg = (
        summary
        .groupby(["n_subsets", "max_base_feats", "n_iter_per_model"])
        .agg(
            best_R2=("best_test_R2","max"),
            avg_runtime_s=("runtime_sec","mean"),
            runs=("best_test_R2","count")
        )
        .reset_index()
    )
    print("\n=== Accuracy vs Runtime (by n_subsets, max_base_feats, n_iter_per_model) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved detailed summary → {summary_path}")
