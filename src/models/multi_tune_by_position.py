# src/models/multi_tune_by_position.py
# ===============================================================
# Position-aware Grade — wide search over feature subsets & hypers
# - position: RB / WR / TE / QB (affects CSV path + target names)
# - Randomly sample feature combos (+ limited interactions)
# - Model: GradientBoostingRegressor (GB)
# - Must/ban features & interactions + hierarchy control
#   * strong: parents required for ALL interactions
#   * weak  : parents required for must_inters only
#   * none  : allow interaction-only; optionally strip parents from bases
# - 80/20 split; CV on TRAIN only; metrics on TEST
# - Saves leaderboards, predictions, global importances, SHAP per-player CSV,
#   overall summary CSV, Pareto chart
# ===============================================================
from __future__ import annotations

import json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# ----------------------------- Features -----------------------------
BASE_FEATURES: list[str] = [
    "DOM+","Speed","BMI","MTF/A","YPC","YPR","ELU","YCO/A",
    "Break%","Draft Capital","Conference Rank","Draft Age","Breakout Age",
    "Y/RR","YAC/R","aDOT","EPA/P","aYPTPA","CTPRR","UCTPRR","Drop Rate","CC%","Wide%","Slot%",
    "Comp%","ADJ%","BTT%","TWP%","DAA","YPA"
]

# Interaction library (add more as you like)
INTERACTIONS: dict[str, tuple[str, str]] = {
    "SpeedxBMI":   ("Speed","BMI"),
    "Wide%xSlot%": ("Wide%","Slot%"),
}

ALIASES: dict[str, list[str]] = {
    "DOM+":             ["DOM+","DOMp","DOM_plus","DOMp_Weighted","DOM","DOM++"],
    "Speed":            ["40 Time","Forty","40","Speed"],
    "BMI":              ["BMI","Body Mass Index"],
    "MTF/A":            ["Missed Tackles Forced Per Attempt","MTFA","MTF/A","Missed Tackles/Att"],
    "YPC":              ["YPC","Yards per Carry","Yards/Carry","Rushing YPC"],
    "YPR":              ["YPR","Yards Per Reception","Yards/Reception"],
    "ELU":              ["ELU","Elusiveness","Elusiveness Rating"],
    "YCO/A":            ["YCO/A","YAC/A","Yards After Contact / Att","Yards After Contact per Attempt"],
    "Break%":           ["Break%","Break %","Breakaway %","Breakaway Percentage","Breakaway%"],
    "Draft Capital":    ["Draft Capital","Draft Cap","Draft Round","Round","Rnd"],
    "Conference Rank":  ["ConRK","Conference Rank","Conf Rk"],
    "Shuttle":          ["Shuttle","Short Shuttle","20 Shuttle","20 Yard Shuttle"],
    "Three Cone":       ["3 Cone","Three Cone","3-Cone"],
    "Rec Yards":        ["Receiving Yards","Rec Yds","RecYds"],
    "Draft Age":        ["Draft Age","Age at Draft","DraftAge","Age (Draft)","AgeDraft","Age_at_Draft"],
    "Breakout Age":     ["Breakout","Breakout Age","Age of Breakout","BOUT"],
    "Y/RR":             ["Yards Route Run","Yards per Route Run","Y/RR","YPRR"],
    "YAC/R":            ["Yards After Catch Per Reception","YAC/R","YAC per Rec"],
    "aDOT":             ["Average Depth of Target","aDOT","ADOT","adot"],
    "EPA/P":            ["Expected Points Added per Play","EPA/P","EPA per Play"],
    "aYPTPA":           ["Adjusted Yards per Team Pass Attempt","aYPTPA","AYPTA","aypta"],
    "CTPRR":            ["Targets Per Route Run","CTPRR"],
    "UCTPRR":           ["Uncontested Targets Per Route Run","UCTPRR"],
    "Drop Rate":        ["Drop %","Drop%","DropRate","Drop Rate"],
    "CC%":              ["Contested Catch Percent","CC","CC %","CC%"],
    "Wide%":            ["X/Z%","X%","Z%","Wide","Wide %","Wide%"],
    "Slot%":            ["Slot","Slot %","Slot%"],
    "Comp%":            ["Completion Percentage","Comp","COMP","Comp%"],
    "ADJ%":             ["Adjusted Completed Percentage","ADJ","ADJ%","ADJ %"],
    "BTT%":             ["Big Time Throw Percentage","BTT","BTT %","BTT%"],
    "TWP%":             ["Turnover Worthy Play Percentage","TWP","TWP %","TWP%"],
    "DAA":              ["Depth-Adjusted Accuracy","Depth Adjusted Accuracy","DAA"],
    "YPA":              ["Yards Per Attempt","YpA","YPA"]
}

# ------------------------- POSITION UTILS -------------------------
def target_cands_for_position(pos: str) -> list[str]:
    p = pos.upper()
    return [f"{p} Grade", f"{p}Grade", f"{p}_Grade"]

def default_csv_for_position(project_root: Path, pos: str) -> Path:
    return project_root / "data" / "Bakery" / pos.upper() / f"Bakery_{pos.upper()}_Overall.csv"

def default_out_dir(project_root: Path, pos: str) -> Path:
    return project_root / "data" / "Bakery" / "_derived" / pos.upper()

# ---------------------------- HELPERS -----------------------------
def find_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    norm = {re.sub(r"\s+","",c).lower(): c for c in frame.columns}
    for cand in candidates:
        key = re.sub(r"\s+","",cand).lower()
        if key in norm:
            return norm[key]
    return None

def to_num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = (s.str.replace('%','',regex=False)
           .str.replace(r'(?i)round\s*','',regex=True)
           .str.replace(r'(?i)^r\s*','',regex=True)
           .str.replace(r'(?i)(st|nd|rd|th)$','',regex=True)
           .str.replace(',','',regex=False)
           .str.replace(r'[^0-9\.\-]','',regex=True))
    return pd.to_numeric(s, errors='coerce')

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def invert_cols(X: pd.DataFrame) -> pd.DataFrame:
    # invert "lower is better" signal
    X = X.copy()
    for c in ["40 Time","Speed","Draft Capital","Shuttle","Three Cone","Draft Age"]:
        if c in X.columns:
            X[c] = -X[c]
    return X

def model_spaces(random_state: int):
    """Only GradientBoostingRegressor (GB)."""
    return [
        ("GB", GradientBoostingRegressor(random_state=random_state), {
            "n_estimators": [400, 600, 800, 1000, 1200],
            "learning_rate": [0.03, 0.05, 0.07, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 3],
        }),
    ]

# ---------------------- PARETO CHART HELPER ----------------------
def save_pareto_chart(summary: pd.DataFrame, position: str, out_dir: Path) -> Path:
    """Save Accuracy (best R² per n_subsets) vs avg runtime (seconds)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = (summary.groupby("n_subsets")
                   .agg(best_R2=("best_test_R2","max"),
                        avg_runtime_s=("runtime_sec","mean"),
                        runs=("best_test_R2","count"))
                   .reset_index())
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax.scatter(agg["avg_runtime_s"], agg["best_R2"], s=80, alpha=0.9)
    for _, row in agg.iterrows():
        ax.annotate(int(row["n_subsets"]),
                    (row["avg_runtime_s"], row["best_R2"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9)
    ordered = agg.sort_values("n_subsets")
    ax.plot(ordered["avg_runtime_s"], ordered["best_R2"], linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Average runtime per setting (seconds)")
    ax.set_ylabel("Best Test R² (across seeds)")
    ax.set_title(f"{position.upper()}: Accuracy vs Runtime by N_SUBSETS (Pareto view)")
    ax.grid(alpha=0.2)
    png_path = out_dir / f"{position.lower()}_accuracy_vs_runtime.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return png_path

# -------------------- SAMPLER W/ HIERARCHY CONTROL --------------------
def sample_subset(
    rng: np.random.Generator,
    *,
    base_pool: list[str],
    max_bases: int,
    allowed_inters: dict[str, tuple[str,str]],
    max_inters: int,
    must_feats: list[str],
    must_inters: list[str],
    interaction_hierarchy: str = "weak",  # "strong" | "weak" | "none"
    drop_parents_when_none: bool = True,  # if True and hierarchy="none", strip parents from bases
) -> tuple[list[str], list[str]]:
    """
    Returns (bases, inter_names) sampled subject to:
      - must_feats always included
      - must_inters always included (subject to hierarchy)
      - interaction_hierarchy:
          strong: parents required for ALL interactions
          weak  : parents required only for must_inters
          none  : allow interaction-only (parents not required)
      - If hierarchy='none' and drop_parents_when_none=True,
        remove parent features from `bases` (unless explicitly in must_feats).
    """
    must_feats = list(dict.fromkeys(must_feats))

    # sample additional bases
    remaining_pool = [f for f in base_pool if f not in must_feats]
    extra_cap = max(0, max_bases - len(must_feats))
    extras = []
    if remaining_pool and extra_cap > 0:
        k = int(rng.integers(0, min(extra_cap, len(remaining_pool)) + 1))
        if k > 0:
            extras = rng.choice(remaining_pool, size=k, replace=False).tolist()
    bases = must_feats + extras

    # choose interactions: must + random others (filter to allowed)
    inter_names = [n for n in must_inters if n in allowed_inters]
    other_eligible = [n for n in allowed_inters if n not in set(inter_names)]
    other_cap = max(0, max_inters - len(inter_names))
    if other_eligible and other_cap > 0:
        k = int(rng.integers(0, min(other_cap, len(other_eligible)) + 1))
        if k > 0:
            inter_names += rng.choice(other_eligible, size=k, replace=False).tolist()

    # force parents if needed (strong/weak)
    if interaction_hierarchy in ("strong", "weak"):
        force_for = list(allowed_inters.keys()) if interaction_hierarchy == "strong" else list(inter_names)
        for iname in force_for:
            if iname in allowed_inters:
                a, b = allowed_inters[iname]
                for p in (a, b):
                    if p not in bases and p in base_pool:
                        bases.append(p)

        # keep only interactions whose parents are present
        inter_names = [n for n in inter_names if all(p in bases for p in allowed_inters[n])]

    # if hierarchy == "none", optionally strip parents from bases (unless must_feat)
    if interaction_hierarchy == "none" and drop_parents_when_none:
        parent_set = set()
        for iname in inter_names:
            a, b = allowed_inters[iname]
            parent_set.update([a, b])
        # preserve parents explicitly forced via must_feats
        keep_parents = set(must_feats)
        bases = [f for f in bases if (f not in parent_set) or (f in keep_parents)]

    # truncate to requested caps
    if len(bases) > max_bases:
        # always keep must_feats at front
        locked = list(dict.fromkeys([f for f in bases if f in must_feats]))
        rest   = [f for f in bases if f not in locked]
        bases  = locked + rest[:max(0, max_bases - len(locked))]
    if len(inter_names) > max_inters:
        must_locked = [n for n in inter_names if n in must_inters]
        rest = [n for n in inter_names if n not in must_locked]
        inter_names = must_locked + rest[:max(0, max_inters - len(must_locked))]

    return bases, inter_names

# ---------------------------- CORE RUN ----------------------------
def run_seed_for_subsets(position: str,
                         project_root: Path,
                         n_subsets: int,
                         *,
                         seeds: list[int],
                         max_base_feats: int,
                         max_interactions: int,
                         n_iter_per_model: int,
                         cv_folds: int,
                         test_size: float = 0.20,
                         # canonical args
                         must_feats: list[str] | None = None,
                         ban_feats: list[str] | None = None,
                         must_inters: list[str] | None = None,
                         ban_inters: list[str] | None = None,
                         interaction_hierarchy: str = "weak",
                         drop_parents_when_none: bool = True,
                         # backward-compat synonyms
                         must_use_features: list[str] | None = None,
                         banned_features: list[str] | None = None,
                         must_use_interactions: list[str] | None = None,
                         banned_interactions: list[str] | None = None,
                         priority_strength: str | None = None
                         ) -> pd.DataFrame:
    """
    Run the wide-search once per seed for a given n_subsets,
    returning a summary row per seed with runtime + best test metrics.
    Also saves:
      - <pos>_feature_importance_seed{seed}_subs{n}.csv
      - <pos>_shap_contributions_seed{seed}_subs{n}.csv (if shap available)
    """

    # ---- normalize legacy argument names ----
    if must_feats is None:            must_feats = must_use_features
    if ban_feats is None:             ban_feats = banned_features
    if must_inters is None:           must_inters = must_use_interactions
    if ban_inters is None:            ban_inters = banned_interactions
    if priority_strength is not None: interaction_hierarchy = priority_strength  # alias

    # default to empty lists if still None
    must_feats = list(must_feats or [])
    ban_feats  = list(ban_feats  or [])
    must_inters= list(must_inters or [])
    ban_inters = list(ban_inters or [])

    pos = position.upper()
    out_dir = default_out_dir(project_root, pos)
    csv_path = default_csv_for_position(project_root, pos)

    # DEBUG PATHS
    print("\n========== DEBUG PATHS ==========")
    print(f"Position          : {pos}")
    print(f"CSV path          : {csv_path}")
    print(f"Output directory  : {out_dir}")
    print(f"Hierarchy         : {interaction_hierarchy} | drop_parents_when_none={drop_parents_when_none}")
    print(f"Must feats        : {must_feats}")
    print(f"Ban feats         : {ban_feats}")
    print(f"Must inters       : {must_inters}")
    print(f"Ban inters        : {ban_inters}")
    print("=================================\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Target + names
    y_col = None
    for cand in target_cands_for_position(pos):
        y_col = find_col(df, [cand])
        if y_col: break
    if not y_col:
        raise ValueError(f"Could not find target column for {pos} among {target_cands_for_position(pos)}")
    name_col = find_col(df, ["Player","Player Name","Name"]) or "Player"

    # Map available features
    mapped: dict[str, str] = {}
    for feat in BASE_FEATURES:
        col = find_col(df, ALIASES.get(feat, [feat]))
        if col is not None:
            mapped[feat] = col
    if not mapped:
        raise ValueError("No usable base features found in CSV for this position.")

    # Apply bans / musts to features (only those present)
    ban_feats_set = {f.strip() for f in ban_feats}
    must_feats_list = [f for f in must_feats if f in mapped]

    # base pool excludes banned and unavailable
    base_pool = [f for f in mapped.keys() if f not in ban_feats_set]

    # Allowed interactions: parents present in mapped and not banned
    allowed_inters: dict[str, tuple[str,str]] = {}
    for iname, (a, b) in INTERACTIONS.items():
        if iname in ban_inters:
            continue
        if a in mapped and b in mapped and a not in ban_feats_set and b not in ban_feats_set:
            allowed_inters[iname] = (a, b)

    # Validate must_inters against allowed
    must_inters_list = [n for n in must_inters if n in allowed_inters]

    # If hierarchy="none" and we're doing interaction-only, preemptively
    # remove parents of *must* interactions from base_pool unless explicitly forced.
    if interaction_hierarchy == "none":
        parent_block = set()
        for iname in must_inters_list:
            a, b = allowed_inters[iname]
            parent_block.update([a, b])
        base_pool = [f for f in base_pool if (f not in parent_block) or (f in must_feats_list)]

    # Build full matrices (all mapped columns, numeric)
    X_all_raw = pd.DataFrame({feat: to_num(df[col]) for feat, col in mapped.items()})
    y_all     = to_num(df[y_col])
    names_all = df[name_col].astype(str).fillna("")

    mask = y_all.notna()
    X_all_raw, y_all, names_all = (
        X_all_raw.loc[mask].reset_index(drop=True),
        y_all.loc[mask].reset_index(drop=True),
        names_all.loc[mask].reset_index(drop=True)
    )

    # Drop Draft Capital == 0 UPFRONT (treat as missing/placeholder)
    if "Draft Capital" in X_all_raw.columns:
        keep = X_all_raw["Draft Capital"] != 0
        X_all_raw = X_all_raw.loc[keep].reset_index(drop=True)
        y_all     = y_all.loc[keep].reset_index(drop=True)
        names_all = names_all.loc[keep].reset_index(drop=True)

    rows: list[dict] = []

    # One block per seed
    for seed in seeds:
        t0 = time.perf_counter()

        pred_path   = out_dir / f"{pos.lower()}_wide_best_preds_seed{seed}_subs{n_subsets}.csv"
        meta_path   = out_dir / f"{pos.lower()}_wide_best_meta_seed{seed}_subs{n_subsets}.json"
        board_path  = out_dir / f"{pos.lower()}_wide_leaderboard_seed{seed}_subs{n_subsets}.csv"
        fi_path     = out_dir / f"{pos.lower()}_feature_importance_seed{seed}_subs{n_subsets}.csv"
        shap_path   = out_dir / f"{pos.lower()}_shap_contributions_seed{seed}_subs{n_subsets}.csv"

        rng = np.random.default_rng(seed)

        X_tr_raw, X_te_raw, y_tr, y_te, n_tr, n_te = train_test_split(
            X_all_raw, y_all, names_all, test_size=test_size, random_state=seed
        )

        leaderboard = []
        baseline_done = False

        for subset_idx in range(n_subsets):
            if not baseline_done:
                # Baseline: if must feats provided, use them; else best single feature
                if must_feats_list:
                    bases = list(dict.fromkeys(must_feats_list))[:max_base_feats]
                    inters = list(dict.fromkeys(must_inters_list))[:max_interactions]
                    baseline_done = True
                else:
                    best_feat, best_cv = None, -1e9
                    for f in [fname for fname in base_pool if fname in X_tr_raw.columns]:
                        imp = SimpleImputer(strategy="median")
                        X_single = imp.fit_transform(X_tr_raw[[f]])
                        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                        cv = cross_val_score(GradientBoostingRegressor(random_state=seed), X_single, y_tr, scoring="r2", cv=kf).mean()
                        if cv > best_cv:
                            best_cv, best_feat = cv, f
                    bases, inters = ([best_feat] if best_feat else []), []
                    baseline_done = True
            else:
                bases, inters = sample_subset(
                    rng,
                    base_pool=base_pool,
                    max_bases=max_base_feats,
                    allowed_inters=allowed_inters,
                    max_inters=max_interactions,
                    must_feats=must_feats_list,
                    must_inters=must_inters_list,
                    interaction_hierarchy=interaction_hierarchy,
                    drop_parents_when_none=drop_parents_when_none,
                )

            # ---- Design matrices ----
            Xtr_df, Xte_df = X_tr_raw[bases].copy(), X_te_raw[bases].copy()

            # add interactions (compute from raw parents even if parents not in bases)
            for iname in inters:
                a, b = allowed_inters[iname]
                if a in X_tr_raw.columns and b in X_tr_raw.columns:
                    Xtr_df[iname] = X_tr_raw[a].values * X_tr_raw[b].values
                    Xte_df[iname] = X_te_raw[a].values * X_te_raw[b].values

            Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)
            imp = SimpleImputer(strategy="median")
            Xtr, Xte = imp.fit_transform(Xtr_df), imp.transform(Xte_df)

            # ---- Tune models on TRAIN (CV) ----
            best_cv, best_tag, best_est = -1e9, None, None
            for tag, est, grid in model_spaces(seed):
                n_iter = min(n_iter_per_model, int(np.prod([len(v) for v in grid.values()])))
                search = RandomizedSearchCV(est, grid, n_iter=n_iter, scoring="r2",
                                            cv=cv_folds, random_state=seed, n_jobs=-1)
                search.fit(Xtr, y_tr)
                if search.best_score_ > best_cv:
                    best_cv, best_tag, best_est = search.best_score_, tag, search.best_estimator_

            # ---- Test performance ----
            best_est.fit(Xtr, y_tr)
            y_pred = best_est.predict(Xte)

            leaderboard.append({
                "position": pos, "seed": seed, "n_subsets": n_subsets, "subset_idx": subset_idx,
                "model": best_tag, "cvR2_mean": best_cv,
                "TEST_R2": r2_score(y_te, y_pred),
                "TEST_MAE": mean_absolute_error(y_te, y_pred),
                "TEST_RMSE": rmse(y_te, y_pred),
                "n_features": len(bases) + len(inters),
                "bases": "|".join(bases), "interactions": "|".join(inters),
                "max_pred": float(np.max(y_pred)) if len(y_pred) else np.nan,
            })

        # Leaderboard: keep top 15, save
        board = pd.DataFrame(leaderboard).sort_values(["TEST_R2","cvR2_mean"], ascending=False).head(15)
        board.to_csv(board_path, index=False)

        # Refit best config & export predictions + importance + SHAP
        best_row = board.iloc[0]
        best_bases = best_row["bases"].split("|") if best_row["bases"] else []
        best_inters = best_row["interactions"].split("|") if best_row["interactions"] else []

        Xtr_df = X_tr_raw[best_bases].copy()
        Xte_df = X_te_raw[best_bases].copy()
        for iname in best_inters:
            if iname:
                a, b = INTERACTIONS[iname]
                if a in X_tr_raw.columns and b in X_tr_raw.columns:
                    Xtr_df[iname] = X_tr_raw[a].values * X_tr_raw[b].values
                    Xte_df[iname] = X_te_raw[a].values * X_te_raw[b].values
        Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(Xtr_df), imp.transform(Xte_df)

        # Re-tune within best model class
        tag_map = {row[0]: (row[1], row[2]) for row in model_spaces(seed)}
        est, grid = tag_map[best_row["model"]]
        n_iter = min(n_iter_per_model, int(np.prod([len(v) for v in grid.values()])))
        search = RandomizedSearchCV(est, grid, n_iter=n_iter, scoring="r2", cv=cv_folds, random_state=seed, n_jobs=-1)
        search.fit(Xtr, y_tr)
        best_est = search.best_estimator_
        y_best = best_est.predict(Xte)

        # Save predictions
        out_df = pd.DataFrame({
            "Player": n_te.values,
            f"Actual_{pos}_Grade": y_te.values,
            f"Predicted_{pos}_Grade": y_best,
            "Error": y_best - y_te.values
        }).sort_values(f"Actual_{pos}_Grade", ascending=False)
        out_df.to_csv(pred_path, index=False)

        # ---- Global feature importance (over base+interaction columns) ----
        try:
            fi = getattr(best_est, "feature_importances_", None)
            if fi is not None:
                cols = list(Xtr_df.columns)
                fi_df = pd.DataFrame({"feature": cols, "importance": fi})
                fi_df.sort_values("importance", ascending=False, inplace=True)
                fi_df.to_csv(fi_path, index=False)
        except Exception as _:
            pass

        # ---- Per-player SHAP contributions (if shap installed) ----
        shap_ok = False
        try:
            import shap  # type: ignore
            explainer = shap.TreeExplainer(best_est)
            shap_values = explainer.shap_values(Xte_df)
            # expected value (base score) is scalar for regression
            base_value = float(np.ravel(explainer.expected_value)[0]) if np.ndim(explainer.expected_value) else float(explainer.expected_value)
            shap_df = pd.DataFrame(shap_values, columns=Xte_df.columns)
            shap_df.insert(0, "Player", n_te.values)
            shap_df["BASE_VALUE"] = base_value
            shap_df["PREDICTED"] = y_best
            shap_df.to_csv(shap_path, index=False)
            shap_ok = True
        except Exception as e:
            print(f"SHAP unavailable or failed ({e}). Skipping SHAP export.")

        # Meta
        meta = {
            "position": pos, "seed": seed, "n_subsets": n_subsets,
            "leaderboard_csv": str(board_path),
            "best_predictions_csv": str(pred_path),
            "feature_importance_csv": str(fi_path),
            "shap_csv": (str(shap_path) if shap_ok else None),
            "best_bases": best_bases,
            "best_interactions": best_inters,
            "best_model_tag": best_row["model"],
            "best_cvR2": float(best_row["cvR2_mean"]),
            "best_test_R2": float(best_row["TEST_R2"]),
            "best_test_MAE": float(best_row["TEST_MAE"]),
            "best_test_RMSE": float(best_row["TEST_RMSE"]),
            "max_pred_test": float(out_df[f"Predicted_{pos}_Grade"].max()),
            "interaction_hierarchy": interaction_hierarchy,
            "drop_parents_when_none": drop_parents_when_none,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        runtime_sec = time.perf_counter() - t0

        rows.append({
            "position": pos,
            "seed": seed,
            "n_subsets": n_subsets,
            "best_test_R2": meta["best_test_R2"],
            "best_test_MAE": meta["best_test_MAE"],
            "best_test_RMSE": meta["best_test_RMSE"],
            "runtime_sec": runtime_sec,
            "leaderboard_csv": str(board_path),
            "predictions_csv": str(pred_path),
            "feature_importance_csv": str(fi_path),
            "shap_csv": (str(shap_path) if shap_ok else None),
            "best_model_tag": meta["best_model_tag"],
            "best_bases": "|".join(best_bases),
            "best_interactions": "|".join(best_inters),
        })

        print(f"\n[{pos}] Seed {seed} | n_subsets={n_subsets} → "
              f"R²={meta['best_test_R2']:.4f} | MAE={meta['best_test_MAE']:.3f} | "
              f"RMSE={meta['best_test_RMSE']:.3f} | time={runtime_sec:.1f}s")

    return pd.DataFrame(rows)

# ---------------------------- CLI ENTRYPOINT ----------------------------
if __name__ == "__main__":
    import argparse
    try:
        _HERE = Path(__file__).resolve()
        PROJECT_ROOT = _HERE.parents[2]  # <project>/src/models/...
    except NameError:
        PROJECT_ROOT = Path.cwd()

    ap = argparse.ArgumentParser(description="Position-aware wide search (GB only) + must/ban controls + SHAP export.")
    ap.add_argument("--position", type=str, default="RB", help="RB, WR, TE, or QB")
    ap.add_argument("--seeds", type=int, nargs="*", default=[123,456,789], help="Seeds to run")
    ap.add_argument("--subset-grid", type=int, nargs="*", default=[10,20,40,60],
                    help="Iteration counts (N_SUBSETS) to profile")
    ap.add_argument("--max-base-feats", type=int, default=8, help="Max base features per subset")
    ap.add_argument("--max-interactions", type=int, default=3, help="Max interactions per subset")
    ap.add_argument("--n-iter-per-model", type=int, default=25, help="RandomizedSearchCV iterations per model")
    ap.add_argument("--cv-folds", type=int, default=5, help="CV folds")
    ap.add_argument("--test-size", type=float, default=0.20, help="Holdout fraction")

    # constraints + hierarchy
    ap.add_argument("--must-feats", type=str, default="", help="Comma-separated canonical features to force include.")
    ap.add_argument("--ban-feats",  type=str, default="", help="Comma-separated canonical features to exclude.")
    ap.add_argument("--must-inters",type=str, default="", help="Comma-separated interaction names to force include.")
    ap.add_argument("--ban-inters", type=str, default="", help="Comma-separated interaction names to exclude.")
    ap.add_argument("--interaction-hierarchy", choices=["strong","weak","none"], default="weak",
                    help="Require parents for interactions: strong(all), weak(must only), none(allow interaction-only).")
    ap.add_argument("--drop-parents-when-none", action="store_true",
                    help="If set with hierarchy='none', strip parents of interactions from bases unless explicitly must-feats.")

    args = ap.parse_args()

    def _csv_to_list(s: str) -> list[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    must_feats = _csv_to_list(args.must_feats)
    ban_feats  = _csv_to_list(args.ban_feats)
    must_inters= _csv_to_list(args.must_inters)
    ban_inters = _csv_to_list(args.ban_inters)

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
            drop_parents_when_none=bool(args.drop_parents_when_none),
        )
        all_results.append(res)

    summary = pd.concat(all_results, ignore_index=True)
    out_dir = default_out_dir(PROJECT_ROOT, args.position)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.position.lower()}_runtime_accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Console summary
    agg = (summary.groupby("n_subsets")
                   .agg(best_R2=("best_test_R2","max"),
                        avg_runtime_s=("runtime_sec","mean"),
                        runs=("best_test_R2","count"))
                   .reset_index())
    print("\n=== Accuracy vs Runtime (by n_subsets) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved detailed summary → {summary_path}")

    # Pareto chart
    try:
        pareto_path = save_pareto_chart(summary, args.position, out_dir)
        print(f"Saved Pareto chart → {pareto_path}")
    except Exception as e:
        print(f"Failed to render Pareto chart: {e}")
