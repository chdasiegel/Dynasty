# src/models/multi_tune_by_position.py
# ===============================================================
# Position-aware Grade — Wide search over feature subsets & hypers
# - position: RB / WR / TE / QB (affects CSV path + target names)
# - Randomly sample feature combos (+ limited interactions)
# - Tune models (GB, HGB) with RandomizedSearchCV
# - 80/20 split; CV on TRAIN only; metrics on TEST
# - Sweeps N_SUBSETS to profile accuracy vs runtime
# - Logs per-run leaderboards and an overall summary CSV
# - Save Pareto chart of Accuracy (R²) vs Runtime
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
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

# ----------------------------- Features -----------------------------
BASE_FEATURES: list[str] = [
    "DOM+","40 Time","BMI","MTF/A","YPC","YPR","ELU","YCO/A",
    "Break%","Draft Capital","Conference Rank","Draft Age","Breakout Age",
    "Y/RR","YAC/R","aDOT","EPA/P","aYPTPA","CTPRR","UCTPRR","Drop Rate","CC%","Wide%","Slot%",
    "Comp%","ADJ%","BTT%","TWP%","DAA","YPA"
]

# Removed Features: Height, Weight

INTERACTIONS: dict[str, tuple[str, str]] = {
   # "DOMxDraft":      ("DOM+", "Draft Capital"),
   # "YPCxELU":        ("YPC",  "ELU"),
   # "ELUxYCOA":       ("ELU",  "YCO/A"),
    "SpeedxBMI":       ("Speed","BMI"),
    "Wide%xSlot%":     ("Wide%","Slot%"),
}

ALIASES: dict[str, list[str]] = {
    "DOM+":             ["DOM+","DOMp","DOM_plus","DOMp_Weighted","DOM","DOM++"],
    "Speed":            ["40 Time","Forty","40","Speed"],
    "BMI":              ["BMI","Body Mass Index"],
   # "Height":           ["Height","Ht","HT (in)","HT_in"],
   # "Weight":           ["Weight","Wt","WT (lbs)","WT_lbs"],
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
    # invert "lower is better" (faster times, earlier rounds, younger age)
    X = X.copy()
    for c in ["40 Time","Speed","Draft Capital","Shuttle","Three Cone","Draft Age"]:
        if c in X.columns:
            X[c] = -X[c]
    return X

def add_interactions(X: pd.DataFrame, inter_names: list[str]) -> pd.DataFrame:
    X = X.copy()
    for name in inter_names:
        a,b = INTERACTIONS[name]
        if a in X.columns and b in X.columns:
            X[name] = X[a]*X[b]
    return X

def model_spaces(random_state: int):
    """Only GB"""
    return [
        ("GB", GradientBoostingRegressor(random_state=random_state), {
            "n_estimators": [400, 600, 800, 1000, 1200],
            "learning_rate": [0.03, 0.05, 0.07, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 3],
        }),
       # ("HGB", HistGradientBoostingRegressor(random_state=random_state), {
        #    "learning_rate": [0.03, 0.05, 0.08, 0.1],
        #    "max_depth": [3, 6, 9],
        #    "l2_regularization": [0.0, 0.1, 0.3, 0.5],
         #   "max_bins": [128, 255],
        #}),
    ]

def sample_subset(rng: np.random.Generator,
                  base_pool: list[str],
                  max_bases: int,
                  allowed_inters: dict[str, tuple[str,str]],
                  max_inters: int) -> tuple[list[str], list[str]]:
    n_bases = int(rng.integers(low=min(3, len(base_pool)), high=min(max_bases, len(base_pool)) + 1))
    bases = rng.choice(base_pool, size=n_bases, replace=False).tolist()

    inter_names: list[str] = []
    if allowed_inters and max_inters > 0:
        eligible = [name for name,(a,b) in allowed_inters.items() if a in bases and b in bases]
        if eligible:
            k = int(rng.integers(low=0, high=min(max_inters, len(eligible)) + 1))
            if k > 0:
                inter_names = rng.choice(eligible, size=k, replace=False).tolist()
    return bases, inter_names

# ---------------------- PARETO CHART HELPER ----------------------
def save_pareto_chart(summary: pd.DataFrame, position: str, out_dir: Path) -> Path:
    """
    Build & save a Pareto-like chart of Accuracy (best R² per n_subsets)
    vs average runtime (seconds). Returns the PNG path.
    """
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
                         test_size: float = 0.20) -> pd.DataFrame:
    """
    Run the wide-search once per seed for a given n_subsets,
    returning a summary row per seed with runtime + best test metrics.
    """
    pos = position.upper()
    out_dir = default_out_dir(project_root, pos)
    csv_path = default_csv_for_position(project_root, pos)

    # DEBUG PATHS
    print("\n========== DEBUG PATHS ==========")
    print(f"Position selected : {pos}")
    print(f"CSV path used     : {csv_path}")
    print(f"Output directory  : {out_dir}")
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

    # Allowed interactions (parents present)
    allowed_inters = {k:v for k,v in INTERACTIONS.items() if v[0] in mapped and v[1] in mapped}

    # Build full matrices
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

        pred_path  = out_dir / f"{pos.lower()}_wide_best_preds_seed{seed}_subs{n_subsets}.csv"
        meta_path  = out_dir / f"{pos.lower()}_wide_best_meta_seed{seed}_subs{n_subsets}.json"
        board_path = out_dir / f"{pos.lower()}_wide_leaderboard_seed{seed}_subs{n_subsets}.csv"

        rng = np.random.default_rng(seed)

        X_tr_raw, X_te_raw, y_tr, y_te, n_tr, n_te = train_test_split(
            X_all_raw, y_all, names_all, test_size=test_size, random_state=seed
        )

        leaderboard = []
        baseline_done = False

        for subset_idx in range(n_subsets):
            if not baseline_done:
                # Best single feature baseline (by CV R²)
                best_feat, best_cv = None, -1e9
                for f in X_tr_raw.columns:
                    imp = SimpleImputer(strategy="median")
                    X_single = imp.fit_transform(X_tr_raw[[f]])
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                    cv = cross_val_score(GradientBoostingRegressor(random_state=seed), X_single, y_tr, scoring="r2", cv=kf).mean()
                    if cv > best_cv:
                        best_cv, best_feat = cv, f
                bases, inters = [best_feat], []
                baseline_done = True
            else:
                bases, inters = sample_subset(rng, list(X_tr_raw.columns), max_base_feats, allowed_inters, max_interactions)

            # Design matrices
            Xtr_df, Xte_df = X_tr_raw[bases].copy(), X_te_raw[bases].copy()
            for iname in inters:
                a,b = INTERACTIONS[iname]
                if a in Xtr_df.columns and b in Xtr_df.columns:
                    Xtr_df[iname] = Xtr_df[a]*Xtr_df[b]
                    Xte_df[iname] = Xte_df[a]*Xte_df[b]

            Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)

            imp = SimpleImputer(strategy="median")
            Xtr, Xte = imp.fit_transform(Xtr_df), imp.transform(Xte_df)

            # Tune models on TRAIN (CV)
            best_cv, best_tag, best_est = -1e9, None, None
            for tag, est, grid in model_spaces(seed):
                n_iter = min(n_iter_per_model, int(np.prod([len(v) for v in grid.values()])))
                search = RandomizedSearchCV(est, grid, n_iter=n_iter, scoring="r2",
                                            cv=cv_folds, random_state=seed, n_jobs=-1)
                search.fit(Xtr, y_tr)
                if search.best_score_ > best_cv:
                    best_cv, best_tag, best_est = search.best_score_, tag, search.best_estimator_

            # Test performance
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

        # Refit best config & export predictions
        best_row = board.iloc[0]
        best_bases = best_row["bases"].split("|") if best_row["bases"] else []
        best_inters = best_row["interactions"].split("|") if best_row["interactions"] else []

        Xtr_df = X_tr_raw[best_bases].copy()
        Xte_df = X_te_raw[best_bases].copy()
        for iname in best_inters:
            if iname:
                a,b = INTERACTIONS[iname]
                if a in Xtr_df.columns and b in Xtr_df.columns:
                    Xtr_df[iname] = Xtr_df[a]*Xtr_df[b]
                    Xte_df[iname] = Xte_df[a]*Xte_df[b]
        Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(Xtr_df), imp.transform(Xte_df)

        # Optional: retune within best model class
        tag_map = {row[0]: (row[1], row[2]) for row in model_spaces(seed)}
        est, grid = tag_map[best_row["model"]]
        n_iter = min(n_iter_per_model, int(np.prod([len(v) for v in grid.values()])))
        search = RandomizedSearchCV(est, grid, n_iter=n_iter, scoring="r2", cv=cv_folds, random_state=seed, n_jobs=-1)
        search.fit(Xtr, y_tr)
        best_est = search.best_estimator_
        y_best = best_est.predict(Xte)

        out_df = pd.DataFrame({
            "Player": n_te.values,
            f"Actual_{pos}_Grade": y_te.values,
            f"Predicted_{pos}_Grade": y_best,
            "Error": y_best - y_te.values
        }).sort_values(f"Actual_{pos}_Grade", ascending=False)
        out_df.to_csv(pred_path, index=False)

        # Meta
        meta = {
            "position": pos, "seed": seed, "n_subsets": n_subsets,
            "leaderboard_csv": str(board_path),
            "best_predictions_csv": str(pred_path),
            "best_bases": best_bases,
            "best_interactions": best_inters,
            "best_model_tag": best_row["model"],
            "best_cvR2": float(best_row["cvR2_mean"]),
            "best_test_R2": float(best_row["TEST_R2"]),
            "best_test_MAE": float(best_row["TEST_MAE"]),
            "best_test_RMSE": float(best_row["TEST_RMSE"]),
            "max_pred_test": float(out_df[f"Predicted_{pos}_Grade"].max()),
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

    ap = argparse.ArgumentParser(description="Position-aware wide search with runtime profiling (GB & HGB only).")
    ap.add_argument("--position", type=str, default="RB", help="RB, WR, TE, or QB")
    ap.add_argument("--seeds", type=int, nargs="*", default=[123,456,789], help="Seeds to run")
    ap.add_argument("--subset-grid", type=int, nargs="*", default=[10,20,40,60],
                    help="Iteration counts (N_SUBSETS) to profile")
    ap.add_argument("--max-base-feats", type=int, default=8, help="Max base features per subset")
    ap.add_argument("--max-interactions", type=int, default=3, help="Max interactions per subset")
    ap.add_argument("--n-iter-per-model", type=int, default=25, help="RandomizedSearchCV iterations per model")
    ap.add_argument("--cv-folds", type=int, default=5, help="CV folds")
    ap.add_argument("--test-size", type=float, default=0.20, help="Holdout fraction")
    args = ap.parse_args()

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
        )
        all_results.append(res)

    summary = pd.concat(all_results, ignore_index=True)
    out_dir = default_out_dir(PROJECT_ROOT, args.position)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.position.lower()}_runtime_accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Console summary: best R² per n_subsets + avg runtime
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
