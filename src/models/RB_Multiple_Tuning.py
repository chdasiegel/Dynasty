#src.RB_Multiple_Tuning.py
# ===============================================================
# RB Grade — Wide search over feature subsets & hyperparameters
# - Randomly sample feature combos (+ limited interactions)
# - Tune multiple models (GB, RF, ET, HGB) with RandomizedSearchCV
# - 80/20 train-test; CV on TRAIN only; metrics on TEST
# - Logs leaderboard + meta for reproducibility
# ===============================================================
import re, json, numpy as np, pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    ExtraTreesRegressor, HistGradientBoostingRegressor
)

# ---------------- Search controls ----------------
CSV_PATH            = Path("./data/Bakery/RB/Bakery_RB_Overall.csv")
OUT_DIR             = Path("./data/Bakery/_derived")
TEST_SIZE           = 0.20

SEEDS               = [123, 456, 789] # run search for multiple seeds
N_SUBSETS           = 60              # random feature subsets per seed
MAX_BASE_FEATS      = 8               # cap base features per subset
MAX_INTERACTIONS    = 3               # cap interactions per subset
N_ITER_PER_MODEL    = 25              # RandomizedSearchCV iterations per model
CV_FOLDS            = 5

#CLIP_MIN, CLIP_MAX  = 0.0, 15.0       # clip predictions

# ---------------- Feature space ----------------
BASE_FEATURES = [
    "DOM+","40 Time","BMI","Height","Weight","MTF/A","YPC","YPR","ELU","YCO/A","Break%","Draft Capital","Conference Rank","Draft Age","Breakout Age","Y/RR","YAC/R","aDOT","EPA/P","aYPTPA","CTPRR","UCTPRR","Drop Rate","CC%","Wide%","Slot%","Comp%","ADJ%","BTT%","TWP%","DAA",
]

INTERACTIONS = {
    "DOMxDraft":      ("DOM+","Draft Capital"),
    "YPCxELU":        ("YPC","ELU"),
    "ELUxYCOA":       ("ELU","YCO/A"),
    "SpeedxHeight":   ("Speed","Height")
    "Wide%xSlot%":    ("Wide%","Slot%")  
}

ALIASES = {
    "DOM+":             ["DOM+","DOMp","DOM_plus","DOMp_Weighted","DOM"],
    "Speed":            ["40 Time","Forty","40","Speed"],
    "BMI":              ["BMI","Body Mass Index"],
    "MTF/A":            ["Missed Tackles Forced Per Attempt","MTFA","MTF/A"],
    "YPC":              ["YPC","Yards per Carry","Yards/Carry","Rushing YPC"],
    "YPR":              ["YPR","Yards Per Reception"],
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
    "Y/RR":             ["Yards Route Run","Yards per Route Run","Y/RR"],
    "YAC/R":            ["Yards After Catch Per Reception","YAC/R"],
    "aDOT":             ["Average Depth of Target","aDOT","ADOT","adot"],
    "EPA/P":            ["Expected Points Added per Play","EPA/P"],
    "aYPTPA":           ["Adjusted Yards per Team Pass Attempt","aYPTPA","AYPTA","aypta"],
    "CTPRR":            ["Targets Per Route Run","Contested Targets Per Route Run","CTPRR"],
    "UCTPRR":           ["Uncontested Targets Per Route Run","UCTPRR"],
    "Drop Rate":        ["Drop %","Drop%","DropRate","Drop Rate"],
    "CC%":              ["Contested Catch Percent","CC","CC %"],
    "Wide%":            ["X/Z%","X","Z","Wide","Wide %","Wide%"],
    "Slot%":            ["Slot","Slot %","Slot%"],
    "Comp%":            ["Completion Percentage","Comp","COMP","Comp%"],
    "ADJ%":             ["Adjusted Completed Percentage","ADJ","ADJ%","ADJ %"],
    "BTT%":             ["Big Time Throw Percentage","BTT","BTT %","BTT%"],
    "TWP%":             ["Turnover Worthy Play Percentage","TWP","TWP %","TWP%"],
    "DAA":              ["Depth-Adjusted Accuracy","Depth Adjusted Accuracy","DAA"],
    "YPA":              ["Yards Per Attempt","YpA","YPA"]
    
}
TARGET_CANDS = ["RB Grade","RBGrade","RB_Grade"]
NAME_CANDS   = ["Player","Player Name","Name"]

# ---------------- Utilities ----------------
def find_col(frame, candidates):
    norm = {re.sub(r"\s+","",c).lower(): c for c in frame.columns}
    for cand in candidates:
        key = re.sub(r"\s+","",cand).lower()
        if key in norm: return norm[key]
    return None

def to_num(series):
    s = series.astype(str).str.strip()
    s = (s.str.replace('%','',regex=False)
           .str.replace(r'(?i)round\s*','',regex=True)
           .str.replace(r'(?i)^r\s*','',regex=True)
           .str.replace(r'(?i)(st|nd|rd|th)$','',regex=True)
           .str.replace(',','',regex=False)
           .str.replace(r'[^0-9\.\-]','',regex=True))
    return pd.to_numeric(s, errors='coerce')

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def invert_cols(X):
    # invert "lower is better"
    for c in ["40 Time","Draft Capital","Shuttle","Three Cone","Draft Age"]:
        if c in X.columns: X[c] = -X[c]
    return X

def add_interactions(X, inter_names):
    X = X.copy()
    for name in inter_names:
        a,b = INTERACTIONS[name]
        if a in X.columns and b in X.columns:
            X[name] = X[a]*X[b]
    return X

# ---------------- Load & prep ----------------
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

y_col    = find_col(df, TARGET_CANDS)
name_col = find_col(df, NAME_CANDS) or "Player"
if not y_col:
    raise ValueError(f"Could not find target column among {TARGET_CANDS}")

mapped = {}
for feat in BASE_FEATURES:
    col = find_col(df, ALIASES.get(feat, [feat]))
    if col is not None:
        mapped[feat] = col

ALLOWED_INTERS = {k:v for k,v in INTERACTIONS.items() if v[0] in mapped and v[1] in mapped}

X_all_raw = pd.DataFrame({feat: to_num(df[col]) for feat, col in mapped.items()})
y_all     = to_num(df[y_col])
names_all = df[name_col].astype(str).fillna("")

mask = y_all.notna()
X_all_raw, y_all, names_all = X_all_raw.loc[mask].reset_index(drop=True), y_all.loc[mask].reset_index(drop=True), names_all.loc[mask].reset_index(drop=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Model spaces ----------------
def model_spaces(random_state):
    return [
        ("GB", GradientBoostingRegressor(random_state=random_state), {
            "n_estimators": [400, 600, 800, 1000, 1200],
            "learning_rate": [0.03, 0.05, 0.07, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 3],
        }),
        ("RF", RandomForestRegressor(random_state=random_state, n_jobs=-1), {
            "n_estimators": [600, 900, 1200, 1500],
            "max_depth": [None, 12, 16, 20],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", 0.7, 0.9, 1.0],
        }),
        ("ET", ExtraTreesRegressor(random_state=random_state, n_jobs=-1), {
            "n_estimators": [600, 900, 1200, 1500],
            "max_depth": [None, 12, 16, 20],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", 0.7, 0.9, 1.0],
        }),
        ("HGB", HistGradientBoostingRegressor(random_state=random_state), {
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "max_depth": [3, 6, 9],
            "l2_regularization": [0.0, 0.1, 0.3, 0.5],
            "max_bins": [128, 255],
        })
    ]

# ---------------- Random subset generator ----------------
def sample_subset(rng, base_pool, max_bases, allowed_inters, max_inters):
    n_bases = rng.integers(low=min(3, len(base_pool)), high=min(max_bases, len(base_pool)) + 1)
    bases = rng.choice(base_pool, size=int(n_bases), replace=False).tolist()

    inter_names = []
    if allowed_inters and max_inters > 0:
        eligible = [name for name,(a,b) in allowed_inters.items() if a in bases and b in bases]
        if eligible:
            k = rng.integers(low=0, high=min(max_inters, len(eligible)) + 1)
            if k > 0:
                inter_names = rng.choice(eligible, size=int(k), replace=False).tolist()
    return bases, inter_names

# ---------------- One full search (per seed) ----------------
def run_seed(seed: int):
    pred_path  = OUT_DIR / f"rb_wide_best_preds_seed{seed}.csv"
    meta_path  = OUT_DIR / f"rb_wide_best_meta_seed{seed}.json"
    board_path = OUT_DIR / f"rb_wide_leaderboard_seed{seed}.csv"

    rng = np.random.default_rng(seed)

    X_tr_raw, X_te_raw, y_tr, y_te, n_tr, n_te = train_test_split(
        X_all_raw, y_all, names_all, test_size=TEST_SIZE, random_state=seed
    )

    leaderboard = []
    baseline_done = False

    for subset_idx in range(N_SUBSETS):
        if not baseline_done:
            # best single feature baseline
            best_feat, best_cv = None, -1e9
            for f in X_tr_raw.columns:
                imp = SimpleImputer(strategy="median")
                X_single = imp.fit_transform(X_tr_raw[[f]])
                kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
                cv = cross_val_score(GradientBoostingRegressor(random_state=seed), X_single, y_tr, scoring="r2", cv=kf).mean()
                if cv > best_cv: best_cv, best_feat = cv, f
            bases, inters = [best_feat], []
            baseline_done = True
        else:
            bases, inters = sample_subset(rng, list(X_tr_raw.columns), MAX_BASE_FEATS, ALLOWED_INTERS, MAX_INTERACTIONS)

        # build matrices
        Xtr_df, Xte_df = X_tr_raw[bases].copy(), X_te_raw[bases].copy()
        for iname in inters:
            a,b = INTERACTIONS[iname]
            Xtr_df[iname], Xte_df[iname] = Xtr_df[a]*Xtr_df[b], Xte_df[a]*Xte_df[b]
        Xtr_df, Xte_df = invert_cols(Xtr_df), invert_cols(Xte_df)

        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(Xtr_df), imp.transform(Xte_df)

        # tune models
        best_cv, best_tag, best_est = -1e9, None, None
        for tag, est, grid in model_spaces(seed):
            n_iter = min(N_ITER_PER_MODEL, int(np.prod([len(v) for v in grid.values()])))
            search = RandomizedSearchCV(est, grid, n_iter=n_iter, scoring="r2", cv=CV_FOLDS, random_state=seed, n_jobs=-1)
            search.fit(Xtr, y_tr)
            if search.best_score_ > best_cv:
                best_cv, best_tag, best_est = search.best_score_, tag, search.best_estimator_

        best_est.fit(Xtr, y_tr)
        y_pred = np.clip(best_est.predict(Xte), CLIP_MIN, CLIP_MAX)

        leaderboard.append({
            "seed": seed, "subset_idx": subset_idx, "model": best_tag,
            "cvR2_mean": best_cv, "TEST_R2": r2_score(y_te, y_pred),
            "TEST_MAE": mean_absolute_error(y_te, y_pred),
            "TEST_RMSE": rmse(y_te, y_pred),
            "n_features": len(bases) + len(inters),
            "bases": "|".join(bases), "interactions": "|".join(inters),
            "max_pred": float(np.max(y_pred)),
        })

    # leaderboard
    board = pd.DataFrame(leaderboard).sort_values(["TEST_R2","cvR2_mean"], ascending=False).head(15)
    board.to_csv(board_path, index=False)

    best_row = board.iloc[0]
    print(f"\n=== Seed {seed} — top of board ===")
    print(board[["subset_idx","model","n_features","cvR2_mean","TEST_R2","TEST_MAE","TEST_RMSE","bases","interactions"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # save meta
    meta = {
        "seed": seed, "leaderboard_csv": str(board_path), "best_predictions_csv": str(pred_path),
        "best_bases": best_row["bases"].split("|"), "best_interactions": best_row["interactions"].split("|") if best_row["interactions"] else [],
        "best_model_tag": best_row["model"], "best_cvR2": float(best_row["cvR2_mean"]),
        "best_test_R2": float(best_row["TEST_R2"]), "best_test_MAE": float(best_row["TEST_MAE"]),
        "best_test_RMSE": float(best_row["TEST_RMSE"]), "max_pred_test": float(best_row["max_pred"])
    }
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)

    return meta

# ---------------- Run across seeds & summarize ----------------
all_meta = [run_seed(s) for s in SEEDS]
summary = pd.DataFrame(all_meta).sort_values("best_test_R2", ascending=False)
print("\n=== Overall summary across seeds ===")
print(summary[["seed","best_cvR2","best_test_R2","best_test_MAE","best_test_RMSE","best_model_tag","best_bases","best_interactions"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
