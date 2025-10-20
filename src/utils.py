# src.utils.py
# =====================================================
# Central Functions Hub
# =====================================================
from __future__ import annotations
import re
from typing import Iterable, Optional, Sequence, Dict
import pandas as pd
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

# ----------------------------- Features -----------------------------
BASE_FEATURES: list[str] = [
    "DOM+","RDOM+","Speed","BMI","MTF/A","YPC","YPR","ELU","YCO/A",
    "Break%","Draft Capital","Conference Rank","Draft Age","Breakout Age",
    "Y/RR","YAC/R","aDOT","EPA/P","aYPTPA","CTPRR","UCTPRR","Drop Rate","CC%","Wide%","Slot%",
    "Comp%","ADJ%","BTT%","TWP%","DAA","YPA"
]

INTERACTIONS: dict[str, tuple[str, str]] = {
    "SpeedxBMI":   ("Speed","BMI"),
    "Wide%xSlot%": ("Wide%","Slot%"),
}

ALIASES: dict[str, list[str]] = {
    "DOM+":             ["DOM+","DOMp","DOM_plus","DOMp_Weighted","DOM"],
    "RDOM+":             ["RDOM", "rdom", "Rdom", "RDom", "RDOM+"],
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


# ----------------------------- Utilities -----------------------------

# ---- Directory Tools ----

def default_out_dir(project_root: Path, pos: str) -> Path:
    """Return the default output directory for a given position."""
    return project_root / "data" / "Bakery" / "_derived" / pos.upper()


pd.set_option("display.max_columns", None)

# ---- Name cleaning ----
def clean_player_name(player_name):
    """Remove punctuation, suffixes, and extra spaces from player names."""
    if not isinstance(player_name, str):
        return player_name
    player_name = re.sub(r"[^\w\s'-]", '', player_name)  # Remove punctuation
    suffixes = ['Jr', 'Sr', 'II', 'III', 'IV', 'V']
    pattern = r'\b(?:' + '|'.join(suffixes) + r')\b'
    player_name = re.sub(pattern, '', player_name, flags=re.IGNORECASE)
    return ' '.join(player_name.split())

def strip_name_marks(s: object) -> object:
    """Strip common marks like '*' from names."""
    if not isinstance(s, str):
        return s
    return s.replace("*", "")


# ---- Numeric String Conversion ----

def to_num(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, stripping common non-numeric characters and handling percentages, rounds, etc."""
    s = series.astype(str).str.strip()
    s = (s.str.replace('%','',regex=False)
           .str.replace(r'(?i)round\s*','',regex=True)
           .str.replace(r'(?i)^r\s*','',regex=True)
           .str.replace(r'(?i)(st|nd|rd|th)$','',regex=True)
           .str.replace(',','',regex=False)
           .str.replace(r'[^0-9\.\-]','',regex=True))
    return pd.to_numeric(s, errors='coerce')

# ---- Find First Column ----

def find_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column in a DataFrame from a list of candidate names (case and space insensitive)."""
    norm = {re.sub(r"\s+","",c).lower(): c for c in frame.columns}
    for cand in candidates:
        key = re.sub(r"\s+","",cand).lower()
        if key in norm:
            return norm[key]
    return None


# ---- Height parsing ----
def height_to_inches(ht: object) -> Optional[float]:
    """Convert 5-11, 5/11, '5 11', or 5'11 to inches. Ignores date-like strings."""
    if ht is None or (isinstance(ht, float) and pd.isna(ht)):
        return None
    ht = str(ht).strip()

    for pat in (r"^(\d+)[-/](\d+)$", r"^(\d+)\s+(\d+)$", r"^(\d+)'\s*(\d+)$"):
        m = re.match(pat, ht)
        if m:
            feet, inches = map(int, m.groups())
            return float(feet * 12 + inches)

    # reject date-like tokens such as 05/31 or May/31
    if re.match(r"^\d{1,2}[-/]\d{1,2}$", ht):
        return None
    if re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/]\d{1,2}$", ht, flags=re.IGNORECASE):
        return None
    return None




# ---- #Target Position ----     #target_cands_fo_position in multi_tune script NEED TO EDIT
def target_position(pos: str) -> list[str]:
    """Return possible target column names for a given position (e.g., WR -> ['WR Grade', 'WRGrade', 'WR_Grade'])."""
    p = pos.upper()
    return [f"{p} Grade", f"{p}Grade", f"{p}_Grade"]


# ---- CSV IO ----
def safe_read_csv(path: str, *, encoding: str = "latin1", dtype="str", parse_dates: bool = False) -> pd.DataFrame | None:
    """Read CSV with consistent defaults, returning None on failure."""
    try:
        return pd.read_csv(path, encoding=encoding, dtype=dtype, parse_dates=parse_dates)
    except Exception:
        return None

def default_csv_for_position(project_root: Path, pos: str) -> Path:
    """Return the default CSV path for a given position."""
    return project_root / "data" / "Bakery" / pos.upper() / f"Bakery_{pos.upper()}_Overall.csv"



# ---- DataFrame utilities ----
def reorder_after(df: pd.DataFrame, move_col: str, after_col: str) -> pd.DataFrame:
    """Return df with `move_col` placed immediately after `after_col` (if both exist)."""
    if move_col not in df.columns or after_col not in df.columns:
        return df
    cols = list(df.columns)
    cols.remove(move_col)
    idx = cols.index(after_col)
    cols = cols[: idx + 1] + [move_col] + cols[idx + 1 :]
    return df[cols]

def filter_positions(df: pd.DataFrame, positions: Sequence[str]) -> pd.DataFrame:
    """Filter df to rows where Pos is in positions, if column exists."""
    if "Pos" in df.columns:
        return df[df["Pos"].isin(list(positions))].copy()
    return df

def parse_drafted_column(df: pd.DataFrame, col: str = "Drafted (tm/rnd/yr)") -> pd.DataFrame:
    """Split drafted column into Draft_Team/Round/Pick if present."""
    if col not in df.columns:
        return df
    split = df[col].str.extract(
        r"^(.*?)\s*/\s*(\d+(?:st|nd|rd|th))\s*/\s*(\d+(?:st|nd|rd|th)\s+pick)"
    )
    df["Draft_Team"] = split[0].str.strip()
    df["Draft_Round"] = split[1].str.strip()
    df["Draft_Pick"]  = split[2].str.strip()
    return df.drop(columns=[col])

def group_to_player_dict(df: pd.DataFrame, player_col: str = "player") -> Dict[str, pd.DataFrame]:
    """Group by player_col -> dict of player: DataFrame with index reset."""
    return {name: g.reset_index(drop=True) for name, g in df.groupby(player_col)}

# ---- Computations ----

def rmse(y_true, y_pred) -> float:
    """Compute root mean squared error between y_true and y_pred."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---- Log ----

def log(message: str, level: str = "info", verbose: bool = True, stream=sys.stdout):
    """
    Tiny logger for consistent printing.
    - message: the text to log
    - level: "info", "warn", "error", "success"
    - verbose: if False, suppress all logs
    - stream: defaults to stdout (use sys.stderr for errors if needed)
    """
    if not verbose:
        return

    prefix_map = {
        "info": "ℹ️ ",
        "warn": "⚠️ ",
        "error": "❌ ",
        "success": "✅ ",
    }
    prefix = prefix_map.get(level.lower(), "")
    print(f"{prefix}{message}", file=stream)


# ---- Invert Lower Better ----

def invert_cols(X: pd.DataFrame) -> pd.DataFrame:   # Replace invert_lower_better with invert_cols in older scripts
    """Invert columns where lower values are better (e.g., times, draft round, age) so higher is always better."""
    X = X.copy()
    for c in ["40 Time","Speed","Draft Capital","Shuttle","Three Cone","Draft Age"]:
        if c in X.columns:
            X[c] = -X[c]
    return X


# ---- Base Feature Interactions ----
def add_interactions(X, inter_names, inter_defs=None):
    """Add interaction features (only if both parents exist)."""
    if inter_defs is None:
        inter_defs = INTERACTIONS
    X = X.copy()
    for name in inter_names:
        a, b = inter_defs[name]
        if a in X.columns and b in X.columns:
            X[name] = X[a] * X[b]
    return X

def sample_subset(
    rng,
    *,
    base_pool,
    max_bases,
    allowed_inters,
    max_inters,
    must_feats,
    must_inters,
    interaction_hierarchy="weak",
):
    """
    Returns (bases, inter_names) subject to:
      - must_feats always included
      - must_inters always included (subject to hierarchy)
      - interaction_hierarchy rules:
          strong: parents must be present for ALL interactions
          weak  : parents must be present only for must_inters
          none  : interaction-only allowed (parents not required)
    """
    must_feats = list(dict.fromkeys(must_feats))
    remaining_pool = [f for f in base_pool if f not in must_feats]
    extra_cap = max(0, max_bases - len(must_feats))
    extras = []
    if remaining_pool and extra_cap > 0:
        k = int(rng.integers(0, min(extra_cap, len(remaining_pool)) + 1))
        if k > 0:
            extras = rng.choice(remaining_pool, size=k, replace=False).tolist()
    bases = must_feats + extras

    # FORCE PARENTS (depending on hierarchy)
    if interaction_hierarchy in ("strong", "weak"):
        force_for = list(allowed_inters.keys()) if interaction_hierarchy == "strong" else list(must_inters)
        for iname in force_for:
            if iname in allowed_inters:
                a, b = allowed_inters[iname]
                for p in (a, b):
                    if p not in bases and p in base_pool:
                        bases.append(p)

    # choose interactions: must + random others
    inter_names = list(must_inters)
    other_eligible = [n for n in allowed_inters if n not in set(inter_names)]
    other_cap = max(0, max_inters - len(inter_names))
    if other_eligible and other_cap > 0:
        k = int(rng.integers(0, min(other_cap, len(other_eligible)) + 1))
        if k > 0:
            inter_names += rng.choice(other_eligible, size=k, replace=False).tolist()

    # If hierarchy != "none", drop interactions whose parents are not present
    if interaction_hierarchy in ("strong", "weak"):
        inter_names = [n for n in inter_names if all(p in bases for p in allowed_inters[n])]

    return bases, inter_names


def limit_draft_capital_series(
    s,
    *,
    cap: float = 0.40,
    lower_q: float = 0.05,
    upper_q: float = 0.95
):
    """
    1) percentile-clip to remove extremes
    2) min-max normalize to [0,1]
    3) compress range to [1-cap, 1]
    We run this BEFORE invert_cols (sign flip later is fine).
    """
    import numpy as np
    import pandas as pd
    x = s.astype(float).copy()
    if not np.isfinite(x).any():
        return x.fillna(0.0)
    lo = np.nanpercentile(x, lower_q * 100)
    hi = np.nanpercentile(x, upper_q * 100)
    x = x.clip(lo, hi)
    rng = x.max() - x.min()
    if rng <= 1e-12:
        return pd.Series(np.zeros_like(x), index=x.index)
    x = (x - x.min()) / rng
    x = (1.0 - cap) + cap * x
    return x


# ---- NaN/Inf sanitizers ----
def sanitize_frame_global(X: pd.DataFrame, *, min_non_na_frac: float = 0.0) -> pd.DataFrame:
    """
    Global cleanup for X_all_raw:
      - replace ±inf with NaN
      - drop columns that are all NaN
      - optionally drop columns with non-NaN fraction < threshold
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    # drop all-NaN columns
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    if min_non_na_frac > 0.0:
        good = X.columns[(X.notna().mean() >= min_non_na_frac)]
        X = X[good]
    return X

def sanitize_train_test(Xtr_df: pd.DataFrame, Xte_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subset-level cleanup:
      - remove columns that are ALL NaN in TRAIN (imputer cannot fit them)
      - keep train/test aligned to the remaining columns
    """
    keep_cols = [c for c in Xtr_df.columns if Xtr_df[c].notna().any()]
    if len(keep_cols) != len(Xtr_df.columns):
        Xtr_df = Xtr_df[keep_cols]
        Xte_df = Xte_df[keep_cols]
    return Xtr_df, Xte_df