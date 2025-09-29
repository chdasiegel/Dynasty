# src/visuals/plot_feature_scatter_batch.py
"""
Batch scatterplots from Bakery_RB_Overall.csv using BASE_FEATURES from your features_config.

Examples
--------
# Default: read BASE_FEATURES from src.config.features_config, plot up to 20 pairs
python -m src.visuals.plot_feature_scatter_batch \
  --csv ./data/Bakery/RB/Bakery_RB_Overall.csv \
  --out-dir ./tests/feature_scatter \
  --max-plots 20

# Pin X features; Y = all other base features (up to 40 plots)
python -m src.visuals.plot_feature_scatter_batch \
  --x "40 Time" BMI \
  --max-plots 40

# Choose exact pairs
python -m src.visuals.plot_feature_scatter_batch \
  --pairs "40 Time:BMI" "ELU:YCO/A" "DOM++:Draft Capital"
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd

# ---------- defaults ----------
try:
    _HERE = Path(__file__).resolve()
except NameError:
    # jupyter/notebook case
    _HERE = Path.cwd()

_PROJECT_ROOT = _HERE.parents[2] if "src" in _HERE.parts else _HERE
_DEFAULT_CSV  = _PROJECT_ROOT / "data" / "Bakery" / "RB" / "Bakery_RB_Overall.csv"
_DEFAULT_OUT  = _PROJECT_ROOT / "tests" / "feature_scatter"


# features where "smaller is better" — we flip X-axis for readability
LOWER_BETTER_X = {"40 Time","Shuttle","Three Cone","Draft Capital","Draft Age"}

# simple numeric sanity filters you can tweak (applied if the column name matches)
PLAU_LIMITS = {
    "40 Time":    (3.8, 6.2),
    "BMI":        (18, 45),
    "Draft Age":  (19, 26),
    "Shuttle":    (3.5, 5.0),
    "Three Cone": (6.0, 8.5),
    "YPC":        (2.0, 10.0),
    "ELU":        (0, 200),
    "YCO/A":      (0.0, 6.0),
    "Break%":     (0, 100),
    "DOM++":      (0, 100),
}

# ---------- helpers ----------
def to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = (s.str.replace("%","",regex=False)
          .str.replace(",","",regex=False)
          .str.replace(r"[^0-9.\-]", "", regex=True))
    return pd.to_numeric(s, errors="coerce")

def first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def resolve_feature_column(df: pd.DataFrame, feature: str, aliases: Dict[str, List[str]]) -> Optional[str]:
    cands = [feature] + list(aliases.get(feature, []))
    # Normalize candidate spacing to match columns with/without extra spaces
    cols_norm = {re.sub(r"\s+","",c).lower(): c for c in df.columns}
    for cand in cands:
        key = re.sub(r"\s+","",cand).lower()
        if key in cols_norm:
            return cols_norm[key]
    # Fallback: direct match if present
    return feature if feature in df.columns else None

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def pick_indices_for_labels(df: pd.DataFrame, x: str, y: str, n: int, rule: str) -> pd.Index:
    if n <= 0 or n >= len(df) or rule == "all":
        return df.index
    if rule == "extremes":
        n1 = n // 2
        n2 = n - n1
        idx1 = df.nsmallest(max(1, n1), x).index  # fastest / smaller x
        idx2 = df.nlargest(max(1, n2), y).index  # biggest y
        return pd.Index(idx1).union(idx2)
    return df.sample(n=min(n, len(df)), random_state=42).index  # fallback: random

def name_to_initial_last(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {parts[-1]}"

def bounds_filter(df: pd.DataFrame, col: str) -> pd.Series:
    if col in PLAU_LIMITS:
        lo, hi = PLAU_LIMITS[col]
        return df[col].between(lo, hi, inclusive="both")
    return pd.Series([True]*len(df), index=df.index)

# ---------- core plotting ----------
def plot_pair(df: pd.DataFrame,
              name_col: str,
              x_col: str,
              y_col: str,
              out_path: Path,
              max_labels: int = 35,
              label_rule: str = "extremes") -> None:
    work = df[[name_col, x_col, y_col]].copy()
    work[x_col] = to_num(work[x_col])
    work[y_col] = to_num(work[y_col])
    work = work.dropna(subset=[x_col, y_col])

    # sanity bounds (if we know them)
    mask = bounds_filter(work, x_col) & bounds_filter(work, y_col)
    work = work[mask]
    if work.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    ax.scatter(work[x_col], work[y_col], s=28, alpha=0.9)

    # reverse x for “lower is better” metrics
    if any(k.lower() == x_col.lower() for k in LOWER_BETTER_X):
        ax.invert_xaxis()

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")

    # choose labels
    idx = pick_indices_for_labels(work, x_col, y_col, max_labels, label_rule)
    for i, row in work.loc[idx].iterrows():
        label = name_to_initial_last(row[name_col])
        ax.annotate(
            label,
            (row[x_col], row[y_col]),
            xytext=(0, 0), textcoords="offset points",
            fontsize=9, ha="left", va="center",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            clip_on=True
        )
    ax.grid(alpha=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

# ---------- main ----------
def main(csv: str | Path = _DEFAULT_CSV,
         out_dir: str | Path = _DEFAULT_OUT,
         config_module: str = "src.config.features_config",
         x: Optional[List[str]] = None,
         y: Optional[List[str]] = None,
         pairs: Optional[List[str]] = None,
         max_plots: int = 20,
         max_labels: int = 35,
         label_rule: str = "extremes",
         name_candidates: Sequence[str] = ("Player","Player Name","Name")) -> None:

    # import features_config (expects BASE_FEATURES, ALIASES)
    mod = import_module(config_module)
    BASE_FEATURES = getattr(mod, "BASE_FEATURES")
    ALIASES       = getattr(mod, "ALIASES")

    csv = Path(csv)
    out_dir = Path(out_dir)
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]

    # resolve name column
    name_col = first_present(df, list(name_candidates)) or "Player"
    if name_col not in df.columns:
        raise SystemExit(f"Could not find a player name column among {name_candidates}")

    # resolve columns for all base features
    feature_col_map: Dict[str, str] = {}
    for feat in BASE_FEATURES:
        col = resolve_feature_column(df, feat, ALIASES)
        if col:
            feature_col_map[feat] = col

    if not feature_col_map:
        raise SystemExit("No BASE_FEATURES could be mapped to columns. Check ALIASES and CSV headers.")

    # figure out which pairs to plot
    pairs_to_plot: List[Tuple[str,str]] = []

    if pairs:
        # explicit "X:Y" requests
        for p in pairs:
            if ":" not in p:
                continue
            a, b = p.split(":", 1)
            a, b = a.strip(), b.strip()
            if a in feature_col_map and b in feature_col_map:
                pairs_to_plot.append((a, b))
    else:
        # sampled pairs from x/y or from all BASE_FEATURES
        xs = x if x else list(BASE_FEATURES)
        ys = y if y else list(BASE_FEATURES)
        # remove identical feature pairs and duplicates
        all_pairs = [(a,b) for a,b in itertools.product(xs, ys) if a != b]
        # prefer time->others first if present
        prefer = ["40 Time", "BMI", "DOM++", "ELU"]
        ordered = sorted(all_pairs, key=lambda ab: (prefer.index(ab[0]) if ab[0] in prefer else 999, ab[1]))
        pairs_to_plot = ordered[:max_plots]

    if not pairs_to_plot:
        raise SystemExit("No pairs selected to plot.")

    # plot each pair
    made = 0
    for a, b in pairs_to_plot:
        if a not in feature_col_map or b not in feature_col_map:
            continue
        x_col = feature_col_map[a]
        y_col = feature_col_map[b]
        fname = sanitize_filename(f"{b}_vs_{a}.png")
        plot_pair(
            df, name_col,
            x_col=x_col, y_col=y_col,
            out_path=out_dir / fname,
            max_labels=max_labels, label_rule=label_rule
        )
        made += 1

    print(f"Saved {made} plots → {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(_DEFAULT_CSV),
                    help="Path to Bakery_RB_Overall.csv")
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT),
                    help="Directory to save plots")
    ap.add_argument("--config-module", type=str, default="src.config.features_config",
                    help="Python path to features_config (exports BASE_FEATURES, ALIASES)")
    ap.add_argument("--x", nargs="*", default=None,
                    help="Optional list of X features to use (names from BASE_FEATURES)")
    ap.add_argument("--y", nargs="*", default=None,
                    help="Optional list of Y features to use (names from BASE_FEATURES)")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help='Explicit pairs like "40 Time:BMI" "ELU:YCO/A" (overrides --x/--y)')
    ap.add_argument("--max-plots", type=int, default=20,
                    help="Cap the number of plots generated")
    ap.add_argument("--max-labels", type=int, default=35,
                    help="Max labels per plot (placed directly on points)")
    ap.add_argument("--label-rule", type=str, default="extremes",
                    choices=["all","extremes","random"],
                    help="Which points to annotate")
    args = ap.parse_args()

    main(
        csv=args.csv,
        out_dir=args.out_dir,
        config_module=args.config_module,
        x=args.x,
        y=args.y,
        pairs=args.pairs,
        max_plots=args.max_plots,
        max_labels=args.max_labels,
        label_rule=args.label_rule,
    )
