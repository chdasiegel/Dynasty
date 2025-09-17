# plot_scatter_wr_combine.py
"""
WR-only combine scatter plot: 40-yard dash (x) vs vertical jump (y).

- Labels: "F. Last (Pick)" placed exactly at the data point (no nudging).
- X-axis reversed (fastest on the left).
- Default limit: 50 WRs.
- Saves figure to Dynasty/tests/combine_wr_scatter.png by default.
- Optional pick filters (--pick-min/--pick-max) and UDFA inclusion (--include-udfa).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from src.process_combine import build_combine_dict

# Column fallbacks (edit if your headers differ)
PREFERRED_40_COLS   = ["40yd", "40Yd", "40-Yard", "40", "Forty", "40 yd"]
PREFERRED_VERT_COLS = ["Vertical", "Vert", "Vertical (in)", "VerticalJump"]
PREFERRED_POS_COLS  = ["Pos", "Position"]
PREFERRED_PICK_COLS = ["Pick", "Overall", "Draft_Pick", "DraftPick"]

# Project-root aware default data dir: <project_root>/data/Combine
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "Combine"
_DEFAULT_OUT_PATH = _PROJECT_ROOT / "tests" / "combine_wr_scatter.png"   # 👈 now "tests"


# ---------- helpers ----------
def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def _prepare_master_df(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for player, df in d.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            tmp = df.copy()
            tmp["player"] = player
            frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def name_to_initial_last(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    parts = name.split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {parts[-1]}"

def _pick_display(val) -> str:
    if pd.isna(val):
        return "UDFA"
    try:
        return f"#{int(float(val))}"
    except Exception:
        return str(val)

def sample_even(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    idx = (pd.Series(range(limit)) * (len(df) - 1) / max(1, limit - 1)).round().astype(int).tolist()
    return df.iloc[idx]

def filter_by_pick_range(df: pd.DataFrame, pick_col: Optional[str],
                         pick_min: Optional[int], pick_max: Optional[int],
                         include_udfa: bool) -> pd.DataFrame:
    if pick_col is None or pick_col not in df.columns:
        return df
    picks = _coerce_numeric(df[pick_col])
    if pick_min is None and pick_max is None:
        return df if include_udfa else df[~picks.isna()].copy()
    mask = ~picks.isna()
    if pick_min is not None:
        mask &= (picks >= pick_min)
    if pick_max is not None:
        mask &= (picks <= pick_max)
    return df[mask].copy()

def select_label_indices(df: pd.DataFrame, n: int, rule: str,
                         col_40: str, col_vert: str, pick_col: Optional[str]) -> pd.Index:
    """Choose which rows to label. rule: 'all' | 'top_picks' | 'extremes'."""
    if n >= len(df) or rule == "all":
        return df.index
    if rule == "top_picks" and pick_col and (pick_col in df.columns):
        picks = pd.to_numeric(df[pick_col], errors="coerce")
        idx = picks.dropna().sort_values().head(n).index
        if len(idx) > 0:
            return idx
    # 'extremes': half fastest 40s, half highest verticals
    n1 = n // 2
    n2 = n - n1
    idx1 = df.nsmallest(max(1, n1), col_40).index
    idx2 = df.nlargest(max(1, n2), col_vert).index
    return pd.Index(idx1).union(idx2)


# ---------- main ----------
def main(years: Iterable[int],
         data_dir: str | Path | None = None,
         limit: int = 50,
         out_path: str | Path | None = None,
         pick_min: Optional[int] = None,
         pick_max: Optional[int] = None,
         include_udfa: bool = False,
         max_labels: int = 35,
         label_rule: str = "extremes") -> None:

    base_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    save_path = Path(out_path) if out_path is not None else _DEFAULT_OUT_PATH

    # ensure output folder exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    d = build_combine_dict(years=years, data_dir=str(base_dir), verbose=True)
    if not d:
        raise SystemExit("No combine data found.")

    df = _prepare_master_df(d)
    if df.empty:
        raise SystemExit("Empty DataFrame after concatenation.")

    col_40   = _first_present(df, PREFERRED_40_COLS)
    col_vert = _first_present(df, PREFERRED_VERT_COLS)
    col_pos  = _first_present(df, PREFERRED_POS_COLS)
    col_pick = _first_present(df, PREFERRED_PICK_COLS)

    # numeric & bounds
    df[col_40] = _coerce_numeric(df[col_40])
    df[col_vert] = _coerce_numeric(df[col_vert])
    df = df.dropna(subset=[col_40, col_vert])
    df = df[(df[col_40] > 3.5) & (df[col_40] < 6.0) & (df[col_vert] > 15) & (df[col_vert] < 50)]

    # WR only
    wr = df[df[col_pos] == "WR"].copy()
    if wr.empty:
        raise SystemExit("No WR rows found.")

    # pick filters & sample cap
    wr = filter_by_pick_range(wr, col_pick, pick_min, pick_max, include_udfa)
    if wr.empty:
        raise SystemExit("No WR rows after draft pick filtering.")
    wr = sample_even(wr, limit)

    # plot
    fig, ax = plt.subplots(figsize=(12, 8.5), constrained_layout=True)
    ax.scatter(wr[col_40], wr[col_vert], s=30, alpha=0.9)
    ax.set_xlabel("40-yard dash (seconds)")
    ax.set_ylabel("Vertical jump (inches)")
    ax.invert_xaxis()  # fastest on the left

    title_bits = [f"WR (≤ {limit} players)"]
    if pick_min is not None or pick_max is not None:
        lo = pick_min if pick_min is not None else "-"
        hi = pick_max if pick_max is not None else "-"
        title_bits.append(f"Pick range: {lo}–{hi}")
    elif include_udfa:
        title_bits.append("UDFAs included")
    ax.set_title("Combine: 40 vs Vertical — " + " | ".join(title_bits))

    # which rows to label
    label_idx = select_label_indices(wr, n=max_labels, rule=label_rule,
                                     col_40=col_40, col_vert=col_vert, pick_col=col_pick)

    # exact-on-point labels, no nudging
    for i, row in wr.iterrows():
        if i not in label_idx:
            continue
        name = name_to_initial_last(row.get("player", "Unknown"))
        pick = f" ({_pick_display(row[col_pick])})" if (col_pick in wr.columns) else ""
        label = f"{name}{pick}"

        ax.annotate(
            label,
            (row[col_40], row[col_vert]),
            xytext=(0, 0), textcoords="offset points",  # no offset
            fontsize=9,
            ha="left", va="center",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            clip_on=True
        )

    ax.grid(alpha=0.15)
    fig.savefig(save_path, dpi=240)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(_DEFAULT_DATA_DIR),
                    help="Directory with {year}_Combine.csv (default: data/Combine)")
    ap.add_argument("--limit", type=int, default=50, help="Max number of WRs to plot (default 50)")
    ap.add_argument("--years", nargs="*", type=int, default=list(range(2016, 2026)), help="Years to include")
    ap.add_argument("--pick-min", type=int, default=None, help="Minimum overall draft pick (inclusive)")
    ap.add_argument("--pick-max", type=int, default=None, help="Maximum overall draft pick (inclusive)")
    ap.add_argument("--include-udfa", action="store_true", help="Include undrafted players when no pick filter is set")
    ap.add_argument("--max-labels", type=int, default=35, help="Max annotated labels (default 35)")
    ap.add_argument("--label-rule", type=str, default="extremes", choices=["all", "top_picks", "extremes"],
                    help="Which points to annotate when limiting labels")
    ap.add_argument("--out-path", type=str, default=None,
                    help="Optional override for save path (default: Dynasty/tests/combine_wr_scatter.png)")
    args = ap.parse_args()

    main(
        years=args.years,
        data_dir=args.data_dir,
        limit=args.limit,
        pick_min=args.pick_min,
        pick_max=args.pick_max,
        include_udfa=args.include_udfa,
        max_labels=args.max_labels,
        label_rule=args.label_rule,
        out_path=args.out_path,
    )
