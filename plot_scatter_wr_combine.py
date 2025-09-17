# plot_scatter_wr_combine.py
"""
WR-only combine scatter plot: 40-yard dash (x) vs vertical jump (y).

Enhancements:
- Labels show "F. Lastname" (first initial + full last name).
- Optional filtering by draft pick range via --pick-min / --pick-max.
- Optionally include UDFAs when no pick filter is applied.

Usage examples (from repo root):
    # Basic (limit 100 WRs, years 2016-2025)
    python plot_scatter_wr_combine.py

    # Filter to top-100 picks only
    python plot_scatter_wr_combine.py --pick-min 1 --pick-max 100

    # Show UDFAs too (only relevant when no pick filter set)
    python plot_scatter_wr_combine.py --include-udfa

Output:
    combine_wr_scatter.png
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Iterable, List, Optional
import pandas as pd
import matplotlib.pyplot as plt

from process_combine import build_combine_dict

PREFERRED_40_COLS = ["40yd", "40Yd", "40-Yard", "40", "Forty", "40 yd"]
PREFERRED_VERT_COLS = ["Vertical", "Vert", "Vertical (in)", "VerticalJump"]
PREFERRED_POS_COLS = ["Pos", "Position"]
PREFERRED_PICK_COLS = ["Pick", "Overall", "Draft_Pick", "DraftPick"]


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    # Strip non-numeric noise (e.g., "4.41s") and coerce
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
    """Convert 'First Middle Last' to 'F. Last'. If only one token, return as-is."""
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    tokens = name.strip().split()
    if len(tokens) == 1:
        return tokens[0]
    first = tokens[0]
    last = tokens[-1]  # Works for most cases
    return f"{first[0]}. {last}"


def _pick_display(val) -> str:
    if pd.isna(val):
        return "UDFA"
    try:
        return f"#{int(float(val))}"
    except Exception:
        return str(val)


def _sample_even(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    # Evenly spaced sampling (deterministic) to reduce bias
    idx = (pd.Series(range(limit)) * (len(df) - 1) / max(1, limit - 1)).round().astype(int).tolist()
    return df.iloc[idx]


def filter_by_pick_range(df: pd.DataFrame, pick_col: Optional[str],
                         pick_min: Optional[int], pick_max: Optional[int],
                         include_udfa: bool) -> pd.DataFrame:
    """
    Filter dataframe to rows whose overall pick is within [pick_min, pick_max].
    - If neither bound is provided: return df unchanged (but can optionally include/exclude UDFAs).
    - If any bound is provided: only numeric picks are considered; UDFAs are dropped.
    """
    if pick_col is None:
        return df  # nothing we can do

    # Coerce pick to numeric
    picks = _coerce_numeric(df[pick_col])

    # If no bounds given, just optionally drop UDFAs
    if pick_min is None and pick_max is None:
        if include_udfa:
            return df
        return df[~picks.isna()].copy()

    mask = pd.Series(True, index=df.index)
    if pick_min is not None:
        mask &= (picks >= pick_min)
    if pick_max is not None:
        mask &= (picks <= pick_max)

    mask &= ~picks.isna()
    return df[mask].copy()


def main(years: Iterable[int],
         data_dir: str,
         limit: int,
         out_path: str = "combine_wr_scatter.png",
         pick_min: Optional[int] = None,
         pick_max: Optional[int] = None,
         include_udfa: bool = False) -> None:

    d = build_combine_dict(years=years, data_dir=data_dir, verbose=True)
    if not d:
        raise SystemExit("No combine data found.")

    df = _prepare_master_df(d)
    if df.empty:
        raise SystemExit("Empty DataFrame after concatenation.")

    col_40 = _first_present(df, PREFERRED_40_COLS)
    col_vert = _first_present(df, PREFERRED_VERT_COLS)
    col_pos = _first_present(df, PREFERRED_POS_COLS)
    col_pick = _first_present(df, PREFERRED_PICK_COLS)

    missing = [name for name, col in [
        ("40-yard dash", col_40),
        ("vertical", col_vert),
        ("position", col_pos),
    ] if col is None]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}.")

    # Numeric cleaning and sanity filters
    df[col_40] = _coerce_numeric(df[col_40])
    df[col_vert] = _coerce_numeric(df[col_vert])
    df = df.dropna(subset=[col_40, col_vert])
    df = df[(df[col_40] > 3.5) & (df[col_40] < 6.0) & (df[col_vert] > 15) & (df[col_vert] < 50)]

    # WR only
    wr = df[df[col_pos] == "WR"].copy()
    if wr.empty:
        raise SystemExit("No WR rows found.")

    # Apply pick filtering
    wr = filter_by_pick_range(wr, col_pick, pick_min, pick_max, include_udfa)

    if wr.empty:
        raise SystemExit("No WR rows remain after draft pick filtering.")

    wr = _sample_even(wr, limit)

    # Plot
    plt.figure(figsize=(11, 8))
    plt.scatter(wr[col_40], wr[col_vert])
    plt.xlabel("40-yard dash (seconds)")
    plt.ylabel("Vertical jump (inches)")
    plt.title(f"Combine: 40 vs Vertical — WR (≤ {limit} players)")

    # Annotate points with "F. Last (Pick)"
    for _, row in wr.iterrows():
        label = name_to_initial_last(row.get("player", "Unknown"))
        if col_pick in wr.columns:
            label = f"{label} ({_pick_display(row[col_pick])})"
        plt.annotate(label, (row[col_40], row[col_vert]), xytext=(3, 2), textcoords="offset points", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="Combine", help="Directory containing {year}_Combine.csv")
    ap.add_argument("--limit", type=int, default=100, help="Max number of WRs to plot")
    ap.add_argument("--years", nargs="*", type=int, default=list(range(2016, 2026)), help="Years to include")
    ap.add_argument("--pick-min", type=int, default=None, help="Minimum overall draft pick (inclusive)")
    ap.add_argument("--pick-max", type=int, default=None, help="Maximum overall draft pick (inclusive)")
    ap.add_argument("--include-udfa", action="store_true",
                    help="Include undrafted players when no pick filter is set")
    args = ap.parse_args()
    main(years=args.years, data_dir=args.data_dir, limit=args.limit,
         pick_min=args.pick_min, pick_max=args.pick_max, include_udfa=args.include_udfa)
