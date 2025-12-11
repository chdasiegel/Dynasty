#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
add_pff_to_dom.py

Takes the wide DOM file produced by dom_from_stathead.py (StatHead-only)
and enriches it with PFF per-season metrics based on:

    - Player name (normalized)
    - Year (from the Year1–Year5 columns)
    - PFF CSV filenames (year extracted from filename)
    - Player position (pos) to decide which PFF table to use

Behavior
--------
- Reads PFF CSVs from:
    ROOT/data/CFB_Data/PFF/Receiving/*.csv
    ROOT/data/CFB_Data/PFF/Rushing/*.csv
    ROOT/data/CFB_Data/PFF/Passing/*.csv

- Extracts year from filename via regex (e.g. "something_2025_REGPO.csv" -> 2025)
- Normalizes player names so "Last, First" ≈ "First Last"
- Builds 3 separate long PFF tables (Receiving, Rushing, Passing) with:
      player, team, pos, year_int, [all PFF columns as-is], player_norm_merge

- For each row in the DOM master:
    * Uses `pos` to decide which PFF table(s) to query:
        - WR / TE: Receiving only
        - RB / HB : Rushing only
        - QB      : Passing + Rushing (merged per metric; see below)
    * For each slot s in 1..5:
        - looks at Year{s}
        - finds PFF row(s) for (player_norm_merge, year_int)
        - writes selected PFF metrics into "<metric_name>{s}", e.g. "yprr1"

- QB merging rules for overlapping metrics:
    - For "fumbles", pass + rush values are SUMMED (if either is present).
    - For other metrics, prefer passing value; if NaN/missing, fall back to rushing.

- DOM & raw StatHead stats are left untouched, but at the very end:
    The output CSV is reduced to ONLY:
        - IDENTITY_COLS
        - Year1–Year5
        - KEEP_BASE_COLS expanded to per-year columns (metric1..metric5)

No `_x` / `_y` suffixes are produced by this script.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ====== PATH CONFIG ======
ROOT = Path("/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PFF_DIR      = ROOT / "data" / "CFB_Data" / "PFF"
PFF_PASS_DIR = PFF_DIR / "Passing"
PFF_RUSH_DIR = PFF_DIR / "Rushing"
PFF_REC_DIR  = PFF_DIR / "Receiving"

# ====== FEATURE SELECTION CONFIG (ALWAYS APPLIED) ======

# Identity / draft metadata that should always be kept
IDENTITY_COLS: List[str] = [
    "player_name",
    "pos",
    "team",
    "conf",
    "draft_year",
    "draft_round",
    "pick_in_round",
    "pick_overall",
]

# Base metric names (no Year suffix) that you want to KEEP.
# These will be expanded to Year1–Year5 columns (metric1, metric2, ..., metric5).
KEEP_BASE_COLS: List[str] = [
    # Receiving / route metrics
    "avg_depth_of_target",
    "avoided_tackles",
    "caught_percent",
    "contested_catch_rate",
    "contested_receptions",
    "contested_targets",
    "drop_rate",
    "drops",
    "receptions",
    "route_rate",
    "routes",
    "slot_rate",
    "slot_snaps",
    "targets",
    "wide_rate",
    "wide_snaps",
    "yards_after_catch",
    "yards_after_catch_per_reception",
    "yards_per_reception",
    "yprr",

    # Rushing / elusiveness
    "attempts",
    "breakaway_attempts",
    "breakaway_percent",
    "breakaway_yards",
    "designed_yards",
    "elu_recv_mtf",
    "elu_rush_mtf",
    "elu_yco",
    "elusive_rating",
    "explosive",
    "gap_attempts",
    "rec_yards",
    "run_plays",
    "scramble_yards",
    "scrambles",
    "total_touches",
    "yards_after_contact",
    "yco_attempt",
    "zone_attempts",

    # Passing / QB metrics
    "accuracy_percent",
    "aimed_passes",
    "avg_time_to_throw",
    "big_time_throws",
    "btt_rate",
    "completion_percent",
    "completions",
    "def_gen_pressures",
    "dropbacks",
    "hit_as_threw",
    "interceptions",
    "passing_snaps",
    "pressure_to_sack_rate",
    "qb_rating",
    "sack_percent",
    "sacks",
    "turnover_worthy_plays",
    "twp_rate",
    "ypa",

    # Ball security
    "fumbles",

    # DOM & StatHead outputs (already in DOM file)
    "DOM",
    "DOM+",
    "PDOM",
    "PDOM+",
    "RDOM",
    "RDOM+",
    "rec_yds",
    "rec_tds",
    "rush_yds",
    "rush_tds",
    "pass_yds",
    "pass_tds",
    "team_pr_yds",
    "team_pr_tds",
    "team_ru_yds",
    "team_ru_tds",
]

def expand_with_slots(base_names: List[str], slots=range(1, 6)) -> List[str]:
    """
    Expand base metric names into all Year-slot variants, e.g.:

        "yards" → ["yards", "yards1", "yards2", ..., "yards5"]
    No _x / _y variants are used.
    """
    out = set()
    for b in base_names:
        out.add(b)
        for s in slots:
            out.add(f"{b}{s}")
    return sorted(out)

def compute_keep_columns(df_columns: List[str]) -> List[str]:
    """
    Given the final df.columns, compute which columns to keep based on:
      - Always keep IDENTITY_COLS and Year1–Year5
      - KEEP_BASE_COLS: keep ONLY those metric families
        (expanded across slots) in addition to the above.
      - If KEEP_BASE_COLS is empty: keep only IDENTITY_COLS + Year1–Year5.
    """
    cols = list(df_columns)

    # Year columns that exist
    year_cols = [c for c in cols if re.fullmatch(r"Year[1-5]", c)]
    slots = [int(m.group(1)) for c in year_cols for m in [re.fullmatch(r"Year([1-5])", c)] if m]

    # Identity + Year columns always kept
    keep_set = set(IDENTITY_COLS) | set(year_cols)

    if KEEP_BASE_COLS:
        keep_set |= set(expand_with_slots(KEEP_BASE_COLS, slots))

    # Final list, preserving original order
    final_cols = [c for c in cols if c in keep_set]
    return final_cols

# ====== NAME NORMALIZATION HELPERS ======

def normalize_name(raw: str) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw).lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_name_for_merge(raw: str) -> str:
    """
    Normalize for joining:
      - if 'Last, First' -> 'first last'
      - then lowercase, remove punctuation, sort tokens
    This makes 'Smith, John A.' and 'John Smith' converge.
    """
    if pd.isna(raw):
        return ""
    s = str(raw).strip()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    base = normalize_name(s)
    tokens = base.split()
    tokens.sort()
    return " ".join(tokens)

# ====== GENERIC HELPERS ======

def find_first_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def extract_year_from_filename(path: Path) -> Optional[int]:
    """
    Look for a 4-digit 20xx year in the filename.
    """
    m = re.search(r"(20[0-9]{2})", path.name)
    if not m:
        return None
    return int(m.group(1))

# ====== PFF LOADING ======

def load_pff_dir(dirpath: Path, stat: str) -> pd.DataFrame:
    """
    Load ALL PFF CSVs in a directory, infer year from filename, normalize to:
      player, team, pos, year_int, [ALL other PFF columns as-is], player_norm_merge
    """
    if not dirpath.exists():
        print(f"[WARN] PFF directory does not exist: {dirpath}")
        return pd.DataFrame()

    frames = []

    for csv_path in sorted(dirpath.glob("*.csv")):
        year = extract_year_from_filename(csv_path)
        if year is None:
            print(f"[WARN] Could not infer year from PFF file name: {csv_path.name}")
            continue

        df_raw = pd.read_csv(csv_path)
        cols = list(df_raw.columns)

        player_col = find_first_col(cols, ["player", "Player", "player_name", "Name", "Player Name"])
        team_col   = find_first_col(cols, ["Team", "team", "School", "team_name"])
        pos_col    = find_first_col(cols, ["Pos", "Position", "position"])

        out = pd.DataFrame()

        # Player
        if player_col:
            out["player"] = df_raw[player_col]
        else:
            out["player"] = df_raw.index.astype(str)

        # Team
        if team_col:
            out["team"] = df_raw[team_col]
        else:
            out["team"] = ""

        # Position
        out["pos"] = df_raw[pos_col] if pos_col else ""

        # Year (from filename)
        out["year_int"] = year

        # Copy ALL other columns as-is (no alias / renaming)
        skip_cols = set(c for c in [player_col, team_col, pos_col] if c is not None)

        for c in df_raw.columns:
            if c in skip_cols:
                continue
            if c not in out.columns:
                out[c] = df_raw[c]
            else:
                # This should be rare; if it happens, last write wins.
                out[c] = df_raw[c]

        # Name normalization key
        out["player_norm_merge"] = out["player"].map(normalize_name_for_merge)

        frames.append(out)

        print(f"[INFO] Loaded PFF {stat} file {csv_path.name} → {len(out)} rows, year={year}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate by (player_norm_merge, year_int)
    subset_cols = ["player_norm_merge", "year_int", "pos"] + [
        c for c in df.columns if c not in ("player_norm_merge", "year_int", "pos")
    ]
    df = df[subset_cols]
    df = df.drop_duplicates(["player_norm_merge", "year_int", "pos"], keep="first")

    return df

def build_pff_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build separate Receiving, Rushing, Passing PFF tables:
      columns: player_norm_merge, year_int, pos, [all PFF metrics as-is]
    """
    rec_df  = load_pff_dir(PFF_REC_DIR,  "rec")
    rush_df = load_pff_dir(PFF_RUSH_DIR, "rush")
    pass_df = load_pff_dir(PFF_PASS_DIR, "pass")

    print(f"[INFO] Receiving PFF rows: {len(rec_df)}")
    print(f"[INFO] Rushing   PFF rows: {len(rush_df)}")
    print(f"[INFO] Passing   PFF rows: {len(pass_df)}")

    return rec_df, rush_df, pass_df

# ====== APPLY PFF TO WIDE DOM FILE ======

def enrich_dom_with_pff(dom_csv: Path, out_csv: Path) -> None:
    # Load DOM master
    df = pd.read_csv(dom_csv)

    # Capture original column order (before we add PFF metrics)
    original_cols = list(df.columns)

    # Determine player name column
    player_col = find_first_col(list(df.columns), ["player", "player_name", "Player", "PlayerName"])
    if not player_col:
        raise ValueError("Could not find player name column in DOM CSV.")

    # Normalize player name for PFF joining
    df["player_norm_merge"] = df[player_col].map(normalize_name_for_merge)

    # Position column
    if "pos" not in df.columns:
        raise ValueError("DOM CSV must contain a 'pos' column for position-based PFF selection.")

    # Build PFF tables
    pff_rec, pff_rush, pff_pass = build_pff_tables()
    if pff_rec.empty and pff_rush.empty and pff_pass.empty:
        print("[INFO] No PFF data loaded; writing original DOM file unchanged.")
        df.drop(columns=["player_norm_merge"], inplace=True, errors="ignore")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return

    # Build indices for quick lookup
    rec_index  = pff_rec.set_index(["year_int", "player_norm_merge"])  if not pff_rec.empty  else None
    rush_index = pff_rush.set_index(["year_int", "player_norm_merge"]) if not pff_rush.empty else None
    pass_index = pff_pass.set_index(["year_int", "player_norm_merge"]) if not pff_pass.empty else None

    # Figure out which Year slots we have
    slots = [s for s in range(1, 6) if f"Year{s}" in df.columns]
    if not slots:
        print("[WARN] No Year1–Year5 columns found in DOM CSV.")
        df.drop(columns=["player_norm_merge"], inplace=True, errors="ignore")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return

    # Determine which PFF metric columns we want to pull (only those in KEEP_BASE_COLS)
    # based on union of columns across rec/rush/pass.
    pff_all_cols = set()
    for d in (pff_rec, pff_rush, pff_pass):
        if not d.empty:
            pff_all_cols.update(d.columns.tolist())

    pff_metric_cols = [
        c for c in pff_all_cols
        if c not in ("player", "team", "pos", "year_int", "player_norm_merge")
        and (not KEEP_BASE_COLS or c in KEEP_BASE_COLS)
    ]
    pff_metric_cols = sorted(pff_metric_cols)
    print(f"[INFO] PFF metric columns used: {pff_metric_cols}")

    # Ensure metric columns exist for each slot e.g. "yprr" -> "yprr1", "yprr2", ...
    for s in slots:
        for m in pff_metric_cols:
            col = f"{m}{s}"
            if col not in df.columns:
                df[col] = np.nan

    matched_count = 0

    # Enrich row-by-row
    for idx, row in df.iterrows():
        pname_key = row["player_norm_merge"]
        if not isinstance(pname_key, str) or not pname_key:
            continue

        pos_val = str(row.get("pos", "")).upper()

        is_wr_te = pos_val in {"WR", "TE"}
        is_rb    = pos_val in {"RB", "HB"}
        is_qb    = pos_val == "QB"

        for s in slots:
            year_col = f"Year{s}"
            year_val = row.get(year_col, np.nan)
            if pd.isna(year_val):
                continue

            try:
                year_int = int(year_val)
            except Exception:
                continue

            key = (year_int, pname_key)

            rec_row  = None
            rush_row = None
            pass_row = None

            if is_wr_te and rec_index is not None and key in rec_index.index:
                rec_row = rec_index.loc[key]

            if is_rb and rush_index is not None and key in rush_index.index:
                rush_row = rush_index.loc[key]

            if is_qb:
                if pass_index is not None and key in pass_index.index:
                    pass_row = pass_index.loc[key]
                if rush_index is not None and key in rush_index.index:
                    rush_row = rush_index.loc[key]

            # Skip if nothing found
            if not any([rec_row is not None, rush_row is not None, pass_row is not None]):
                continue

            matched_count += 1

            # If any of these are DataFrames (multirow), reduce to first
            if isinstance(rec_row, pd.DataFrame):
                rec_row = rec_row.iloc[0]
            if isinstance(rush_row, pd.DataFrame):
                rush_row = rush_row.iloc[0]
            if isinstance(pass_row, pd.DataFrame):
                pass_row = pass_row.iloc[0]

            for m in pff_metric_cols:
                dest_col = f"{m}{s}"
                val = np.nan

                if is_wr_te:
                    src = rec_row
                    if src is not None and m in src.index:
                        val = src[m]

                elif is_rb:
                    src = rush_row
                    if src is not None and m in src.index:
                        val = src[m]

                elif is_qb:
                    pr_val = np.nan
                    rr_val = np.nan
                    if pass_row is not None and m in pass_row.index:
                        pr_val = pass_row[m]
                    if rush_row is not None and m in rush_row.index:
                        rr_val = rush_row[m]

                    if m == "fumbles":
                        # Sum fumbles from pass + rush if present
                        has_pr = pd.notna(pr_val)
                        has_rr = pd.notna(rr_val)
                        if has_pr or has_rr:
                            val = (0 if not has_pr else pr_val) + (0 if not has_rr else rr_val)
                    else:
                        # Prefer passing, fall back to rushing
                        if pd.notna(pr_val):
                            val = pr_val
                        elif pd.notna(rr_val):
                            val = rr_val

                if pd.notna(val):
                    df.at[idx, dest_col] = val

    print(f"[INFO] PFF season matches written: {matched_count}")

    # Clean up helper col
    df.drop(columns=["player_norm_merge"], inplace=True, errors="ignore")

    # ====== REORDER COLUMNS: group PFF metrics by Year1, Year2, ... ======
    new_order: List[str] = []

    # Start with original columns, and after each Year{s} insert its PFF metric columns
    for col in original_cols:
        if col == "player_norm_merge":
            continue
        new_order.append(col)

        m = re.fullmatch(r"Year([1-5])", col)
        if m:
            s = int(m.group(1))
            for metric_name in pff_metric_cols:
                metric_col = f"{metric_name}{s}"
                if metric_col in df.columns and metric_col not in new_order:
                    new_order.append(metric_col)

    # Add any remaining columns that weren't in original_cols (safety net)
    for col in df.columns:
        if col not in new_order:
            new_order.append(col)

    df = df[new_order]

    # ====== ALWAYS APPLY KEEP-ONLY LOGIC ======
    final_cols = compute_keep_columns(list(df.columns))
    print(f"[INFO] Reducing columns from {len(df.columns)} to {len(final_cols)} using KEEP_BASE_COLS + IDENTITY_COLS.")
    df = df[final_cols]

    # Save enriched file
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving PFF-enriched DOM file to {out_csv}")
    df.to_csv(out_csv, index=False)

# ====== CLI ENTRYPOINT ======

def main():
    parser = argparse.ArgumentParser(
        description="Add PFF per-season metrics to wide DOM file (Year1–Year5) using player name + year + position."
    )
    parser.add_argument("--dom_csv", type=Path, required=True,
                        help="DOM CSV produced by dom_from_stathead.py")
    parser.add_argument("--out_csv", type=Path, required=True,
                        help="Output CSV with PFF metrics added")
    args = parser.parse_args()

    enrich_dom_with_pff(args.dom_csv, args.out_csv)

if __name__ == "__main__":
    main()
