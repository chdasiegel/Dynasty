#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dom_from_stathead_allcols.py — Combine ALL Stathead player CSV columns into a player-year record
and compute Dominator (DOM / DOM+) with team totals (prefer explicit passing/rushing CSVs, then globs).

===============================================================================
USAGE FROM A NOTEBOOK (your notebooks live in: Dynasty/notebooks)
-------------------------------------------------------------------------------
# 1) Make sure these files exist (empty is fine):
#    Dynasty/src/__init__.py
#    Dynasty/src/pipelines/__init__.py

# 2) In the FIRST cell:
import sys, os
PROJECT_ROOT = os.path.abspath("..")      # -> Dynasty/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
%load_ext autoreload
%autoreload 2

# 3) Import and run
from src.pipelines.dom_from_stathead_allcols import (
    run_stathead_player_year_pipeline,
    summarize_skips,
)

player_root = "../data/CFB_Data/StatHead"   # expects player_{year}/ subfolders with CSVs

# Your explicit team totals CSVs (read FIRST and prioritized):
teams_passing = "/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead/team_total_passing/team_passing_2000_2024.csv"
teams_rushing = "/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead/team_total_rushing/team_rushing_2000_2024.csv"

# Optional supplemental globs (read AFTER the two canonical files; fill gaps only)
team_globs  = [f"{player_root}/teams_*.csv", f"{player_root}/*teams*.csv"]

player_dict, df_player_year, skip_df = run_stathead_player_year_pipeline(
    player_root=player_root,
    team_globs=team_globs,                   # optional
    teams_passing_path=teams_passing,        # prioritized
    teams_rushing_path=teams_rushing,        # prioritized
    start=2013, end=2024,
    write_csv=False
)

display(df_player_year.head(10))
display(summarize_skips(skip_df))

# Optional exports from the notebook:
from pathlib import Path
out_dir = Path("../data/processed")
out_dir.mkdir(parents=True, exist_ok=True)
df_player_year.to_csv(out_dir/"dom_stathead_allcols_2013_2024.csv", index=False)
if not skip_df.empty:
    skip_df.to_csv(out_dir/"dom_stathead_allcols_skipped_2013_2024.csv", index=False)

===============================================================================
USAGE FROM TERMINAL (project root = Dynasty/)
-------------------------------------------------------------------------------
python -m src.pipelines.dom_from_stathead_allcols \
  --player-root "Dynasty/data/CFB_Data/StatHead" \
  --teams-passing "/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead/team_total_passing/team_passing_2000_2024.csv" \
  --teams-rushing "/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead/team_total_rushing/team_rushing_2000_2024.csv" \
  --teams "Dynasty/data/CFB_Data/StatHead/*teams*.csv" \
  --start 2013 --end 2024

# Outputs (when --start/--end provided and not using --no-save):
#   Dynasty/data/processed/dom_stathead_allcols_{start}_{end}.csv
#   Dynasty/data/processed/dom_stathead_allcols_skipped_{start}_{end}.csv

===============================================================================
BEHAVIOR & PRIORITY
-------------------------------------------------------------------------------
TEAM TOTALS PRIORITY:
  1) Explicit CSVs (passing + rushing) — REQUIRED first pass
     * passing -> provides team_receiving_yards/team_receiving_tds (as pass yards/TDs)
     * rushing -> currently not needed for DOM; merged for completeness if desired
  2) Globs (e.g., *teams*.csv) — fill ONLY missing team_year totals
  3) If still missing after (1)+(2), DOM is skipped and logged.

DOM rules:
  * Uses player receiving_yards + receiving_tds (auto-detected) and team totals above.
  * Multi-team year -> DOM skipped (logged) but row kept.
  * Team totals both zero -> DOM = 0, but logged as 'zero_team_totals_used'.

Dictionary:
  {player: {year: { …ALL normalized Stathead columns…,
                    team (possibly "TeamA | TeamB"), conference,
                    team_receiving_yards, team_receiving_tds, DOM, DOM_plus }, ...}, ...}

Save this file at: Dynasty/src/pipelines/dom_from_stathead_allcols.py
===============================================================================
"""

from __future__ import annotations
import os
import re
import glob
import math
import itertools
from typing import Dict, List, Optional, Tuple

import pandas as pd
import unicodedata


# ------------------------------ Config ------------------------------
OUT_DIR = "Dynasty/data/processed"  # processed/ holds derived/clean outputs

CONF_MULT: Dict[str, float] = {
    "SEC": 1.0,
    "BIG TEN": 1.0, "BIG 10": 1.0, "B1G": 1.0,
    "BIG 12": 0.95,
    "ACC": 0.95,
    "PAC-12": 0.95, "PAC 12": 0.95, "PAC12": 0.95,
    "AAC": 0.85,
    "MOUNTAIN WEST": 0.76, "MWC": 0.76,
    "SUN BELT": 0.76,
    "MAC": 0.70,
    "C-USA": 0.70, "CONFERENCE USA": 0.70, "CUSA": 0.70,
    "DII": 0.50,
    "OTHER": 0.60, "FCS": 0.60, "INDEPENDENT": 0.60, "NON-FBS": 0.60
}


# ----------------------------- Helpers ------------------------------
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_conf(conf: str) -> str:
    c = normalize_text(conf).upper().replace("-", " ").strip()
    c = c.replace("PAC 10", "PAC 12") \
         .replace("BIG TEN CONFERENCE", "BIG TEN") \
         .replace("ATLANTIC COAST CONFERENCE", "ACC") \
         .replace("SOUTHEASTERN CONFERENCE", "SEC")
    if c in CONF_MULT:
        return c
    for k in CONF_MULT:
        if k in c:
            return k
    return "OTHER"


def conf_multiplier(conf: str) -> float:
    return CONF_MULT.get(normalize_conf(conf), CONF_MULT["OTHER"])


def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        s = str(x).replace(",", "").strip()
        if s == "" or s.lower() in {"na", "n/a", "nan", "none", "--"}:
            return None
        return float(s)
    except Exception:
        return None


def snake(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", str(s))
    s = re.sub(r"\s+", "_", s.strip().lower())
    return s


def lower_map(cols: List[str]) -> Dict[str, str]:
    return {str(c).lower(): c for c in cols}


def pick_one(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the FIRST matching column (case-insensitive, substring ok)."""
    if df is None or df.empty:
        return None
    lmap = lower_map(df.columns.tolist())
    lowers = list(lmap.keys())
    for cand in candidates:
        c = cand.lower()
        if c in lmap:
            return lmap[c]
    for cand in candidates:
        c = cand.lower()
        for L in lowers:
            if c == L or c in L:
                return lmap[L]
    return None


# ------------------------ Load Player CSVs --------------------------
def discover_player_year_dirs(player_root: str, start: Optional[int], end: Optional[int]) -> List[Tuple[int, str]]:
    """
    Find subfolders like player_{year} under player_root and return [(year, path)].
    """
    dirs = []
    for path in glob.glob(os.path.join(player_root, "player_*")):
        base = os.path.basename(path)
        m = re.match(r"player_(\d{4})$", base)
        if not m:
            continue
        year = int(m.group(1))
        if start is not None and year < start:
            continue
        if end is not None and year > end:
            continue
        dirs.append((year, path))
    return sorted(dirs)


def load_all_player_csvs_for_year(year: int, year_dir: str) -> pd.DataFrame:
    """
    Read ALL CSVs inside player_{year}/ and return a concatenated DataFrame
    with normalized headers and ensured 'year' column.
    """
    csvs = glob.glob(os.path.join(year_dir, "*.csv"))
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f, encoding="latin-1")

        # ensure year column
        if "year" not in [snake(c) for c in df.columns]:
            df["year"] = year

        # normalize headers to snake and handle duplicates
        new_columns = []
        seen = set()
        for c in df.columns:
            snake_col = snake(c)
            if snake_col in seen:
                # Add suffix to make it unique
                counter = 1
                while f"{snake_col}_{counter}" in seen:
                    counter += 1
                snake_col = f"{snake_col}_{counter}"
            new_columns.append(snake_col)
            seen.add(snake_col)
        
        df.columns = new_columns

        # normalize known keys
        if "player" in df.columns:
            df["player"] = df["player"].map(normalize_text)
        if "school" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"school": "team"})
        if "team" in df.columns:
            df["team"] = df["team"].map(normalize_text)
        if "conf" in df.columns:
            df["conf"] = df["conf"].map(normalize_conf)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # union all columns
    all_cols = sorted(set(itertools.chain.from_iterable([f.columns for f in frames])))
    
    # Safety check: ensure no DataFrame has duplicate columns before reindexing
    safe_frames = []
    for f in frames:
        if f.columns.duplicated().any():
            # Remove duplicate columns by keeping the first occurrence
            f = f.loc[:, ~f.columns.duplicated()]
        safe_frames.append(f.reindex(columns=all_cols))
    
    return pd.concat(safe_frames, ignore_index=True)


def collapse_to_player_year(df_year: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse many rows per player-year (across many CSVs) -> ONE row per (player, year).
    - numeric columns: sum
    - text columns: first non-null
    - 'team': keep ALL unique teams joined by " | " to detect multi-team years
    - 'conf': first non-null (best effort) -> also mapped to 'conference'
    """
    if df_year.empty:
        return df_year

    cols = df_year.columns.tolist()
    has_conf = "conf" in cols

    # Build aggregations dynamically
    agg_map = {}
    for c in cols:
        if c in {"player", "year"}:
            continue
        if c == "team":
            agg_map[c] = lambda s: " | ".join(sorted({normalize_text(x)
                                                      for x in s.dropna().astype(str)
                                                      if x and x != "nan"})) if s.notna().any() else ""
        elif c == "conf":
            agg_map[c] = "first"
        else:
            if pd.api.types.is_numeric_dtype(df_year[c]):
                agg_map[c] = "sum"
            else:
                agg_map[c] = "first"

    grouped = (df_year
               .groupby(["player", "year"], as_index=False)
               .agg(agg_map))

    # Ensure conference present
    if "conference" not in grouped.columns:
        if has_conf:
            grouped["conference"] = grouped["conf"]
        else:
            grouped["conference"] = "OTHER"

    return grouped


# ------------------------ Load Team Totals -------------------------
def _read_csv_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def _normalize_team_totals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a team totals table (any shape) to snake headers and standard keys."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    
    # normalize headers to snake and handle duplicates
    new_columns = []
    seen = set()
    for c in df.columns:
        snake_col = snake(c)
        if snake_col in seen:
            # Add suffix to make it unique
            counter = 1
            while f"{snake_col}_{counter}" in seen:
                counter += 1
            snake_col = f"{snake_col}_{counter}"
        new_columns.append(snake_col)
        seen.add(snake_col)
    
    df.columns = new_columns
    
    # Handle season -> year mapping
    if "season" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"season": "year"})
    
    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})
    if "team" in df.columns:
        df["team"] = df["team"].map(normalize_text)
    if "conf" in df.columns:
        df["conf"] = df["conf"].map(normalize_conf)
    if "conference" in df.columns:
        df["conference"] = df["conference"].map(normalize_conf)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["year"]).astype({"year": "int"})
    return df

def _pick_pass_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    # passing yards / td candidate headers (passing yards = receiving yards for team totals)
    y = pick_one(df, ["pass_yds","yds_pass","yds (pass)","passing_yards","pass yds","yds"])
    t = pick_one(df, ["pass_td","td_pass","td (pass)","passing_td","pass tds","td","tds"])
    return y, t

def _team_totals_from_explicit_files(passing_csv: Optional[str], rushing_csv: Optional[str]) -> pd.DataFrame:
    """
    Read the two canonical CSVs (passing + rushing). We only NEED passing totals for DOM (as receiving totals).
    """
    frames = []
    if passing_csv and os.path.exists(passing_csv):
        dfp = _normalize_team_totals_df(_read_csv_any(passing_csv))
        if not dfp.empty:
            ycol, tcol = _pick_pass_columns(dfp)
            keep = ["year","team","conf","conference"]
            if ycol: keep.append(ycol)
            if tcol: keep.append(tcol)
            dfp = dfp[[c for c in keep if c in dfp.columns]].copy()
            dfp = dfp.rename(columns={
                (ycol or ""): "team_receiving_yards",
                (tcol or ""): "team_receiving_tds",
            })
            frames.append(dfp)

    if rushing_csv and os.path.exists(rushing_csv):
        dfr = _normalize_team_totals_df(_read_csv_any(rushing_csv))
        if not dfr.empty:
            # rushing columns are not used for DOM; we still keep conference/team/year for merge confidence
            dfr = dfr[[c for c in dfr.columns if c in {"year","team","conf","conference"}]].copy()
            frames.append(dfr)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # coalesce conference columns
    if "conference" not in out.columns and "conf" in out.columns:
        out["conference"] = out["conf"]
    if "conference" in out.columns:
        out["conference"] = out["conference"].map(normalize_conf)
    out = out.drop_duplicates(subset=["year","team"], keep="first")
    return out

def load_team_totals(
    team_globs: Optional[List[str]],
    teams_passing_path: Optional[str],
    teams_rushing_path: Optional[str]
) -> pd.DataFrame:
    """
    Priority:
      1) Read explicit passing/rushing CSVs first (preferred source).
      2) Read supplemental globs; fill ONLY missing team_year totals.
    Returns a table with ['year','team','conference','team_receiving_yards','team_receiving_tds'] where available.
    """
    # 1) explicit files
    base = _team_totals_from_explicit_files(teams_passing_path, teams_rushing_path)

    # 2) supplemental globs
    supp = pd.DataFrame()
    if team_globs:
        files = []
        for g in team_globs:
            files.extend(glob.glob(g))
        if files:
            frames = []
            for f in files:
                df = _normalize_team_totals_df(_read_csv_any(f))
                if df.empty:
                    continue
                # try to detect passing totals as proxy for receiving
                ycol = pick_one(df, [
                    "team_rec_yds","team_receiving_yards","receiving_yards","rec_yds",
                    "yds_(receiving)","pass_receiving_yards","pass_yds","passing_yards","yds"
                ])
                tcol = pick_one(df, [
                    "team_rec_td","team_receiving_td","receiving_td","rec_td","td_(receiving)",
                    "pass_receiving_td","pass_td","passing_td","td","tds","receiving_tds"
                ])
                keep = [c for c in ["year","team","conf","conference", ycol, tcol] if c and c in df.columns]
                if not keep:
                    continue
                dfx = df[keep].copy()
                dfx = dfx.rename(columns={
                    (ycol or ""): "team_receiving_yards",
                    (tcol or ""): "team_receiving_tds",
                })
                frames.append(dfx)
            if frames:
                supp = pd.concat(frames, ignore_index=True)
                if "conference" not in supp.columns and "conf" in supp.columns:
                    supp["conference"] = supp["conf"]
                if "conference" in supp.columns:
                    supp["conference"] = supp["conference"].map(normalize_conf)
                supp["team"] = supp["team"].map(normalize_text)
                supp = supp.drop_duplicates(subset=["year","team"], keep="first")

    # 3) merge with priority to explicit files
    if base is None or base.empty:
        teams = supp
    elif supp is None or supp.empty:
        teams = base
    else:
        # Left join base (priority) with supp (fill missing receiving totals only)
        teams = base.merge(
            supp[["year","team","team_receiving_yards","team_receiving_tds"]],
            on=["year","team"], how="left", suffixes=("","_supp")
        )
        for c in ["team_receiving_yards","team_receiving_tds"]:
            if c not in teams.columns and f"{c}_supp" in teams.columns:
                teams[c] = teams[f"{c}_supp"]
            else:
                teams[c] = teams[c].where(teams[c].notna(), teams.get(f"{c}_supp"))
        drop_cols = [c for c in ["team_receiving_yards_supp","team_receiving_tds_supp"] if c in teams.columns]
        if drop_cols:
            teams = teams.drop(columns=drop_cols)

    if teams is None or teams.empty:
        return pd.DataFrame()

    # Final normalize
    if "conference" in teams.columns:
        teams["conference"] = teams["conference"].map(normalize_conf)
    teams["team"] = teams["team"].map(normalize_text)
    teams["year"] = pd.to_numeric(teams["year"], errors="coerce").astype("Int64")
    teams = teams.dropna(subset=["year","team"]).astype({"year":"int"})
    
    # Select columns conditionally based on what's available
    select_cols = ["year", "team"]
    if "conference" in teams.columns:
        select_cols.append("conference")
    if "team_receiving_yards" in teams.columns:
        select_cols.append("team_receiving_yards")
    if "team_receiving_tds" in teams.columns:
        select_cols.append("team_receiving_tds")
    
    return teams[select_cols].drop_duplicates()


# --------------------------- DOM calc -----------------------------
def detect_rec_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Try to locate player receiving yards & TD columns among all Stathead headers (snake_case)."""
    y = pick_one(df, ["rec_yds", "receiving_yards", "yds", "rec yds"])
    t = pick_one(df, ["rec_td", "receiving_td", "tds", "td", "rec td"])
    return y, t


def compute_dom_with_skips(df: pd.DataFrame, team_totals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute DOM / DOM_plus where possible.
    Skips (DOM not computed):
      - missing team totals for (year, team)
      - multi-team year (team field includes ' | ')
      - missing receiving_yards/tds on the player row

    Returns:
      domdf  : df with DOM, DOM_plus, team_receiving_yards, team_receiving_tds columns added
      skip_df: log of skipped reasons (and 'zero_team_totals_used' cases for visibility)
    """
    if df.empty:
        return df.assign(DOM=pd.NA, DOM_plus=pd.NA,
                         team_receiving_yards=pd.NA, team_receiving_tds=pd.NA), pd.DataFrame()

    # Find player rec cols
    ycol, tcol = detect_rec_cols(df)
    skip_rows = []

    # Prepare merge key: single-team only
    domdf = df.copy()
    domdf["team"] = domdf.get("team", "").fillna("")
    domdf["single_team"] = ~domdf["team"].astype(str).str.contains(r"\|")

    # Identify if player has the needed rec cols
    if ycol and tcol:
        domdf["has_rec_cols"] = domdf[ycol].notna() | domdf[tcol].notna()
    else:
        domdf["has_rec_cols"] = False

    # Merge team totals on (year, team)
    if team_totals is not None and not team_totals.empty:
        # Select only columns that exist in team_totals
        merge_cols = ["year", "team", "team_receiving_yards", "team_receiving_tds"]
        if "conference" in team_totals.columns:
            merge_cols.append("conference")
        
        merged = domdf.merge(
            team_totals[merge_cols].drop_duplicates(),
            on=["year", "team"], how="left", suffixes=("", "_team")
        )
        
        # prefer player-level conference if present; else team conf (if team conf exists)
        if "conference_team" in merged.columns:
            merged["conference"] = merged.get("conference", "").where(
                merged.get("conference", "").ne("") & merged.get("conference", "").notna(),
                merged["conference_team"]
            )
            merged = merged.drop(columns=["conference_team"])
        domdf = merged
    else:
        domdf["team_receiving_yards"] = pd.NA
        domdf["team_receiving_tds"] = pd.NA

    # Build skip masks
    missing_team_totals = domdf["team_receiving_yards"].isna() | domdf["team_receiving_tds"].isna()
    zero_team_totals = (~missing_team_totals) & \
        (domdf["team_receiving_yards"].fillna(0) == 0) & (domdf["team_receiving_tds"].fillna(0) == 0)
    multi_team = ~domdf["single_team"]
    missing_player_rec = ~(domdf["has_rec_cols"])

    # Log everything (skips + zero totals used)
    for _, r in domdf[multi_team].iterrows():
        skip_rows.append({"player": r["player"], "year": int(r["year"]), "team": r.get("team", ""), "reason": "multi_team_year"})
    for _, r in domdf[missing_team_totals & ~multi_team].iterrows():
        skip_rows.append({"player": r["player"], "year": int(r["year"]), "team": r.get("team", ""), "reason": "missing_team_totals"})
    for _, r in domdf[missing_player_rec & ~multi_team & ~missing_team_totals].iterrows():
        skip_rows.append({"player": r["player"], "year": int(r["year"]), "team": r.get("team", ""), "reason": "missing_player_receiving_cols"})
    for _, r in domdf[zero_team_totals & ~multi_team].iterrows():
        # we KEEP these (DOM=0 by math), but still log for visibility
        skip_rows.append({"player": r["player"], "year": int(r["year"]), "team": r.get("team", ""), "reason": "zero_team_totals_used"})

    # Eligible for DOM: single team AND have team totals AND have rec cols
    eligible = domdf["single_team"] & (~missing_team_totals) & (domdf["has_rec_cols"])

    # If we can't find the columns at all, return with log
    if ycol is None or tcol is None:
        domdf["DOM"] = pd.NA
        domdf["DOM_plus"] = pd.NA
        return domdf, pd.DataFrame(skip_rows)

    # Compute DOM only for eligible rows
    domdf["DOM"] = pd.NA
    domdf["DOM_plus"] = pd.NA

    idx = domdf.index[eligible]
    # avoid divide-by-zero
    ymask = domdf.loc[idx, "team_receiving_yards"].fillna(0) != 0
    tmask = domdf.loc[idx, "team_receiving_tds"].fillna(0) != 0

    part_y = domdf.loc[idx, ycol] / domdf.loc[idx, "team_receiving_yards"].where(ymask, other=1.0)
    part_t = domdf.loc[idx, tcol] / domdf.loc[idx, "team_receiving_tds"].where(tmask, other=1.0)

    dom_vals = 0.5 * (part_y.where(ymask, 0.0).fillna(0.0) + part_t.where(tmask, 0.0).fillna(0.0))
    domdf.loc[idx, "DOM"] = dom_vals
    domdf.loc[idx, "DOM_plus"] = dom_vals * domdf.loc[idx, "conference"].map(conf_multiplier).fillna(conf_multiplier("OTHER"))

    return domdf, pd.DataFrame(skip_rows)


# ----------------------- Dict + Summaries --------------------------
def build_player_dict_all_cols(df: pd.DataFrame) -> Dict[str, Dict[int, Dict[str, object]]]:
    """
    Include ALL columns present in df for each player-year (plus DOM fields already present).
    Keys are snake_case to match normalized headers.
    """
    result: Dict[str, Dict[int, Dict[str, object]]] = {}

    if "year" in df.columns:
        df = df.copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["year"]).astype({"year": "int"})

    cols = [c for c in df.columns if c != "player"]

    for _, r in df.iterrows():
        player = r["player"]
        year = int(r["year"])
        payload = {}
        for c in cols:
            v = r[c]
            if isinstance(v, float) and math.isnan(v):
                v = None
            payload[c] = v
        result.setdefault(player, {})[year] = payload
    return result


def summarize_skips(skip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenient summary of skip reasons by year/team for Jupyter.
    """
    if skip_df is None or skip_df.empty:
        return pd.DataFrame({"note": ["No skipped rows"]})
    out = (skip_df
           .groupby(["year", "team", "reason"], as_index=False)
           .size()
           .rename(columns={"size": "rows"}))
    return out.sort_values(["year", "team", "reason"]).reset_index(drop=True)


# ------------------------- Orchestrator ----------------------------
def run_stathead_player_year_pipeline(
    player_root: str,
    team_globs: Optional[List[str]] = None,
    teams_passing_path: Optional[str] = None,
    teams_rushing_path: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    write_csv: bool = True
) -> Tuple[Dict[str, Dict[int, Dict[str, object]]], pd.DataFrame, pd.DataFrame]:
    """
    1) Read every CSV under CFB_Data/StatHead/player_{year}/ folders.
    2) Normalize headers, union ALL columns, collapse to one row per (player, year).
    3) Load team totals with priority:
         a) Explicit passing/rushing CSVs (preferred; read FIRST)
         b) Supplemental globs to fill any missing team_year totals
       Then compute DOM/DOM_plus where possible (skip + log otherwise).
    4) Return {player: {year: {...all cols..., DOM, DOM_plus}}}, player-year DF, and skip log DF.
    """
    # 1) Gather per-year player data
    year_dirs = discover_player_year_dirs(player_root, start, end)
    frames = []
    for year, ydir in year_dirs:
        dfy = load_all_player_csvs_for_year(year, ydir)
        if not dfy.empty:
            frames.append(dfy)

    if not frames:
        # Nothing found
        return {}, pd.DataFrame(), pd.DataFrame()

    players_all = pd.concat(frames, ignore_index=True)

    if "player" not in players_all.columns:
        raise ValueError("No 'player' column found in any player CSVs.")
    if "year" not in players_all.columns:
        raise ValueError("No 'year' column resolved for player rows.")

    # 2) Collapse to one row per (player, year)
    player_year = collapse_to_player_year(players_all)

    # Optional slice by year (post-collapse safety)
    if start is not None:
        player_year = player_year[player_year["year"] >= start]
    if end is not None:
        player_year = player_year[player_year["year"] <= end]

    # 3) Team totals + DOM (explicit files first, then globs)
    teams = load_team_totals(team_globs, teams_passing_path, teams_rushing_path)
    domdf, skip_df = compute_dom_with_skips(player_year, teams)

    # 4) Build nested dict with ALL columns
    player_dict = build_player_dict_all_cols(domdf)

    # Optional CSV outputs (CLI/CI)
    if write_csv and start is not None and end is not None:
        ensure_outdir()
        outp = os.path.join(OUT_DIR, f"dom_stathead_allcols_{start}_{end}.csv")
        domdf.to_csv(outp, index=False)
        print(f"✅ Wrote data: {outp}")

        if not skip_df.empty:
            skip_p = os.path.join(OUT_DIR, f"dom_stathead_allcols_skipped_{start}_{end}.csv")
            skip_df.sort_values(["year", "team", "player"]).to_csv(skip_p, index=False)
            print(f"⚠️  Wrote skip log: {skip_p}  (rows: {len(skip_df):,})")
        else:
            print("ℹ️  No skipped rows.")

    return player_dict, domdf, skip_df


# ------------------------------ CLI -------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-root", type=str, required=True,
                    help='Root directory containing "player_{year}" subfolders (e.g., Dynasty/data/CFB_Data/StatHead)')
    ap.add_argument("--teams", nargs="*", default=[],
                    help="Glob(s) for supplemental team totals CSVs (used AFTER explicit files to fill gaps)")
    ap.add_argument("--teams-passing", type=str, default=None,
                    help="Path to explicit team passing totals CSV (preferred; used FIRST)")
    ap.add_argument("--teams-rushing", type=str, default=None,
                    help="Path to explicit team rushing totals CSV (read FIRST; complements conference/year)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--no-save", action="store_true", help="Do not write CSV/logs")
    args = ap.parse_args()

    team_globs = args.teams if args.teams else None

    player_dict, df, skip_df = run_stathead_player_year_pipeline(
        player_root=args.player_root,
        team_globs=team_globs,
        teams_passing_path=args.teams_passing,
        teams_rushing_path=args.teams_rushing,
        start=args.start,
        end=args.end,
        write_csv=not args.no_save
    )

    print(f"Players in dict: {len(player_dict):,}")
    print(df.head(10))
    if not skip_df.empty:
        print("Skipped sample:")
        print(skip_df.head(10))
