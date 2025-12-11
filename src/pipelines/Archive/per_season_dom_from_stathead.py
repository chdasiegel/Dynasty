#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
per_season_dom_from_stathead.py

Per-season version using dom_master_{year}.csv as player data:

  - Uses dom_master_{year}.csv files as the player-level stats source
  - DOES NOT recompute DOM/PDOM/RDOM; uses the values present in dom_master.
  - Computes DOM+/PDOM+/RDOM+ using conference multipliers if base metrics exist.
  - Matches players in the master CSV to dom_master players through:
        1) Same-last-name block (full-name fuzzy)
        2) Same-last-name block (INITIAL MATCH)
        3) Full-name fuzzy across entire table
  - Outputs ONE ROW PER (player, season) with columns:

        player,team,year,age,conference,
        receiving_yards,receiving_tds,
        rushing_yards,rushing_tds,
        passing_yards,passing_tds,
        DOM,DOM+,RDOM,RDOM+,PDOM,PDOM+,elusive_rating

    where age is computed as of September 1st of that season year,
    using the player's birthday from the input CSV (if available).

  - ALL players from the master CSV appear in the output:
        • Matched players → one row per season
        • Completely unmatched players → a single row with NaNs for year/stats

  - IMPORTANT: This script writes exactly ONE CSV: out_csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# ====== PATH CONFIG ======

ROOT = Path("/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty")

# Where dom_master_{year}.csv live
DOM_DIR = ROOT / "data" / "processed"

# ====== CONF MULTIPLIERS ======

CONF_MULT: Dict[str, float] = {
    "SEC": 1.0,
    "BIG TEN": 1.0, "BIG 10": 1.0, "B1G": 1.0,
    "BIG 12": 0.95,
    "ACC": 0.95,
    "PAC-12": 0.95, "PAC 12": 0.95,
    "AAC": 0.85, "AMERICAN": 0.85,
    "MOUNTAIN WEST": 0.76, "MWC": 0.76,
    "SUN BELT": 0.76,
    "MAC": 0.70,
    "C-USA": 0.70, "CUSA": 0.70, "CONFERENCE USA": 0.70,
    "INDEPENDENT": 0.60, "IND": 0.60,
    "OTHER": 0.60, "FCS": 0.60, "NON-FBS": 0.60,
}

def get_conf_multiplier(conf: str) -> float:
    """
    Map raw conference string to CONF_MULT (case-insensitive-ish).
    Defaults to 0.60 if not found.
    """
    if pd.isna(conf):
        return 0.60
    key = str(conf).strip().upper()
    return CONF_MULT.get(key, 0.60)

# ====== NORMALIZATION HELPERS ======

def normalize_name(name: str) -> str:
    """Lowercase, strip, keep only alphanumeric + space."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_name_from_raw(name: str) -> Tuple[str, str]:
    """Extract (first, last) from raw names: First Last OR Last, First."""
    if pd.isna(name):
        return ("", "")
    s = str(name).strip()

    # Handle "Last, First"
    if "," in s:
        last_raw, first_raw = s.split(",", 1)
        last_norm  = normalize_name(last_raw)
        first_norm = normalize_name(first_raw).split()[0] if normalize_name(first_raw) else ""
        return (first_norm, last_norm)

    # "First Middle Last"
    norm = normalize_name(s)
    parts = norm.split()
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], parts[-1])

def first_initial(s: str) -> str:
    return s[0] if s else ""

# ====== GENERIC COLUMN FINDER ======

def find_first_col(columns: List[str], candidates: List[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

# ====== DISCOVER dom_master YEARS ======

def discover_dom_years(dom_dir: Path) -> List[int]:
    """
    Look for files like dom_master_2005.csv and return [2005, ...].
    """
    years: List[int] = []
    for p in dom_dir.glob("dom_master_*.csv"):
        stem = p.stem  # e.g. "dom_master_2005"
        tail = stem.split("_")[-1]
        if tail.isdigit():
            years.append(int(tail))
    years = sorted(set(years))
    print(f"[INFO] Found dom_master years: {years}")
    return years

# ====== LOAD dom_master PLAYER DATA ======

def load_dom_for_year(year: int, dom_dir: Path) -> pd.DataFrame:
    """
    Load dom_master_{year}.csv and normalize key columns.
    """
    path = dom_dir / f"dom_master_{year}.csv"
    if not path.exists():
        print(f"[WARN] No dom_master file for year {year}: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    cols = df.columns.tolist()

    player_col = find_first_col(cols, ["player", "Player", "player_name", "PlayerName"])
    team_col   = find_first_col(cols, ["team", "Team", "School"])
    conf_col   = find_first_col(cols, ["conf", "Conf", "conference", "Conference"])
    year_col   = find_first_col(cols, ["year", "Year", "season", "Season"])

    rec_yds_col  = find_first_col(cols, ["rec_yds", "Rec_Yds", "receiving_yards", "RecYds"])
    rec_tds_col  = find_first_col(cols, ["rec_tds", "Rec_Tds", "receiving_tds", "RecTD"])
    rush_yds_col = find_first_col(cols, ["rush_yds", "Rush_Yds", "rushing_yards", "RushYds"])
    rush_tds_col = find_first_col(cols, ["rush_tds", "Rush_Tds", "rushing_tds", "RushTD"])
    pass_yds_col = find_first_col(cols, ["pass_yds", "Pass_Yds", "passing_yards", "PassYds"])
    pass_tds_col = find_first_col(cols, ["pass_tds", "Pass_Tds", "passing_tds", "PassTD"])

    dom_col    = find_first_col(cols, ["DOM", "Dom"])
    domp_col   = find_first_col(cols, ["DOM+", "Dom+"])
    rdom_col   = find_first_col(cols, ["RDOM", "RDom"])
    rdomp_col  = find_first_col(cols, ["RDOM+", "RDom+"])
    pdom_col   = find_first_col(cols, ["PDOM", "PDom"])
    pdomp_col  = find_first_col(cols, ["PDOM+", "PDom+"])

    elusive_col = find_first_col(cols, ["elusive_rating", "elusive", "ElusiveRating"])

    out = pd.DataFrame()

    out["player"] = df[player_col] if player_col else df.index.astype(str)
    out["team"]   = df[team_col] if team_col else ""
    out["conf"]   = df[conf_col] if conf_col else ""
    out["year"]   = df[year_col] if year_col else year

    out["rec_yds"]  = df[rec_yds_col]  if rec_yds_col  else 0.0
    out["rec_tds"]  = df[rec_tds_col]  if rec_tds_col  else 0.0
    out["rush_yds"] = df[rush_yds_col] if rush_yds_col else 0.0
    out["rush_tds"] = df[rush_tds_col] if rush_tds_col else 0.0
    out["pass_yds"] = df[pass_yds_col] if pass_yds_col else 0.0
    out["pass_tds"] = df[pass_tds_col] if pass_tds_col else 0.0

    # Existing DOM metrics if present
    if dom_col:
        out["DOM"] = df[dom_col]
    if domp_col:
        out["DOM+"] = df[domp_col]
    if rdom_col:
        out["RDOM"] = df[rdom_col]
    if rdomp_col:
        out["RDOM+"] = df[rdomp_col]
    if pdom_col:
        out["PDOM"] = df[pdom_col]
    if pdomp_col:
        out["PDOM+"] = df[pdomp_col]

    if elusive_col:
        out["elusive_rating"] = df[elusive_col]

    # Fill missing numeric stuff
    for c in ["rec_yds","rec_tds","rush_yds","rush_tds","pass_yds","pass_tds"]:
        out[c] = out[c].fillna(0.0)

    return out


def build_players_table(years: List[int], dom_dir: Path) -> pd.DataFrame:
    frames = []
    for y in years:
        print(f"[INFO] Loading dom_master data for {y}")
        df_y = load_dom_for_year(y, dom_dir)
        if not df_y.empty:
            frames.append(df_y)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out

# ====== PLAYER-LEVEL MATCHING ======

def select_best_stathead_name(master_norm: str,
                              last_missing: str,
                              players_df: pd.DataFrame,
                              full_thresh: float = 0.80) -> str | None:
    """
    Enhanced matching:
       1) Same last name block → full-name similarity
       2) Same last name block → first-name initials
       3) Full table → full-name similarity
    """
    if not master_norm:
        return None

    # PRIMARY: same last name block
    block = players_df[players_df["ln_stat"] == last_missing]
    if not block.empty:
        block = block.copy()
        block["full_score"] = block["player_norm"].apply(
            lambda p: SequenceMatcher(None, master_norm, p).ratio()
        )
        best = block.loc[block["full_score"].idxmax()]
        if best["full_score"] >= full_thresh:
            return best["player_norm"]

        # INITIAL MATCH LOGIC
        m_first = first_initial(master_norm.split()[0] if master_norm else "")
        for _, row in block.iterrows():
            s_first = first_initial(row["fn_stat"])
            if m_first == s_first and m_first != "":
                return row["player_norm"]

    # FALLBACK: full-name fuzzy over full table
    block2 = players_df.copy()
    block2["full_score"] = block2["player_norm"].apply(
        lambda p: SequenceMatcher(None, master_norm, p).ratio()
    )
    best2 = block2.loc[block2["full_score"].idxmax()]
    if best2["full_score"] >= full_thresh:
        return best2["player_norm"]

    return None

# ====== AGE HELPER ======

def compute_age_on_sept1(birth_ts, year: int):
    """
    Compute integer age on September 1 of a given year.
    birth_ts: pandas.Timestamp or NaT
    """
    if pd.isna(birth_ts):
        return np.nan
    try:
        sept1 = datetime(year=year, month=9, day=1)
    except Exception:
        return np.nan
    age_years = (sept1 - birth_ts.to_pydatetime()).days / 365.25
    return int(np.floor(age_years))

# ====== MAIN PIPELINE ======

def main(players_csv: Path, out_csv: Path):
    df = pd.read_csv(players_csv)

    # Player name column in master list
    player_col = find_first_col(df.columns.tolist(), ["player","player_name","Player","PlayerName"])
    if not player_col:
        raise ValueError("Could not find player name column in players_csv.")

    # Birthday column (for age)
    birthday_col = find_first_col(
        df.columns.tolist(),
        ["birthday", "Birthday", "BIRTHDAY", "dob", "DOB", "Birthdate", "Birth_date"]
    )
    if birthday_col:
        df["birthday_parsed"] = pd.to_datetime(df[birthday_col], errors="coerce")
        print(f"[INFO] Using birthday column: {birthday_col}")
    else:
        df["birthday_parsed"] = pd.NaT
        print("[WARN] No birthday column found; ages will be NaN.")

    # Discover dom_master years
    years = discover_dom_years(DOM_DIR)
    if not years:
        raise RuntimeError("No dom_master_{year}.csv files found in DOM_DIR.")

    # Load dom_master data
    players_all = build_players_table(years, DOM_DIR)
    if players_all.empty:
        raise RuntimeError("No player stats loaded from dom_master files.")

    # Apply conference multipliers to create DOM+, PDOM+, RDOM+ (without recomputing DOM/PDOM/RDOM)
    players_dom = players_all.copy()
    players_dom["conf_mult"] = players_dom["conf"].map(get_conf_multiplier)

    for base, plus in [("DOM", "DOM+"), ("PDOM", "PDOM+"), ("RDOM", "RDOM+")]:
        if base in players_dom.columns:
            players_dom[plus] = players_dom[base] * players_dom["conf_mult"]

    # Normalization fields for dom_master side
    players_dom["player_norm"] = players_dom["player"].map(normalize_name)
    players_dom["year_int"]    = players_dom["year"].astype(int)
    players_dom[["fn_stat","ln_stat"]] = players_dom["player"].apply(
        lambda x: pd.Series(split_name_from_raw(x))
    )

    # Normalize main names
    df["player_norm"] = df[player_col].map(normalize_name)
    df[["fn_missing","ln_missing"]] = df[player_col].apply(
        lambda x: pd.Series(split_name_from_raw(x))
    )

    per_season_rows = []
    unmatched_indices = []

    has_elusive = "elusive_rating" in players_dom.columns

    # Per-player matching and row expansion
    for idx, row in df.iterrows():
        norm = row["player_norm"]
        ln   = row["ln_missing"]

        match_norm = select_best_stathead_name(norm, ln, players_dom)

        if match_norm is None:
            unmatched_indices.append(idx)
            continue

        seasons = players_dom[players_dom["player_norm"] == match_norm].copy()
        seasons = seasons.sort_values("year_int")
        if seasons.empty:
            unmatched_indices.append(idx)
            continue

        # For each season, build one row in the per-season output
        birth_ts = row["birthday_parsed"]

        for _, s in seasons.iterrows():
            year_int = int(s["year_int"])
            age = compute_age_on_sept1(birth_ts, year_int)

            per_season_rows.append({
                "player":           row[player_col],
                "team":             s.get("team", ""),
                "year":             year_int,
                "age":              age,
                "conference":       s.get("conf", ""),
                "receiving_yards":  float(s.get("rec_yds", 0.0)),
                "receiving_tds":    float(s.get("rec_tds", 0.0)),
                "rushing_yards":    float(s.get("rush_yds", 0.0)),
                "rushing_tds":      float(s.get("rush_tds", 0.0)),
                "passing_yards":    float(s.get("pass_yds", 0.0)),
                "passing_tds":      float(s.get("pass_tds", 0.0)),
                "DOM":              float(s.get("DOM", np.nan)),
                "DOM+":             float(s.get("DOM+", np.nan)),
                "RDOM":             float(s.get("RDOM", np.nan)),
                "RDOM+":            float(s.get("RDOM+", np.nan)),
                "PDOM":             float(s.get("PDOM", np.nan)),
                "PDOM+":            float(s.get("PDOM+", np.nan)),
                "elusive_rating":   float(s.get("elusive_rating", np.nan)) if has_elusive else np.nan,
            })

    # Add one "empty" row for each completely unmatched player
    for idx in unmatched_indices:
        row = df.loc[idx]
        per_season_rows.append({
            "player":           row[player_col],
            "team":             "",
            "year":             np.nan,
            "age":              np.nan,
            "conference":       "",
            "receiving_yards":  np.nan,
            "receiving_tds":    np.nan,
            "rushing_yards":    np.nan,
            "rushing_tds":      np.nan,
            "passing_yards":    np.nan,
            "passing_tds":      np.nan,
            "DOM":              np.nan,
            "DOM+":             np.nan,
            "RDOM":             np.nan,
            "RDOM+":            np.nan,
            "PDOM":             np.nan,
            "PDOM+":            np.nan,
            "elusive_rating":   np.nan,
        })

    # Build final DataFrame
    out_df = pd.DataFrame(per_season_rows, columns=[
        "player","team","year","age","conference",
        "receiving_yards","receiving_tds",
        "rushing_yards","rushing_tds",
        "passing_yards","passing_tds",
        "DOM","DOM+","RDOM","RDOM+","PDOM","PDOM+",
        "elusive_rating"
    ])

    print(f"[INFO] Final per-season rows (including unmatched): {len(out_df)}")
    print(f"[INFO] Players with no dom_master match: {len(unmatched_indices)}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Per-season DOM table written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-season DOM table from dom_master_{year} and master player list.")
    parser.add_argument("--players_csv", type=Path, required=True)
    parser.add_argument("--out_csv",      type=Path, required=True)
    args = parser.parse_args()
    main(args.players_csv, args.out_csv)
