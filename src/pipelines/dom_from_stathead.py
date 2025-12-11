#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dom_from_stathead.py  (StatHead-only version)

What this does
--------------
  - Derives seasons purely from StatHead exports
  - Fills Year1–Year5 chronologically based on StatHead seasons
  - Computes DOM / RDOM / PDOM for each season using StatHead team totals
  - Matches players through:
        1) Same-last-name block (first-name fuzzy)
        2) Same-last-name block (full-name fuzzy)
        3) Same-last-name block (INITIAL MATCH)
        4) Full-name fuzzy across entire table
  - Writes to the master CSV:
        Year1–Year5
        DOM / DOM+ / PDOM / PDOM+ / RDOM / RDOM+ with 1–5 suffixes
        Raw StatHead metrics with 1–5 suffixes:
            rec_yds, rec_tds, rush_yds, rush_tds, pass_yds, pass_tds
            team_pr_yds, team_pr_tds, team_ru_yds, team_ru_tds
        team, conf (base columns, taken from LATEST StatHead season)
  - Removes helper columns before saving:
        player_norm, fn_missing, ln_missing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# ====== PATH CONFIG ======
ROOT = Path("/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATHEAD_DIR   = ROOT / "data" / "CFB_Data" / "StatHead"

PLAYER_DIR_FMT = STATHEAD_DIR / "player_{year}"
TEAM_PASS_DIR  = STATHEAD_DIR / "team_total_passing"
TEAM_RUSH_DIR  = STATHEAD_DIR / "team_total_rushing"

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
    if pd.isna(conf):
        return 0.60
    key = str(conf).strip().upper()
    return CONF_MULT.get(key, 0.60)

# ====== CORE METRIC LISTS ======

# DOM inputs & raw stats
RAW_METRICS: List[str] = [
    "rec_yds", "rec_tds",
    "rush_yds", "rush_tds",
    "pass_yds", "pass_tds",
    "team_pr_yds", "team_pr_tds",
    "team_ru_yds", "team_ru_tds",
]

# DOM outputs
BASE_METRICS: List[str] = [
    "DOM", "DOM+",
    "PDOM", "PDOM+",
    "RDOM", "RDOM+",
]

# ====== NORMALIZATION HELPERS ======

def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_name_from_raw(name: str) -> Tuple[str, str]:
    if pd.isna(name):
        return ("", "")
    s = str(name).strip()

    # "Last, First" pattern
    if "," in s:
        last_raw, first_raw = s.split(",", 1)
        last_norm  = normalize_name(last_raw)
        first_norm = normalize_name(first_raw).split()[0] if normalize_name(first_raw) else ""
        return (first_norm, last_norm)

    # "First Last" pattern
    norm = normalize_name(s)
    parts = norm.split()
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], parts[-1])

def first_initial(s: str) -> str:
    return s[0] if s else ""

# ====== DISCOVER AVAILABLE YEARS (from StatHead) ======

def discover_stathead_years(stathead_dir: Path) -> List[int]:
    years = []
    for p in stathead_dir.glob("player_*"):
        if not p.is_dir():
            continue
        tail = p.name.split("_", 1)[-1]
        if tail.isdigit():
            years.append(int(tail))
    years = sorted(set(years))
    print(f"[INFO] Found StatHead player years: {years}")
    return years

# ====== UTILS ======

def find_first_col(columns: List[str], candidates: List[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

# ====== LOAD STATHEAD PLAYER DATA ======

def load_player_stats_for_year(year: int, fmt: Path) -> pd.DataFrame:
    """
    Load per-player StatHead stats for a given year.
    """
    year_dir = Path(str(fmt).format(year=year))
    if not year_dir.exists():
        print(f"[WARN] No player directory for year {year}: {year_dir}")
        return pd.DataFrame()

    rec_rows, rush_rows, pass_rows = [], [], []

    for csv_path in year_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)

        name = csv_path.name.lower()
        if name.endswith("_receiving.csv") or name.endswith("_rushing.csv"):
            # trim to first 10 cols for some StatHead exports
            df = df.iloc[:, :10]

        cols_lower = [c.lower() for c in df.columns]

        is_rec  = any("rec" in c for c in cols_lower)
        is_pass = any("cmp" in c for c in cols_lower)
        is_rush = (not is_pass) and any("att" in c for c in cols_lower)

        if is_rec:
            rec_rows.append(df)
        elif is_pass:
            pass_rows.append(df)
        elif is_rush:
            rush_rows.append(df)

    def norm_df(df: pd.DataFrame | None, stat: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        player_col = find_first_col(df.columns, ["Player", "Name", "player", "player_name"])
        team_col   = find_first_col(df.columns, ["School", "Team", "team"])
        conf_col   = find_first_col(df.columns, ["Conf", "Conference", "conf"])
        pos_col    = find_first_col(df.columns, ["Pos", "Position", "pos"])
        ycol       = find_first_col(df.columns, ["Season", "Year", "season"])

        yds_col = find_first_col(df.columns, ["Yds", "Yds.", "Yards"])
        td_col  = find_first_col(df.columns, ["TD", "Touchdowns", "TD."])

        out = pd.DataFrame()
        out["player"] = df[player_col] if player_col else df.index.astype(str)
        out["team"]   = df[team_col] if team_col else ""
        out["conf"]   = df[conf_col] if conf_col else ""
        out["pos"]    = df[pos_col] if pos_col else ""
        out["year"]   = df[ycol] if ycol else year

        if stat == "rec":
            if yds_col:
                out["rec_yds"] = df[yds_col]
            if td_col:
                out["rec_tds"] = df[td_col]
        elif stat == "rush":
            if yds_col:
                out["rush_yds"] = df[yds_col]
            if td_col:
                out["rush_tds"] = df[td_col]
        elif stat == "pass":
            if yds_col:
                out["pass_yds"] = df[yds_col]
            if td_col:
                out["pass_tds"] = df[td_col]

        return out

    rec_df  = pd.concat([norm_df(df, "rec")  for df in rec_rows],  ignore_index=True) if rec_rows else pd.DataFrame()
    rush_df = pd.concat([norm_df(df, "rush") for df in rush_rows], ignore_index=True) if rush_rows else pd.DataFrame()
    pass_df = pd.concat([norm_df(df, "pass") for df in pass_rows], ignore_index=True) if pass_rows else pd.DataFrame()

    dfs = [d for d in [rec_df, rush_df, pass_df] if not d.empty]
    if not dfs:
        return pd.DataFrame()

    out = dfs[0]
    for other in dfs[1:]:
        out = out.merge(other, on=["player", "team", "conf", "pos", "year"], how="outer")

    return out

def build_players_table(years: List[int], fmt: Path) -> pd.DataFrame:
    frames = []
    for y in years:
        print(f"[INFO] Loading StatHead data for {y}")
        df_y = load_player_stats_for_year(y, fmt)
        if not df_y.empty:
            frames.append(df_y)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ====== LOAD TEAM TOTALS ======

def load_team_passing_totals(dirpath: Path) -> pd.DataFrame:
    frames = []
    for p in dirpath.glob("*.csv"):
        df = pd.read_csv(p)
        year_col = find_first_col(df.columns, ["Season","Year","season"])
        team_col = find_first_col(df.columns, ["School","Team","team"])
        yds_col  = find_first_col(df.columns, ["Yds","Yards","Yds."])
        td_col   = find_first_col(df.columns, ["TD","Touchdowns","TD."])
        if not all([year_col, team_col, yds_col, td_col]):
            continue

        out = pd.DataFrame()
        out["year"] = df[year_col]
        out["team"] = df[team_col]
        out["team_pr_yds"] = df[yds_col]
        out["team_pr_tds"] = df[td_col]
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df["team"].astype(str)
    return df

def load_team_rushing_totals(dirpath: Path) -> pd.DataFrame:
    frames = []
    for p in dirpath.glob("*.csv"):
        df = pd.read_csv(p)
        year_col = find_first_col(df.columns, ["Season","Year","season"])
        team_col = find_first_col(df.columns, ["School","Team","team"])
        yds_col  = find_first_col(df.columns, ["Yds","Yards","Yds."])
        td_col   = find_first_col(df.columns, ["TD","Touchdowns","TD."])
        if not all([year_col, team_col, yds_col, td_col]):
            continue

        out = pd.DataFrame()
        out["year"] = df[year_col]
        out["team"] = df[team_col]
        out["team_ru_yds"] = df[yds_col]
        out["team_ru_tds"] = df[td_col]
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df["team"].astype(str)
    return df

# ====== DOM METRICS ======

def compute_dom_metrics(players: pd.DataFrame, team_pr: pd.DataFrame, team_ru: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DOM / PDOM / RDOM using StatHead player stats and StatHead team totals.
    """
    players = players.merge(team_pr, on=["year", "team"], how="left")
    players = players.merge(team_ru, on=["year", "team"], how="left")

    # Ensure team totals columns exist
    for c in ["team_pr_yds", "team_pr_tds", "team_ru_yds", "team_ru_tds"]:
        if c not in players.columns:
            players[c] = np.nan

    # Ensure player stat columns exist
    for c in ["rec_yds", "rec_tds", "rush_yds", "rush_tds", "pass_yds", "pass_tds"]:
        if c not in players.columns:
            players[c] = np.nan

    # Replace NaN with 0.0
    for c in ["team_pr_yds", "team_pr_tds", "team_ru_yds", "team_ru_tds",
              "rec_yds", "rec_tds", "rush_yds", "rush_tds", "pass_yds", "pass_tds"]:
        players[c] = players[c].fillna(0.0)

    # Receiving DOM
    dom_y = players["rec_yds"] / players["team_pr_yds"].replace(0, np.nan)
    dom_t = players["rec_tds"] / players["team_pr_tds"].replace(0, np.nan)
    players["DOM"] = 0.5 * (dom_y.fillna(0) + dom_t.fillna(0))

    # Passing DOM
    p_y = players["pass_yds"] / players["team_pr_yds"].replace(0, np.nan)
    p_t = players["pass_tds"] / players["team_pr_tds"].replace(0, np.nan)
    players["PDOM"] = 0.5 * (p_y.fillna(0) + p_t.fillna(0))

    # Rushing DOM
    r_y = players["rush_yds"] / players["team_ru_yds"].replace(0, np.nan)
    r_t = players["rush_tds"] / players["team_ru_tds"].replace(0, np.nan)
    players["RDOM"] = 0.5 * (r_y.fillna(0) + r_t.fillna(0))

    players["conf_mult"] = players["conf"].map(get_conf_multiplier)
    players["DOM+"]  = players["DOM"]  * players["conf_mult"]
    players["PDOM+"] = players["PDOM"] * players["conf_mult"]
    players["RDOM+"] = players["RDOM"] * players["conf_mult"]

    return players

# ====== PLAYER-LEVEL MATCHING ======

def select_best_stathead_name(
    master_norm: str,
    last_missing: str,
    players_df: pd.DataFrame,
    full_thresh: float = 0.80
) -> str | None:
    """
    Given a normalized master name and last name, pick the best matching
    StatHead player_norm from players_df, using:
      - full-name similarity within same-last-name block
      - first-initial matches within same-last-name block
      - full-name similarity across entire table as final fallback
    """
    if not master_norm:
        return None

    # 1) Same-last-name block
    block = players_df[players_df["ln_stat"] == last_missing]
    if not block.empty:
        block = block.copy()
        block["full_score"] = block["player_norm"].apply(
            lambda p: SequenceMatcher(None, master_norm, p).ratio()
        )
        best = block.loc[block["full_score"].idxmax()]
        if best["full_score"] >= full_thresh:
            return best["player_norm"]

        # 2) Same-last-name block (first-initial match)
        m_first = first_initial(master_norm.split()[0] if master_norm else "")
        for _, row in block.iterrows():
            s_first = first_initial(row["fn_stat"])
            if m_first == s_first and m_first != "":
                return row["player_norm"]

    # 3) Full-name fuzzy across entire table
    block2 = players_df.copy()
    block2["full_score"] = block2["player_norm"].apply(
        lambda p: SequenceMatcher(None, master_norm, p).ratio()
    )
    best2 = block2.loc[block2["full_score"].idxmax()]
    if best2["full_score"] >= full_thresh:
        return best2["player_norm"]

    return None

# ====== MAIN PIPELINE ======

def main(players_csv: Path, out_csv: Path):
    df = pd.read_csv(players_csv)

    player_col = find_first_col(df.columns, ["player","player_name","Player","PlayerName"])
    if not player_col:
        raise ValueError("Could not find player name column in master CSV.")

    # --- Discover available years from StatHead ---
    years = discover_stathead_years(STATHEAD_DIR)
    if not years:
        raise RuntimeError("No player_{year} folders under StatHead.")

    # --- Load StatHead player-level data ---
    players_all = build_players_table(years, PLAYER_DIR_FMT)
    if players_all.empty:
        raise RuntimeError("No StatHead player stats loaded.")

    # Normalization for later matching
    players_all["player_norm"] = players_all["player"].map(normalize_name)
    players_all["year_int"]    = players_all["year"].astype(int)

    # --- Load team totals ---
    team_pr = load_team_passing_totals(TEAM_PASS_DIR)
    team_ru = load_team_rushing_totals(TEAM_RUSH_DIR)

    # --- Compute DOM metrics with StatHead player stats ---
    players_dom = compute_dom_metrics(players_all, team_pr, team_ru)

    # --- Prepare StatHead players for matching ---
    players_dom["player_norm"] = players_dom["player"].map(normalize_name)
    players_dom["year_int"]    = players_dom["year"].astype(int)
    players_dom[["fn_stat","ln_stat"]] = players_dom["player"].apply(
        lambda x: pd.Series(split_name_from_raw(x))
    )

    # --- Prepare master players for matching ---
    df["player_norm"] = df[player_col].map(normalize_name)
    df[["fn_missing","ln_missing"]] = df[player_col].apply(
        lambda x: pd.Series(split_name_from_raw(x))
    )

    slots = [1, 2, 3, 4, 5]

    # Ensure base team/conf columns exist with object dtype
    if "team" not in df.columns:
        df["team"] = pd.Series(index=df.index, dtype="object")
    else:
        df["team"] = df["team"].astype("object")

    if "conf" not in df.columns:
        df["conf"] = pd.Series(index=df.index, dtype="object")
    else:
        df["conf"] = df["conf"].astype("object")

    # Ensure Year1-Year5 and metric columns exist
    for s in slots:
        if f"Year{s}" not in df.columns:
            df[f"Year{s}"] = np.nan

        for b in BASE_METRICS:
            col = f"{b}{s}"
            if col not in df.columns:
                df[col] = np.nan

        for m in RAW_METRICS:
            col = f"{m}{s}"
            if col not in df.columns:
                df[col] = np.nan

    unmatched: List[int] = []

    # --- Row-wise matching from master df -> StatHead (players_dom) ---
    for idx, row in df.iterrows():
        norm_name = row["player_norm"]
        ln   = row["ln_missing"]

        match_norm = select_best_stathead_name(norm_name, ln, players_dom)

        if match_norm is None:
            unmatched.append(idx)
            continue

        seasons = players_dom[players_dom["player_norm"] == match_norm].copy()
        seasons = seasons.sort_values("year_int")
        if seasons.empty:
            unmatched.append(idx)
            continue

        # LATEST StatHead season for team/conf
        latest = seasons.iloc[-1]
        df.at[idx, "team"] = latest.get("team", np.nan)
        df.at[idx, "conf"] = latest.get("conf", np.nan)

        # Year1..Year5 earliest → latest
        for slot, (_, srow) in enumerate(seasons.head(5).iterrows(), start=1):
            df.at[idx, f"Year{slot}"] = int(srow["year_int"])

            # Base DOM metrics
            for b in BASE_METRICS:
                df.at[idx, f"{b}{slot}"] = srow.get(b, np.nan)

            # Raw metrics (DOM inputs & team totals)
            for m in RAW_METRICS:
                df.at[idx, f"{m}{slot}"] = srow.get(m, np.nan)

    # Drop helper columns from master
    df = df.drop(columns=["player_norm", "fn_missing", "ln_missing"], errors="ignore")

    # --- Unmatched logging ---
    print(f"[INFO] Unmatched players: {len(unmatched)}")
    if unmatched:
        ufile = out_csv.with_name(out_csv.stem + "_UNMATCHED.csv")
        df_unmatched = pd.read_csv(players_csv).iloc[unmatched]
        df_unmatched[[player_col]].to_csv(ufile, index=False)
        print(f"[INFO] Unmatched names written to {ufile}")

    # --- Reorder: team, conf right after position column ---
    cols = list(df.columns)
    pos_col = find_first_col(cols, ["pos", "Pos", "position", "Position"])
    if pos_col and "team" in cols and "conf" in cols:
        cols.remove("team")
        cols.remove("conf")
        insert_idx = cols.index(pos_col) + 1
        cols.insert(insert_idx, "team")
        cols.insert(insert_idx + 1, "conf")
        df = df[cols]

    print(f"[INFO] Saving to {out_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild Year1–Year5 and DOM from StatHead only (no PFF)."
    )
    parser.add_argument("--players_csv", type=Path, required=True)
    parser.add_argument("--out_csv",      type=Path, required=True)
    args = parser.parse_args()
    main(args.players_csv, args.out_csv)
