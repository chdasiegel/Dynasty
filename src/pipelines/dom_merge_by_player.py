#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dom_merge_by_player.py — Merge player-season rows by Player key and compute:
  DOM/DOM+ (receiving share),
  RDOM/RDOM+ (rushing share),
  PDOM/PDOM+ (passing share).

This version builds file paths from YEAR arguments following your layout:

Players (per year)
  {BASE}/player_{Y}/{Y}_receiving.csv
  {BASE}/player_{Y}/{Y}_rushing.csv
  {BASE}/player_{Y}/{Y}_passing.csv

Team totals (per year)
  {BASE}/team_total_passing/team_passing_{Y}.csv   -> baseline for DOM & PDOM
  {BASE}/team_total_rushing/team_rushing_{Y}.csv   -> baseline for RDOM

Outputs (per-year)
  Dynasty/data/processed/dom_master_{Y}.csv
"""

from __future__ import annotations
import os
import re
import sys
import argparse
import pandas as pd
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ------------------------------- Config --------------------------------
# Get the absolute path to the project root (Dynasty folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# Default base directory (override with --base-dir)
DEFAULT_BASE_DIR = "/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty/data/CFB_Data/StatHead"

# PFF elusive rating directory
PFF_DIR = PROJECT_ROOT / "data" / "CFB_Data" / "PFF"

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
    "OTHER": 0.60, "FCS": 0.60, "NON-FBS": 0.60
}

# ------------------------------ Utilities ------------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip()

def normalize_conf(conf: str) -> str:
    c = normalize_text(conf).upper().replace("-", " ").strip()
    
    # Handle specific mappings
    c = c.replace("PAC 10", "PAC 12").replace("BIG TEN CONFERENCE", "BIG TEN")
    c = c.replace("ATLANTIC COAST CONFERENCE", "ACC").replace("SOUTHEASTERN CONFERENCE", "SEC")
    c = c.replace("CONFERENCE USA", "C-USA")
    
    # Handle common variations
    if c == "AMERICAN" or "AMERICAN ATHLETIC" in c:
        return "AMERICAN"
    if c == "SUN BELT":
        return "SUN BELT"
    if c == "CUSA":
        return "CUSA"
    if c == "IND":
        return "INDEPENDENT"
    
    # Check exact matches first
    for key in CONF_MULT.keys():
        if c == key:
            return key
    
    # Then check partial matches
    for key in CONF_MULT.keys():
        if key in c:
            return key
    
    return "OTHER"

def conf_multiplier(conf: str) -> float:
    return CONF_MULT.get(normalize_conf(conf), CONF_MULT["OTHER"])

def normalize_player_name(name: str) -> str:
    """Normalize player names for matching between DOM and PFF data."""
    if pd.isna(name) or not name:
        return ""
    
    # Remove asterisks that appear in DOM data
    cleaned = name.replace("*", "").strip()
    
    # Handle common suffixes and variations
    # Remove common suffixes like Jr., Sr., III, etc.
    suffixes = [" Jr.", " Sr.", " III", " II", " IV", " V"]
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
            break
    
    return cleaned

def normalize_pff_team_name(pff_team: str) -> str:
    """Map PFF team names (abbreviated, ALL CAPS) to DOM team names (full names, proper case)"""
    pff_team = pff_team.upper().strip()
    
    # Common PFF abbreviations to full names
    team_mappings = {
        'ARIZONA ST': 'Arizona State',
        'APP STATE': 'Appalachian State', 
        'ARK STATE': 'Arkansas State',
        'BOISE ST': 'Boise State',
        'BOSTON COL': 'Boston College',
        'COLORADO ST': 'Colorado State',
        'EAST CAR': 'East Carolina',
        'FLORIDA ST': 'Florida State',
        'FRESNO ST': 'Fresno State',
        'GEORGIA ST': 'Georgia State',
        'IOWA ST': 'Iowa State',
        'KANSAS ST': 'Kansas State',
        'KENT ST': 'Kent State',
        'MICHIGAN ST': 'Michigan State',
        'MISS STATE': 'Mississippi State',
        'MISSISSIPPI ST': 'Mississippi State',
        'NC STATE': 'North Carolina State',
        'NEW MEXICO ST': 'New Mexico State',
        'OHIO ST': 'Ohio State',
        'OKLAHOMA ST': 'Oklahoma State',
        'OREGON ST': 'Oregon State',
        'PENN ST': 'Penn State',
        'SAN DIEGO ST': 'San Diego State',
        'SAN JOSE ST': 'San Jose State',
        'TEXAS ST': 'Texas State',
        'UTAH ST': 'Utah State',
        'WASHINGTON ST': 'Washington State',
        'WEST VIRGINIA': 'West Virginia',
        'WESTERN KY': 'Western Kentucky',
        'WESTERN MICH': 'Western Michigan',
    }
    
    if pff_team in team_mappings:
        return team_mappings[pff_team]
    
    # For non-abbreviated names, just convert to proper case
    return pff_team.title()

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", "")
        if s.lower() in {"", "na", "n/a", "nan", "none", "--"}:
            return None
        return float(s)
    except Exception:
        return None

def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = [normalize_text(c).lower().replace(" ", "_") for c in df.columns]
    return df

def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

# --------------------------- Path Builders ------------------------------
def year_paths(base: str, y: int) -> Tuple[str, str, str, str, str]:
    """
    Returns tuple of (rec, rush, pass, team_pr, team_ru) absolute file paths for a year.
    team_pr (pass/rec baseline) = team_total_passing/team_passing_{y}.csv
    team_ru (rush baseline)     = team_total_rushing/team_rushing_{y}.csv
    """
    player_dir = os.path.join(base, f"player_{y}")
    rec  = os.path.join(player_dir, f"{y}_receiving.csv")
    rush = os.path.join(player_dir, f"{y}_rushing.csv")
    pas  = os.path.join(player_dir, f"{y}_passing.csv")
    team_pr = os.path.join(base, "team_total_passing", f"team_passing_{y}.csv")
    team_ru = os.path.join(base, "team_total_rushing", f"team_rushing_{y}.csv")
    return rec, rush, pas, team_pr, team_ru

def _check_exists(path: str, strict: bool, label: str) -> bool:
    if os.path.exists(path):
        return True
    msg = f"⚠️ Missing {label}: {path}"
    if strict:
        raise FileNotFoundError(msg)
    print(msg, file=sys.stderr)
    return False

# --------------------------- Load: Team Totals --------------------------
def load_team_passrecv_totals(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        df = _read_csv_safe(p)
        # year from filename if not present
        y_match = re.search(r"(19|20)\d{2}", os.path.basename(p))
        y = int(y_match.group(0)) if y_match else None
        if "year" not in df.columns:
            df["year"] = y
        out = pd.DataFrame({
            "year": df["year"],
            "team": df["team"].map(normalize_text) if "team" in df.columns else df.get("team", ""),
            "team_passrecv_yards": df[_pick(df, "yards", "yds", "pass_yards", "rec_yards")].map(safe_float),
            "team_passrecv_tds": df[_pick(df, "tds", "td", "pass_tds", "rec_tds")].map(safe_float),
        })
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["year","team","team_passrecv_yards","team_passrecv_tds"])
    g = (pd.concat(frames, ignore_index=True)
           .assign(team_passrecv_yards=lambda d: d["team_passrecv_yards"].fillna(0.0),
                   team_passrecv_tds=lambda d: d["team_passrecv_tds"].fillna(0.0)))
    return g.groupby(["year","team"], as_index=False).sum(numeric_only=True)

def load_team_rush_totals(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        df = _read_csv_safe(p)
        y_match = re.search(r"(19|20)\d{2}", os.path.basename(p))
        y = int(y_match.group(0)) if y_match else None
        if "year" not in df.columns:
            df["year"] = y
        out = pd.DataFrame({
            "year": df["year"],
            "team": df["team"].map(normalize_text) if "team" in df.columns else df.get("team", ""),
            "team_rush_yards": df[_pick(df, "yards", "yds", "rush_yards", "rushing_yards")].map(safe_float),
            "team_rush_tds": df[_pick(df, "tds", "td", "rush_tds", "rushing_tds")].map(safe_float),
        })
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["year","team","team_rush_yards","team_rush_tds"])
    g = (pd.concat(frames, ignore_index=True)
           .assign(team_rush_yards=lambda d: d["team_rush_yards"].fillna(0.0),
                   team_rush_tds=lambda d: d["team_rush_tds"].fillna(0.0)))
    return g.groupby(["year","team"], as_index=False).sum(numeric_only=True)

# --------------------------- Load: Player Stats -------------------------
def _common_player_cols(df: pd.DataFrame, fallback_year: Optional[int]) -> pd.DataFrame:
    # Handle columns properly - create Series with correct length if they don't exist
    n_rows = len(df)
    
    # Look for Player (capital P) or player (lowercase)
    if "Player" in df.columns:
        player_col = df["Player"].map(normalize_text)
    elif "player" in df.columns:
        player_col = df["player"].map(normalize_text)
    else:
        player_col = pd.Series([""] * n_rows)
        
    # Look for Team (capital T) or team (lowercase)
    if "Team" in df.columns:
        team_col = df["Team"].map(normalize_text) 
    elif "team" in df.columns:
        team_col = df["team"].map(normalize_text) 
    else:
        team_col = pd.Series([""] * n_rows)
    
    # Look for conference column (check both "Conf", "conf", and "conference")
    if "Conf" in df.columns:
        conf_col = df["Conf"].map(normalize_conf)
    elif "conf" in df.columns:
        conf_col = df["conf"].map(normalize_conf)
    elif "conference" in df.columns:
        conf_col = df["conference"].map(normalize_conf)
    else:
        conf_col = pd.Series(["OTHER"] * n_rows)
    
    year_col = "year" if "year" in df.columns else None
    year_val = df[year_col] if year_col else fallback_year
    
    # Return with player as first column, then team, year, conference
    out = pd.DataFrame({
        "player": player_col,
        "team": team_col,
        "year": year_val,
        "conference": conf_col,
    })
    return out

def _load_player_kind(paths: List[str], kind: str, y_cols: Tuple[str, ...], td_cols: Tuple[str, ...]) -> pd.DataFrame:
    """Generic loader for receiving/rushing/passing."""
    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        df = _read_csv_safe(p)
        y_match = re.search(r"(19|20)\d{2}", os.path.basename(p))
        y = int(y_match.group(0)) if y_match else None
        base = _common_player_cols(df, y)
        n_rows = len(df)
        ycol = _pick(df, *y_cols)
        tdcol = _pick(df, *td_cols)
        frames.append(pd.concat([
            base,
            pd.DataFrame({
                f"{kind}_yards": df[ycol].map(safe_float) if ycol else pd.Series([0.0] * n_rows),
                f"{kind}_tds":   df[tdcol].map(safe_float) if tdcol else pd.Series([0.0] * n_rows),
            })
        ], axis=1))
    if not frames:
        return pd.DataFrame(columns=["year","team","player","conference",f"{kind}_yards",f"{kind}_tds"])
    g = pd.concat(frames, ignore_index=True)
    # Group by key columns and aggregate, preserving the conference column
    numeric_cols = [col for col in g.columns if col not in ["year","team","player","conference"]]
    grouped = g.groupby(["year","team","player","conference"], as_index=False)[numeric_cols].sum()
    return grouped

def load_player_receiving(paths: List[str]) -> pd.DataFrame:
    return _load_player_kind(paths, "receiving",
                             ("receiving_yards","rec_yds","rec_yards","yds_receiving","yds"),
                             ("receiving_tds","rec_tds","tds_receiving","td"))

def load_player_rushing(paths: List[str]) -> pd.DataFrame:
    return _load_player_kind(paths, "rushing",
                             ("rushing_yards","rush_yds","yds_rushing","yds"),
                             ("rushing_tds","rush_tds","tds_rushing","td"))

def load_player_passing(paths: List[str]) -> pd.DataFrame:
    return _load_player_kind(paths, "passing",
                             ("passing_yards","pass_yds","yds_passing","yds"),
                             ("passing_tds","pass_tds","tds_rushing","td"))

# --------------------------- Load: PFF Elusive Rating --------------------
def load_pff_elusive_rating(year: int) -> pd.DataFrame:
    """Load PFF elusive rating data for a specific year."""
    pff_file = PFF_DIR / f"rushing_summary_{year}.csv"
    
    if not pff_file.exists():
        print(f"⚠️ No PFF data found for {year}")
        return pd.DataFrame(columns=["player", "team", "year", "elusive_rating"])
    
    try:
        df = pd.read_csv(pff_file)
        
        # Clean and normalize the data
        result = pd.DataFrame({
            "player": df["player"].map(normalize_text).map(normalize_player_name),
            "team": df["team_name"].map(normalize_pff_team_name), 
            "year": year,
            "elusive_rating": df["elusive_rating"].fillna(0.0)
        })
        
        # Filter out rows with empty player names
        result = result[result['player'].str.strip().str.len() > 0]
        
        print(f"📊 Loaded {len(result)} PFF elusive ratings for {year}")
        return result
        
    except Exception as e:
        print(f"❌ Error loading PFF data for {year}: {e}")
        return pd.DataFrame(columns=["player", "team", "year", "elusive_rating"])

# ------------------------------ Merge & Metrics -------------------------
def combine_players_by_key(recv: pd.DataFrame, rush: pd.DataFrame, pas: pd.DataFrame) -> pd.DataFrame:
    cols = ["year","team","player","conference"]
    frames = [f for f in (recv, rush, pas) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=cols + [
            "receiving_yards","receiving_tds",
            "rushing_yards","rushing_tds",
            "passing_yards","passing_tds",
        ])
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=cols, how="outer")
    for c in ["receiving_yards","receiving_tds","rushing_yards","rushing_tds","passing_yards","passing_tds"]:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = merged[c].fillna(0.0)
    return merged

def compute_dom_rdom_pdom(players: pd.DataFrame, team_pr: pd.DataFrame, team_ru: pd.DataFrame, elusive_df: pd.DataFrame = None) -> pd.DataFrame:
    m = (players.merge(
            team_pr[["year","team","team_passrecv_yards","team_passrecv_tds"]],
            on=["year","team"],
            how="left"
        ).merge(
            team_ru[["year","team","team_rush_yards","team_rush_tds"]],
            on=["year","team"],
            how="left"
        ))
    
    # Merge elusive rating data if available
    if elusive_df is not None and not elusive_df.empty:
        # Create normalized player names for better matching
        m["player_normalized"] = m["player"].map(normalize_player_name)
        elusive_normalized = elusive_df.copy()
        elusive_normalized["player_normalized"] = elusive_normalized["player"].map(normalize_player_name)
        
        # Merge on normalized names
        m = m.merge(
            elusive_normalized[["year","team","player_normalized","elusive_rating"]],
            on=["year","team","player_normalized"],
            how="left"
        )
        
        # Drop the temporary normalized column
        m = m.drop(columns=["player_normalized"])
        
        print(f"🔗 Merged elusive ratings: {m['elusive_rating'].notna().sum()} matches found")
    else:
        m["elusive_rating"] = 0.0

    for col in ["team_passrecv_yards","team_passrecv_tds","team_rush_yards","team_rush_tds"]:
        if col not in m.columns:
            m[col] = 0.0
        m[col] = m[col].fillna(0.0)

    # DOM (Receiving) - multiply by 100 and round to 2 decimal places
    m["DOM"] = (0.5 * (
        (m["receiving_yards"] / m["team_passrecv_yards"]).where(m["team_passrecv_yards"] > 0, 0) +
        (m["receiving_tds"]   / m["team_passrecv_tds"]).where(m["team_passrecv_tds"] > 0, 0)
    ) * 100).round(2)
    
    # RDOM (Rushing) - multiply by 100 and round to 2 decimal places
    m["RDOM"] = (0.5 * (
        (m["rushing_yards"] / m["team_rush_yards"]).where(m["team_rush_yards"] > 0, 0) +
        (m["rushing_tds"]   / m["team_rush_tds"]).where(m["team_rush_tds"] > 0, 0)
    ) * 100).round(2)
    
    # PDOM (Passing) - multiply by 100 and round to 2 decimal places — team pass+rec totals as baseline
    m["PDOM"] = (0.5 * (
        (m["passing_yards"] / m["team_passrecv_yards"]).where(m["team_passrecv_yards"] > 0, 0) +
        (m["passing_tds"]   / m["team_passrecv_tds"]).where(m["team_passrecv_tds"] > 0, 0)
    ) * 100).round(2)

    cm = m["conference"].map(conf_multiplier)
    m["DOM+"]  = (m["DOM"]  * cm).round(2)
    m["RDOM+"] = (m["RDOM"] * cm).round(2)
    m["PDOM+"] = (m["PDOM"] * cm).round(2)

    out_cols = [
        "player","team","year","conference",
        "receiving_yards","receiving_tds",
        "rushing_yards","rushing_tds",
        "passing_yards","passing_tds",
        "DOM","DOM+","RDOM","RDOM+","PDOM","PDOM+",
        "elusive_rating"
    ]
    for c in ["passing_yards","passing_tds"]:
        if c not in m.columns: m[c] = 0.0

    # Filter out rows with blank/empty player names
    result = m[out_cols].copy()
    result = result[result['player'].str.strip().str.len() > 0]  # Remove empty player names
    return result.sort_values(["player","team","year"]).reset_index(drop=True)

# ---------------------------- Save per-year -----------------------------
def save_per_year(df: pd.DataFrame, prefix: str = "dom_master"):
    ensure_outdir()
    for y, chunk in df.groupby("year"):
        outp = OUT_DIR / f"{prefix}_{int(y)}.csv"
        chunk.to_csv(outp, index=False)
        print(f"✅ Wrote {int(y)} -> {outp}")

# ----------------------------- Orchestration ----------------------------
def run_for_years(years: List[int], base_dir: str, strict: bool) -> pd.DataFrame:
    """Build paths for each year, load, merge, compute, and save outputs."""
    all_player_frames = []
    all_team_pr_frames = []
    all_team_ru_frames = []

    for y in sorted(set(years)):
        print(f"📅 Year {y}")

        rec, rush, pas, team_pr, team_ru = year_paths(base_dir, y)

        # Check presence (respect strict mode)
        rec_ok  = _check_exists(rec,  strict, "player receiving")
        rush_ok = _check_exists(rush, strict, "player rushing")
        pas_ok  = _check_exists(pas,  strict, "player passing")
        pr_ok   = _check_exists(team_pr, strict, "team pass/rec totals")
        ru_ok   = _check_exists(team_ru, strict, "team rush totals")

        # Load players (missing files just skip unless strict)
        recv_df = load_player_receiving([rec]) if rec_ok else pd.DataFrame()
        rush_df = load_player_rushing([rush]) if rush_ok else pd.DataFrame()
        pas_df  = load_player_passing([pas])  if pas_ok  else pd.DataFrame()
        players = combine_players_by_key(recv_df, rush_df, pas_df)
        all_player_frames.append(players)

        # Load teams
        team_pr_df = load_team_passrecv_totals([team_pr]) if pr_ok else pd.DataFrame()
        team_ru_df = load_team_rush_totals([team_ru]) if ru_ok else pd.DataFrame()
        all_team_pr_frames.append(team_pr_df)
        all_team_ru_frames.append(team_ru_df)

    # Combine across years
    players_all = pd.concat([f for f in all_player_frames if not f.empty], ignore_index=True) if any(not f.empty for f in all_player_frames) else pd.DataFrame()
    team_pr_all = pd.concat([f for f in all_team_pr_frames if not f.empty], ignore_index=True) if any(not f.empty for f in all_team_pr_frames) else pd.DataFrame()
    team_ru_all = pd.concat([f for f in all_team_ru_frames if not f.empty], ignore_index=True) if any(not f.empty for f in all_team_ru_frames) else pd.DataFrame()

    if players_all.empty:
        print("No player rows loaded; exiting.")
        return pd.DataFrame()

    # Load PFF elusive rating data for all years
    print("📊 Loading PFF elusive rating data...")
    all_elusive_frames = []
    for year in years:
        elusive_df = load_pff_elusive_rating(year)
        if not elusive_df.empty:
            all_elusive_frames.append(elusive_df)
    
    elusive_all = pd.concat([f for f in all_elusive_frames if not f.empty], ignore_index=True) if all_elusive_frames else pd.DataFrame()

    # Compute metrics & save per year
    out = compute_dom_rdom_pdom(players_all, team_pr_all, team_ru_all, elusive_all)
    save_per_year(out, prefix="dom_master")
    return out

# --------------------------------- CLI ---------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute DOM/DOM+, RDOM/RDOM+, PDOM/PDOM+ from StatHead CSVs by year.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--years", type=int, nargs="+", help="Explicit list of years (e.g., --years 2000 2001 2002)")
    grp.add_argument("--start", type=int, help="Start year")
    ap.add_argument("--end", type=int, help="End year (inclusive) — required with --start")
    ap.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR, help="Base directory that contains 'player_<Y>/' and 'team_total_*' folders.")
    ap.add_argument("--strict", action="store_true", help="Fail if any expected file is missing (default: warn and continue).")
    return ap.parse_args()

def resolve_years(args: argparse.Namespace) -> List[int]:
    if args.years:
        return args.years
    if args.start is not None and args.end is not None:
        if args.end < args.start:
            raise ValueError("--end must be >= --start")
        return list(range(args.start, args.end + 1))
    raise ValueError("Provide --years ... OR --start ... --end ...")

if __name__ == "__main__":
    args = parse_args()
    years = resolve_years(args)
    df = run_for_years(years, base_dir=args.base_dir, strict=args.strict)
    # Show a quick preview if any
    if df is not None and not df.empty:
        print(df.head(20))
