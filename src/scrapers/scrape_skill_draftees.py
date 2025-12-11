#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_skill_draftees.py

Scrape drafted skill-position players (QB/RB/WR/TE) from
Pro-Football-Reference for a given year range.

Outputs columns:
    - player_name
    - pos
    - draft_year
    - team
    - draft_round        (from 'Rnd')
    - pick_in_round      (computed within round)
    - pick_overall       (from 'Pick')
    - pfr_draft_url
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, asdict
from io import StringIO
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import requests

# ----------------- Config -----------------

BASE_URL = "https://www.pro-football-reference.com"
DRAFT_URL_TEMPLATE = BASE_URL + "/years/{year}/draft.htm"

SKILL_POS: Set[str] = {"QB", "RB", "WR", "TE"}

POSITION_NORMALIZATION = {
    "FB": "RB",
    "HB": "RB",
    "TB": "RB",
}

DEFAULT_OUT = (
    Path("/Users/chasesiegel/Desktop/Comp_Sci/Capstone/Dynasty")
    / "data"
    / "scraper"
    / "skill_draftees_2000_2025.csv"
)


@dataclass
class DraftPick:
    player_name: str
    pos: str
    draft_year: int
    team: str
    draft_round: Optional[int]
    pick_in_round: Optional[int]
    pick_overall: Optional[int]
    pfr_draft_url: str


# ----------------- Helpers -----------------


def fetch_year_html(year: int) -> str:
    """Fetch raw HTML for a given draft year."""
    url = DRAFT_URL_TEMPLATE.format(year=year)
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_draft_table(year: int) -> pd.DataFrame:
    """
    Return the draft table (id='drafts') as a DataFrame.

    - Uses read_html on the specific table id.
    - Flattens MultiIndex columns by preferring the second level
      (where 'Player', 'Pos', 'Rnd', 'Pick', etc. usually live).
    """
    html = fetch_year_html(year)

    # Wrap in StringIO to avoid FutureWarning and be explicit
    dfs = pd.read_html(StringIO(html), attrs={"id": "drafts"})
    if not dfs:
        raise RuntimeError(f"No draft table found for year {year}")

    df = dfs[0]

    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for top, bottom in df.columns:
            top = str(top)
            bottom = str(bottom)
            # Prefer the bottom (more specific) level if it isn't Unnamed
            if bottom and not bottom.startswith("Unnamed"):
                new_cols.append(bottom)
            elif top and not top.startswith("Unnamed"):
                new_cols.append(top)
            else:
                # Fallback: whatever isn't empty
                new_cols.append(bottom or top)
        df.columns = new_cols
    else:
        df.columns = [str(col) for col in df.columns]

    return df


def normalize_pos(pos: str) -> str:
    pos = (pos or "").strip()
    pos = POSITION_NORMALIZATION.get(pos, pos)
    return pos


def _resolve_column(name_hint: str, columns: list[str]) -> str:
    """
    Find the first column whose name contains name_hint (case-insensitive).
    Raise a clear error if not found.
    """
    lower_hint = name_hint.lower()
    for col in columns:
        if lower_hint in col.lower():
            return col
    raise RuntimeError(
        f"Could not find a column containing '{name_hint}' in columns: {columns}"
    )


def scrape_year(year: int) -> List[DraftPick]:
    df = fetch_draft_table(year)

    cols = list(df.columns)

    # Dynamically resolve the column names
    col_player = _resolve_column("Player", cols)
    col_pos = _resolve_column("Pos", cols)
    col_team = _resolve_column("Tm", cols)
    col_rnd = _resolve_column("Rnd", cols)    # 'Rnd' or similar
    col_pick = _resolve_column("Pick", cols)  # 'Pick' or similar

    # Drop repeated header rows / blanks
    df = df[df[col_player].notna()].copy()
    df = df[df[col_player] != col_player]

    # Coerce numeric
    df["draft_round"] = pd.to_numeric(df[col_rnd], errors="coerce").astype("Int64")
    df["pick_overall"] = pd.to_numeric(df[col_pick], errors="coerce").astype("Int64")

    # Compute pick_in_round within each round
    df = df.sort_values(["draft_round", "pick_overall"])
    df["pick_in_round"] = df.groupby("draft_round").cumcount() + 1

    picks: List[DraftPick] = []

    for _, row in df.iterrows():
        pos_raw = str(row[col_pos])
        pos_norm = normalize_pos(pos_raw)
        if pos_norm not in SKILL_POS:
            continue

        player_name = str(row[col_player]).strip()
        if not player_name:
            continue

        team = str(row[col_team]).strip() if pd.notna(row[col_team]) else ""

        draft_round = int(row["draft_round"]) if pd.notna(row["draft_round"]) else None
        pick_overall = int(row["pick_overall"]) if pd.notna(row["pick_overall"]) else None
        pick_in_round = int(row["pick_in_round"]) if pd.notna(row["pick_in_round"]) else None

        picks.append(
            DraftPick(
                player_name=player_name,
                pos=pos_norm,
                draft_year=year,
                team=team,
                draft_round=draft_round,
                pick_in_round=pick_in_round,
                pick_overall=pick_overall,
                pfr_draft_url=DRAFT_URL_TEMPLATE.format(year=year),
            )
        )

    return picks


def scrape_skill_draftees(start_year: int, end_year: int, sleep_sec: float = 1.0) -> pd.DataFrame:
    all_picks: List[DraftPick] = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching draft {year}...")
        year_picks = scrape_year(year)
        print(f"  -> Found {len(year_picks)} skill-position picks")
        all_picks.extend(year_picks)
        time.sleep(sleep_sec)

    df = pd.DataFrame([asdict(p) for p in all_picks])
    if not df.empty:
        df = df.sort_values(
            ["draft_year", "draft_round", "pick_in_round"], ignore_index=True
        )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Scrape drafted skill-position players (QB/RB/WR/TE) from PFR."
    )
    parser.add_argument("--start", type=int, default=2000, help="Start year (inclusive). Default: 2000")
    parser.add_argument("--end", type=int, default=2025, help="End year (inclusive). Default: 2025")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"Output CSV path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between year requests. Default: 1.0",
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = scrape_skill_draftees(args.start, args.end, sleep_sec=args.sleep)
    print(f"\nTotal skill-position picks {args.start}-{args.end}: {len(df)}")

    df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
