# process_combine.py
"""
Build a dictionary {player_name: DataFrame} from Combine CSVs named {year}_Combine.csv.

CSV files should live in the Combine/ folder, e.g.:
    Combine/2016_Combine.csv
    Combine/2017_Combine.csv
    ...

Usage:
    from process_combine import build_combine_dict

    d = build_combine_dict(verbose=True)
    print(len(d))
    print(d.get("Marvin Harrison"))
"""

from __future__ import annotations

import os
from typing import Dict, Iterable

import pandas as pd
from utils import (
    clean_player_name,
    height_to_inches,
    parse_drafted_column,
    filter_positions,
    reorder_after,
    group_to_player_dict,
    safe_read_csv,
    log,
)

pd.set_option("display.max_columns", None)

__all__ = ["build_combine_dict"]


def _load_one_combine(path: str, year: int, verbose: bool) -> pd.DataFrame | None:
    """Read and normalize a single Combine CSV at `path` for given year."""
    df = safe_read_csv(path, encoding="latin1", dtype=str, parse_dates=False)
    if df is None:
        log(f"Error reading {path}", level="error", verbose=verbose)
        return None

    if "Player" not in df.columns:
        log(f"'Player' column missing in {os.path.basename(path)}, skipping.", level="error", verbose=verbose)
        return None

    df = df.rename(columns={"Player": "player"})
    df["Year"] = str(year)

    if "Yr" in df.columns:
        df = df.drop(columns=["Yr"])

    # Split drafted info if present
    df = parse_drafted_column(df)

    # Filter to QB/RB/WR/TE
    df = filter_positions(df, ["QB", "RB", "WR", "TE"])

    # Normalize height
    if "Ht" in df.columns:
        df["Ht"] = df["Ht"].apply(height_to_inches)

    # Clean player names
    df["player"] = df["player"].apply(clean_player_name)

    return df


def build_combine_dict(
    years: Iterable[int] = range(2016, 2026),
    *,
    data_dir: str = "Combine",   # 👈 default to Combine folder
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build a dictionary of player DataFrames from local Combine CSVs in `Combine/`.

    Processing:
      - Reads CSVs per year
      - Renames 'Player' -> 'player', drops 'Yr' if present
      - Splits 'Drafted (tm/rnd/yr)' into Draft_Team/Round/Pick
      - Cleans 'player' names
      - Converts height to inches
      - Filters to QB/RB/WR/TE
      - Reorders 'Year' to appear after 'player'
      - Groups into {player: DataFrame}
    """
    frames: list[pd.DataFrame] = []

    for year in years:
        path = os.path.join(data_dir, f"{year}_Combine.csv")
        if not os.path.exists(path):
            log(f"File not found: {path}", level="warn", verbose=verbose)
            continue

        df = _load_one_combine(path, year, verbose)
        if df is not None:
            frames.append(df)
            log(f"Loaded {os.path.basename(path)} ({len(df)} rows)", level="success", verbose=verbose)

    if not frames:
        log("No combine data found.", level="error", verbose=verbose)
        return {}

    combined = pd.concat(frames, ignore_index=True)

    # Reorder: put 'Year' after 'player'
    combined = reorder_after(combined, move_col="Year", after_col="player")

    # Group into dictionary
    player_combine_dict: Dict[str, pd.DataFrame] = group_to_player_dict(combined, player_col="player")
    log(f"Created player_combine_dict with {len(player_combine_dict)} players.", level="success", verbose=verbose)

    return player_combine_dict


if __name__ == "__main__":
    d = build_combine_dict(verbose=True)  # defaults to Combine folder
    print(f"Players: {len(d)}")
