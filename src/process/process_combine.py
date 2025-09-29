# process_combine.py
"""
Build a dictionary {player_name: DataFrame} from Combine CSVs named {year}_Combine.csv.

Default CSV location (new): <project_root>/data/Combine/
Examples:
    data/Combine/2016_Combine.csv
    data/Combine/2017_Combine.csv
    ...

Usage:
    from src.process.process_combine import build_combine_dict

    d = build_combine_dict(verbose=True)                       # uses data/Combine by default
    d = build_combine_dict(data_dir="path/to/other/folder")    # override if needed
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from src.utils import (
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

# Resolve project root as the parent of src/, then default to <project_root>/data/Combine
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "Combine"


def _load_one_combine(path: Path, year: int, verbose: bool) -> pd.DataFrame | None:
    """Read and normalize a single Combine CSV at `path` for given year."""
    df = safe_read_csv(str(path), encoding="latin1", dtype=str, parse_dates=False)
    if df is None:
        log(f"Error reading {path}", level="error", verbose=verbose)
        return None

    if "Player" not in df.columns:
        log(f"'Player' column missing in {path.name}, skipping.", level="error", verbose=verbose)
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
    data_dir: str | Path | None = None,   # 👈 defaults to data/Combine
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build a dictionary of player DataFrames from local Combine CSVs in `data/Combine/`.

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
    base_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    if verbose:
        log(f"Reading Combine CSVs from: {base_dir}", level="info", verbose=verbose)

    frames: list[pd.DataFrame] = []

    for year in years:
        path = base_dir / f"{year}_Combine.csv"
        if not path.exists():
            log(f"File not found: {path}", level="warn", verbose=verbose)
            continue

        df = _load_one_combine(path, year, verbose)
        if df is not None:
            frames.append(df)
            log(f"Loaded {path.name} ({len(df)} rows)", level="success", verbose=verbose)

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
    d = build_combine_dict(verbose=True)  # defaults to data/Combine
    print(f"Players: {len(d)}")
