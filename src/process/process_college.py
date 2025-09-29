# process_college.py
"""
Build a dictionary {player_name: DataFrame} from college CSVs named {year}_{stat}.csv.

Default CSV location (new): <project_root>/data/CFB_Data/
Examples:
    data/CFB_Data/2016_passing.csv
    data/CFB_Data/2016_rushing.csv
    data/CFB_Data/2016_receiving.csv

Usage:
    from src.process.process_college import build_player_dict
    d = build_player_dict(verbose=True)                       # uses data/CFB_Data by default
    d = build_player_dict(data_dir="path/to/other/folder")    # override if needed
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd
from src.utils import (
    clean_player_name,
    reorder_after,
    group_to_player_dict,
    safe_read_csv,
    log,
)

pd.set_option("display.max_columns", None)

__all__ = ["build_player_dict"]

# Resolve project root as the parent of src/, then default data dir to <project_root>/data/CFB_Data
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "CFB_Data"


def _load_one_csv(path: Path, year: int, stat: str, verbose: bool) -> pd.DataFrame | None:
    """Read and normalize a single CSV at `path` for given year/stat."""
    df = safe_read_csv(str(path), encoding="latin1", dtype=str, parse_dates=False)
    if df is None:
        log(f"Error reading {path}", level="error", verbose=verbose)
        return None

    # Standardize columns
    if "Player" in df.columns:
        df = df.rename(columns={"Player": "player"})

    if "Awards" in df.columns:
        df = df.drop(columns=["Awards"])

    # Rename ambiguous columns if present
    rename_dict = {"Yds.2": "Scrim_Yds", "Avg": "Scrim_Avg", "TD.2": "Tot_TD"}
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    # Required columns
    if "player" not in df.columns or "Rk" not in df.columns:
        log(f"'player' or 'Rk' column missing in {path.name}, skipping.", level="warn", verbose=verbose)
        return None

    # Normalize types + annotate
    df["Rk"] = pd.to_numeric(df["Rk"], errors="coerce")
    df["season"] = year
    df["stat_type"] = stat
    return df


def build_player_dict(
    years: Iterable[int] = range(2016, 2025),
    stats: Sequence[str] = ("passing", "rushing", "receiving"),
    *,
    data_dir: str | Path | None = None,   # 👈 defaults to Dynasty/data/CFB_Data
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build a dictionary of player DataFrames from local CSVs.

    Processing:
      - Reads CSVs per year/stat from `data_dir` (default: <project_root>/data/CFB_Data)
      - Renames 'Player' -> 'player', drops 'Awards' if present
      - Cleans 'player' names
      - Drops rows with missing Rk
      - Keeps the lowest Rk per (player, season)
      - Reorders columns to put 'season' after 'player'
      - Groups into {player: DataFrame}
    """
    # Resolve default data directory
    base_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    if verbose:
        log(f"Reading college CSVs from: {base_dir}", level="info", verbose=verbose)

    frames: list[pd.DataFrame] = []

    for year in years:
        for stat in stats:
            path = base_dir / f"{year}_{stat}.csv"
            if not path.exists():
                log(f"File not found: {path}", level="warn", verbose=verbose)
                continue

            df = _load_one_csv(path, year, stat, verbose)
            if df is not None:
                frames.append(df)
                log(f"Loaded {path.name} ({len(df)} rows)", level="success", verbose=verbose)

    if not frames:
        log("No data found. Please check your CSV files.", level="error", verbose=verbose)
        return {}

    combined = pd.concat(frames, ignore_index=True)

    # Clean names
    combined["player"] = combined["player"].str.replace("*", "", regex=False)
    combined["player"] = combined["player"].apply(clean_player_name)

    # Drop rows without rank
    combined = combined.dropna(subset=["Rk"])

    # Keep the best (lowest) rank per player-season
    best_idx = combined.groupby(["player", "season"])["Rk"].idxmin()
    filtered = combined.loc[best_idx].copy()

    # Reorder: put 'season' right after 'player'
    filtered = reorder_after(filtered, move_col="season", after_col="player")

    # Drop Rk if present
    if "Rk" in filtered.columns:
        filtered = filtered.drop(columns=["Rk"])

    # Group into dictionary
    player_college_dict: Dict[str, pd.DataFrame] = group_to_player_dict(filtered, player_col="player")
    log(
        f"Created dictionary for {len(player_college_dict)} players (lowest Rk per season, all seasons included).",
        level="success",
        verbose=verbose,
    )
    return player_college_dict


if __name__ == "__main__":
    d = build_player_dict(verbose=True)  # defaults to data/CFB_Data
    print(f"Players: {len(d)}")
