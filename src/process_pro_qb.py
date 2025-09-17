# process_pro_qb.py
"""
Build a dict {player_name_clean: DataFrame} for NFL QBs using nfl_data_py.

Depends on utils.py:
- clean_player_name
- log

Usage:
    from process_pro_qb import run_pro_qb_player

    qb_dict = run_pro_qb_player(years=range(2016, 2025), s_type="REG", verbose=False)
    print(len(qb_dict))
    print(qb_dict.get("Patrick Mahomes"))
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import pandas as pd

from src.utils import clean_player_name, log

pd.set_option("display.max_columns", None)

__all__ = ["run_pro_qb_player"]

# Columns to retain for QB analysis
DEFAULT_QB_COLUMNS: List[str] = [
    "player_name", "team", "season", "week", "games",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost",
    "passing_air_yards", "passing_yards_after_catch", "passing_first_downs",
    "passing_epa", "passing_2pt_conversions", "pacr", "dakota",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
    "rushing_2pt_conversions", "ry_sh", "rtd_sh", "rfd_sh", "rtdfd_sh",
    "fantasy_points", "fantasy_points_ppr",
]


def _fetch_qb_df_for_year(
    year: int,
    s_type: str,
    keep_cols: List[str],
    verbose: bool,
) -> Optional[pd.DataFrame]:
    """Fetch seasonal + roster data for one year, filter to QBs, keep desired cols."""
    try:
        from nfl_data_py import import_seasonal_data, import_seasonal_rosters
    except Exception as e:  # pragma: no cover
        raise ImportError("nfl_data_py is required. Install with: pip install nfl_data_py") from e

    # Pull data
    stats = import_seasonal_data([year], s_type=s_type)
    rosters = import_seasonal_rosters([year])

    # Merge rosters to add position and team
    merged = stats.merge(
        rosters[["player_id", "player_name", "position", "team"]],
        on="player_id",
        how="left",
    )

    # Filter to QBs
    qbs = merged[merged["position"] == "QB"].copy()

    # Drop ids and put player_name first when present
    if "player_id" in qbs.columns:
        qbs = qbs.drop(columns=["player_id"])
    if "player_name" in qbs.columns:
        cols = ["player_name"] + [c for c in qbs.columns if c != "player_name"]
        qbs = qbs[cols]

    # Keep only the columns that exist
    cols_to_keep = [c for c in keep_cols if c in qbs.columns]
    qb_filtered = qbs[cols_to_keep].copy()

    # Clean name
    qb_filtered["player_name_clean"] = qb_filtered["player_name"].apply(clean_player_name)

    log(f"Loaded {len(qb_filtered)} QB rows for {year}", level="success", verbose=verbose)
    return qb_filtered


def run_pro_qb_player(
    years: Iterable[int] = range(2016, 2025),
    *,
    s_type: str = "REG",                     # "REG", "POST", or "BOTH"
    keep_cols: Optional[List[str]] = None,   # defaults to DEFAULT_QB_COLUMNS
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Return dict {player_name_clean: DataFrame} aggregating QB rows across `years`.

    Steps:
      - Pull seasonal data + rosters per year via nfl_data_py
      - Merge, filter to QBs, keep selected columns
      - Clean player names into 'player_name_clean'
      - Group rows per player and return dictionary of DataFrames
    """
    keep_cols = keep_cols or DEFAULT_QB_COLUMNS

    qb_season_dict: Dict[str, pd.DataFrame] = {}

    for year in years:
        qb_filtered = _fetch_qb_df_for_year(year, s_type=s_type, keep_cols=keep_cols, verbose=verbose)
        if qb_filtered is None or qb_filtered.empty:
            continue

        # Aggregate into dict
        for name, group in qb_filtered.groupby("player_name_clean", dropna=False):
            existing = qb_season_dict.get(name, pd.DataFrame())
            # Drop the name columns from the per-player frame to avoid duplication
            group_out = group.drop(columns=[c for c in ["player_name", "player_name_clean"] if c in group.columns])
            qb_season_dict[name] = pd.concat([existing, group_out], ignore_index=True)

    log(f"Created qb dict with {len(qb_season_dict)} players.", level="success", verbose=verbose)
    return qb_season_dict


if __name__ == "__main__":
    d = run_pro_qb_player(verbose=True)
    print(f"Players: {len(d)}")
