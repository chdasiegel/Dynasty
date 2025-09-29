# process_pro_wr.py
"""
Build a dict {player_name_clean: DataFrame} for NFL WRs using nfl_data_py.

Depends on utils.py:
- clean_player_name
- log

Usage:
    from src.process.process_pro_wr import run_pro_wr_player

    wr_dict = run_pro_wr_player(years=range(2016, 2025), s_type="REG", verbose=False)
    print(len(wr_dict))
    print(wr_dict.get("Rashee Rice"))
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd

from src.utils import clean_player_name, log

pd.set_option("display.max_columns", None)

__all__ = ["run_pro_wr_player"]

# Columns to retain for WR analysis
DEFAULT_WR_COLUMNS: List[str] = [
    "player_name", "team", "season", "week", "games",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "receiving_2pt_conversions", "racr", "target_share", "air_yards_share",
    "wopr_x", "tgt_sh", "ay_sh", "yac_sh", "wopr_y", "ppr_sh",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
    "rushing_2pt_conversions", "ry_sh", "rtd_sh", "rfd_sh", "rtdfd_sh",
    "fantasy_points", "fantasy_points_ppr", "dom", "w8dom",
]


def _fetch_wr_df_for_year(
    year: int,
    s_type: str,
    keep_cols: List[str],
    verbose: bool,
) -> Optional[pd.DataFrame]:
    """Fetch seasonal + roster data for one year, filter to WRs, keep desired cols."""
    try:
        from nfl_data_py import import_seasonal_data, import_seasonal_rosters
    except Exception as e:
        raise ImportError("nfl_data_py is required. Install with: pip install nfl_data_py") from e

    stats = import_seasonal_data([year], s_type=s_type)
    rosters = import_seasonal_rosters([year])

    merged = stats.merge(
        rosters[["player_id", "player_name", "position", "team"]],
        on="player_id",
        how="left",
    )

    wrs = merged[merged["position"] == "WR"].copy()

    if "player_id" in wrs.columns:
        wrs = wrs.drop(columns=["player_id"])

    if "player_name" in wrs.columns:
        cols = ["player_name"] + [c for c in wrs.columns if c != "player_name"]
        wrs = wrs[cols]

    cols_to_keep = [c for c in keep_cols if c in wrs.columns]
    wr_filtered = wrs[cols_to_keep].copy()

    wr_filtered["player_name_clean"] = wr_filtered["player_name"].apply(clean_player_name)

    log(f"Loaded {len(wr_filtered)} WR rows for {year}", level="success", verbose=verbose)
    return wr_filtered


def run_pro_wr_player(
    years: Iterable[int] = range(2016, 2025),
    *,
    s_type: str = "REG",                      # "REG", "POST", or "BOTH"
    keep_cols: Optional[List[str]] = None,    # defaults to DEFAULT_WR_COLUMNS
    verbose: bool = False,
    return_yearly: bool = False,              # if True, also return {year: DataFrame}
) -> Dict[str, pd.DataFrame] | Tuple[Dict[str, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Return dict {player_name_clean: DataFrame} aggregating WR rows across `years`.
    Optionally also return a dict of yearly DataFrames when `return_yearly=True`.
    """
    keep_cols = keep_cols or DEFAULT_WR_COLUMNS

    wr_season_dict: Dict[str, pd.DataFrame] = {}
    wr_data_by_year: Dict[int, pd.DataFrame] = {}

    for year in years:
        wr_filtered = _fetch_wr_df_for_year(year, s_type=s_type, keep_cols=keep_cols, verbose=verbose)
        if wr_filtered is None or wr_filtered.empty:
            continue

        wr_data_by_year[year] = wr_filtered

        for name, group in wr_filtered.groupby("player_name_clean", dropna=False):
            existing = wr_season_dict.get(name, pd.DataFrame())
            group_out = group.drop(columns=[c for c in ["player_name", "player_name_clean"] if c in group.columns])
            wr_season_dict[name] = pd.concat([existing, group_out], ignore_index=True)

    log(f"Created WR dict with {len(wr_season_dict)} players.", level="success", verbose=verbose)

    if return_yearly:
        return wr_season_dict, wr_data_by_year
    return wr_season_dict


if __name__ == "__main__":
    d, by_year = run_pro_wr_player(verbose=True, return_yearly=True)
    print(f"Players: {len(d)}  |  Years loaded: {len(by_year)}")
