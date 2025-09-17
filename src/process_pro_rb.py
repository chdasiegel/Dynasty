# process_pro_rb.py
"""
Build a dict {player_name_clean: DataFrame} for NFL RBs using nfl_data_py.

Depends on utils.py:
- clean_player_name
- log

Usage:
    from process_pro_rb import run_pro_rb_player

    rb_dict = run_pro_rb_player(years=range(2016, 2025), s_type="REG", verbose=False)
    print(len(rb_dict))
    print(rb_dict.get("Christian McCaffrey"))
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import pandas as pd

from src.utils import clean_player_name, log

pd.set_option("display.max_columns", None)

__all__ = ["run_pro_rb_player"]

# Columns to retain for RB analysis
DEFAULT_RB_COLUMNS: List[str] = [
    "player_name", "team", "season", "week", "games",
    # Rushing
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
    "rushing_2pt_conversions", "ry_sh", "rtd_sh", "rfd_sh", "rtdfd_sh",
    # Receiving
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "receiving_2pt_conversions", "racr", "target_share", "air_yards_share",
    "wopr_x", "tgt_sh", "ay_sh", "yac_sh", "wopr_y", "ppr_sh",
    # Fantasy / dominance
    "fantasy_points", "fantasy_points_ppr", "dom", "w8dom",
]


def _fetch_rb_df_for_year(
    year: int,
    s_type: str,
    keep_cols: List[str],
    verbose: bool,
) -> Optional[pd.DataFrame]:
    """Fetch seasonal + roster data for one year, filter to RBs, keep desired cols."""
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

    rbs = merged[merged["position"] == "RB"].copy()

    if "player_id" in rbs.columns:
        rbs = rbs.drop(columns=["player_id"])

    if "player_name" in rbs.columns:
        cols = ["player_name"] + [c for c in rbs.columns if c != "player_name"]
        rbs = rbs[cols]

    cols_to_keep = [c for c in keep_cols if c in rbs.columns]
    rb_filtered = rbs[cols_to_keep].copy()

    rb_filtered["player_name_clean"] = rb_filtered["player_name"].apply(clean_player_name)

    log(f"Loaded {len(rb_filtered)} RB rows for {year}", level="success", verbose=verbose)
    return rb_filtered


def run_pro_rb_player(
    years: Iterable[int] = range(2016, 2025),
    *,
    s_type: str = "REG",                     # "REG", "POST", or "BOTH"
    keep_cols: Optional[List[str]] = None,   # defaults to DEFAULT_RB_COLUMNS
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Return dict {player_name_clean: DataFrame} aggregating RB rows across `years`.

    Steps:
      - Pull seasonal data + rosters per year via nfl_data_py
      - Merge, filter to RBs, keep selected columns
      - Clean player names into 'player_name_clean'
      - Group rows per player and return dictionary of DataFrames
    """
    keep_cols = keep_cols or DEFAULT_RB_COLUMNS

    rb_season_dict: Dict[str, pd.DataFrame] = {}

    for year in years:
        rb_filtered = _fetch_rb_df_for_year(year, s_type=s_type, keep_cols=keep_cols, verbose=verbose)
        if rb_filtered is None or rb_filtered.empty:
            continue

        # Aggregate into dict
        for name, group in rb_filtered.groupby("player_name_clean", dropna=False):
            existing = rb_season_dict.get(name, pd.DataFrame())
            group_out = group.drop(columns=[c for c in ["player_name", "player_name_clean"] if c in group.columns])
            rb_season_dict[name] = pd.concat([existing, group_out], ignore_index=True)

    log(f"Created RB dict with {len(rb_season_dict)} players.", level="success", verbose=verbose)
    return rb_season_dict


if __name__ == "__main__":
    d = run_pro_rb_player(verbose=True)
    print(f"Players: {len(d)}")
