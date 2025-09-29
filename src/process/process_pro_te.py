# process_pro_te.py
"""
Build a dict {player_name_clean: DataFrame} for NFL TEs using nfl_data_py.

Depends on utils.py:
- clean_player_name
- log

Usage:
    from src.process.process_pro_te import run_pro_te_player

    te_dict = run_pro_te_player(years=range(2016, 2025), s_type="REG", verbose=False)
    print(len(te_dict))
    print(te_dict.get("Travis Kelce"))
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import pandas as pd

from src.utils import clean_player_name, log

pd.set_option("display.max_columns", None)

__all__ = ["run_pro_te_player"]

# Columns to retain for TE analysis
DEFAULT_TE_COLUMNS: List[str] = [
    "player_name", "team", "season", "week", "games",
    # Receiving
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "receiving_2pt_conversions", "racr", "target_share", "air_yards_share",
    "wopr_x", "tgt_sh", "ay_sh", "yac_sh", "wopr_y", "ppr_sh",
    # Rushing (occasional)
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
    "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
    "rushing_2pt_conversions", "ry_sh", "rtd_sh", "rfd_sh", "rtdfd_sh",
    # Fantasy / dominance
    "fantasy_points", "fantasy_points_ppr", "dom", "w8dom",
]


def _fetch_te_df_for_year(
    year: int,
    s_type: str,
    keep_cols: List[str],
    verbose: bool,
) -> Optional[pd.DataFrame]:
    """Fetch seasonal + roster data for one year, filter to TEs, keep desired cols."""
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

    tes = merged[merged["position"] == "TE"].copy()

    if "player_id" in tes.columns:
        tes = tes.drop(columns=["player_id"])

    if "player_name" in tes.columns:
        cols = ["player_name"] + [c for c in tes.columns if c != "player_name"]
        tes = tes[cols]

    cols_to_keep = [c for c in keep_cols if c in tes.columns]
    te_filtered = tes[cols_to_keep].copy()

    te_filtered["player_name_clean"] = te_filtered["player_name"].apply(clean_player_name)

    log(f"Loaded {len(te_filtered)} TE rows for {year}", level="success", verbose=verbose)
    return te_filtered


def run_pro_te_player(
    years: Iterable[int] = range(2016, 2025),
    *,
    s_type: str = "REG",                     # "REG", "POST", or "BOTH"
    keep_cols: Optional[List[str]] = None,   # defaults to DEFAULT_TE_COLUMNS
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Return dict {player_name_clean: DataFrame} aggregating TE rows across `years`.

    Steps:
      - Pull seasonal data + rosters per year via nfl_data_py
      - Merge, filter to TEs, keep selected columns
      - Clean player names into 'player_name_clean'
      - Group rows per player and return dictionary of DataFrames
    """
    keep_cols = keep_cols or DEFAULT_TE_COLUMNS

    te_season_dict: Dict[str, pd.DataFrame] = {}

    for year in years:
        te_filtered = _fetch_te_df_for_year(year, s_type=s_type, keep_cols=keep_cols, verbose=verbose)
        if te_filtered is None or te_filtered.empty:
            continue

        for name, group in te_filtered.groupby("player_name_clean", dropna=False):
            existing = te_season_dict.get(name, pd.DataFrame())
            group_out = group.drop(columns=[c for c in ["player_name", "player_name_clean"] if c in group.columns])
            te_season_dict[name] = pd.concat([existing, group_out], ignore_index=True)

    log(f"Created TE dict with {len(te_season_dict)} players.", level="success", verbose=verbose)
    return te_season_dict


if __name__ == "__main__":
    d = run_pro_te_player(verbose=True)
    print(f"Players: {len(d)}")
