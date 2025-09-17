#Example Usage
#from pro_te_player import run_pro_te_player
#pro_te_dict = run_pro_te_player()

from nfl_data_py import import_seasonal_data, import_seasonal_rosters
import pandas as pd
import re

pd.set_option('display.max_columns', None)

# Clean player names (punctuation and suffixes)
def clean_player_name(player_name):
    if not isinstance(player_name, str):
        return player_name
    player_name = re.sub(r'[^\w\s]', '', player_name)
    suffixes = ['Jr', 'Sr', 'II', 'III', 'IV', 'V']
    pattern = r'\b(?:' + '|'.join(suffixes) + r')\b'
    player_name = re.sub(pattern, '', player_name, flags=re.IGNORECASE)
    return ' '.join(player_name.split())

# Columns to keep for TEs
te_columns = [
    'player_name', 'team', 'season', 'week', 'games',
    'receptions', 'targets', 'receiving_yards', 'receiving_tds',
    'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
    'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa',
    'receiving_2pt_conversions', 'racr', 'target_share', 'air_yards_share',
    'wopr_x', 'tgt_sh', 'ay_sh', 'yac_sh', 'wopr_y', 'ppr_sh',

    'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
    'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
    'rushing_2pt_conversions', 'ry_sh', 'rtd_sh', 'rfd_sh', 'rtdfd_sh',

    'fantasy_points', 'fantasy_points_ppr', 'games', 'dom', 'w8dom'
]

# Main function
def run_pro_te_player():
    years = list(range(2016, 2025))
    te_data_by_year = {}
    te_season_dict = {}

    for year in years:
        stats = import_seasonal_data([year], s_type="REG")
        rosters = import_seasonal_rosters([year])

        merged = stats.merge(
            rosters[['player_id', 'player_name', 'position', 'team']],
            on='player_id',
            how='left'
        )

        tes = merged[merged['position'] == 'TE'].copy()
        tes.drop(columns=['player_id'], inplace=True)
        cols = ['player_name'] + [col for col in tes.columns if col != 'player_name']
        tes = tes[cols]

        te_filtered = tes[[col for col in te_columns if col in tes.columns]].copy()
        te_filtered['player_name_clean'] = te_filtered['player_name'].apply(clean_player_name)

        # Store by year
        te_data_by_year[year] = te_filtered

        # Aggregate into dict by player
        for name, group in te_filtered.groupby('player_name_clean'):
            te_season_dict[name] = pd.concat(
                [te_season_dict.get(name, pd.DataFrame()), group.drop(columns=['player_name', 'player_name_clean'])],
                ignore_index=True
            )

        print(f"✅ Loaded {len(te_filtered)} TE rows for {year}")

    return te_season_dict
