# Example Usage:
# from pro_wr_player import load_pro_wr_data
# pro_wr_player, wr_data_by_year = load_pro_wr_data()

from nfl_data_py import import_seasonal_data, import_seasonal_rosters
import pandas as pd
import re

pd.set_option('display.max_columns', None)

# Function to clean player names (remove punctuation and suffixes)
def clean_player_name(player_name):
    if not isinstance(player_name, str):
        return player_name
    player_name = re.sub(r'[^\w\s]', '', player_name)  # Remove punctuation
    suffixes = ['Jr', 'Sr', 'II', 'III', 'IV', 'V']
    pattern = r'\b(?:' + '|'.join(suffixes) + r')\b'
    player_name = re.sub(pattern, '', player_name, flags=re.IGNORECASE)
    return ' '.join(player_name.split())  # Remove extra spaces

def load_pro_wr_data(start_year=2016, end_year=2025):
    # Define WR columns
    wr_columns = [
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

    wr_data_by_year = {}
    pro_wr_player = {}

    for year in range(start_year, end_year):
        stats = import_seasonal_data([year], s_type="REG")
        rosters = import_seasonal_rosters([year])

        merged = stats.merge(
            rosters[['player_id', 'player_name', 'position', 'team']],
            on='player_id',
            how='left'
        )

        wrs = merged[merged['position'] == 'WR'].copy()
        wrs.drop(columns=['player_id'], inplace=True)
        cols = ['player_name'] + [col for col in wrs.columns if col != 'player_name']
        wrs = wrs[cols]

        wr_filtered = wrs[[col for col in wr_columns if col in wrs.columns]].copy()
        wr_filtered['player_name_clean'] = wr_filtered['player_name'].apply(clean_player_name)

        wr_data_by_year[year] = wr_filtered

        for name, group in wr_filtered.groupby('player_name_clean'):
            pro_wr_player[name] = pd.concat(
                [pro_wr_player.get(name, pd.DataFrame()), group.drop(columns=['player_name', 'player_name_clean'])],
                ignore_index=True
            )

        print(f"✅ Loaded {len(wr_filtered)} WR rows for {year}")

    return pro_wr_player, wr_data_by_year
