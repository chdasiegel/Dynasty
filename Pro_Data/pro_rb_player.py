#Example Usage
#from pro_rb_player import load_pro_rb_data
#pro_rb_player, rb_data_by_year = load_pro_rb_data()

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

def load_pro_rb_data(start_year=2016, end_year=2025):
    # Define RB columns
    rb_columns = [
        'player_name', 'team', 'season', 'week', 'games',
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
        'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
        'rushing_2pt_conversions', 'ry_sh', 'rtd_sh', 'rfd_sh', 'rtdfd_sh',

        'receptions', 'targets', 'receiving_yards', 'receiving_tds',
        'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
        'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa',
        'receiving_2pt_conversions', 'racr', 'target_share', 'air_yards_share',
        'wopr_x', 'tgt_sh', 'ay_sh', 'yac_sh', 'wopr_y', 'ppr_sh',

        'fantasy_points', 'fantasy_points_ppr', 'games', 'dom', 'w8dom'
    ]

    rb_data_by_year = {}
    pro_rb_player = {}

    for year in range(start_year, end_year):
        stats = import_seasonal_data([year], s_type="REG")
        rosters = import_seasonal_rosters([year])

        merged = stats.merge(
            rosters[['player_id', 'player_name', 'position', 'team']],
            on='player_id',
            how='left'
        )

        rbs = merged[merged['position'] == 'RB'].copy()
        rbs.drop(columns=['player_id'], inplace=True)
        cols = ['player_name'] + [col for col in rbs.columns if col != 'player_name']
        rbs = rbs[cols]

        rb_filtered = rbs[[col for col in rb_columns if col in rbs.columns]].copy()
        rb_filtered['player_name_clean'] = rb_filtered['player_name'].apply(clean_player_name)

        rb_data_by_year[year] = rb_filtered

        for name, group in rb_filtered.groupby('player_name_clean'):
            pro_rb_player[name] = pd.concat(
                [pro_rb_player.get(name, pd.DataFrame()), group.drop(columns=['player_name', 'player_name_clean'])],
                ignore_index=True
            )

        print(f"✅ Loaded {len(rb_filtered)} RB rows for {year}")

    return pro_rb_player, rb_data_by_year
