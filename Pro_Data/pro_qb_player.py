#Example Usage
#from pro_qb_player import run_pro_qb_player
#pro_qb_dict = run_pro_qb_player()

from nfl_data_py import import_seasonal_data, import_seasonal_rosters
import pandas as pd
import re

pd.set_option('display.max_columns', None)

# Function to clean player names
def clean_player_name(player_name):
    if not isinstance(player_name, str):
        return player_name
    player_name = re.sub(r'[^\w\s]', '', player_name)  # Remove punctuation
    suffixes = ['Jr', 'Sr', 'II', 'III', 'IV', 'V']
    pattern = r'\b(?:' + '|'.join(suffixes) + r')\b'
    player_name = re.sub(pattern, '', player_name, flags=re.IGNORECASE)
    return ' '.join(player_name.split())

# Columns to retain for QB analysis
qb_columns = [
    'player_name', 'team', 'season', 'week', 'games',
    'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
    'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost',
    'passing_air_yards', 'passing_yards_after_catch', 'passing_first_downs',
    'passing_epa', 'passing_2pt_conversions', 'pacr', 'dakota',
    'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
    'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
    'rushing_2pt_conversions', 'ry_sh', 'rtd_sh', 'rfd_sh', 'rtdfd_sh',
    'fantasy_points', 'fantasy_points_ppr'
]

# Main callable function
def run_pro_qb_player():
    years = list(range(2016, 2025))
    qb_data_by_year = {}
    qb_season_dict = {}

    for year in years:
        stats = import_seasonal_data([year], s_type="REG")
        rosters = import_seasonal_rosters([year])

        merged = stats.merge(
            rosters[['player_id', 'player_name', 'position', 'team']],
            on='player_id',
            how='left'
        )

        qbs = merged[merged['position'] == 'QB'].copy()
        qbs.drop(columns=['player_id'], inplace=True)
        cols = ['player_name'] + [col for col in qbs.columns if col != 'player_name']
        qbs = qbs[cols]

        qb_filtered = qbs[[col for col in qb_columns if col in qbs.columns]].copy()
        qb_filtered['player_name_clean'] = qb_filtered['player_name'].apply(clean_player_name)

        qb_data_by_year[year] = qb_filtered

        for name, group in qb_filtered.groupby('player_name_clean'):
            qb_season_dict[name] = pd.concat(
                [qb_season_dict.get(name, pd.DataFrame()), group.drop(columns=['player_name', 'player_name_clean'])],
                ignore_index=True
            )

        print(f"✅ Loaded {len(qb_filtered)} QB rows for {year}")

    return qb_season_dict
