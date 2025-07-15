#Example Usage
#from cfb_player import run_cfb_player
#college_dict = run_cfb_player()

import pandas as pd
import os
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

# Main function to run data cleaning and dictionary creation
def run_cfb_player():
    years = list(range(2016, 2025))
    stats = ["passing", "rushing", "receiving"]
    all_records = []

    for year in years:
        for stat in stats:
            file_name = f"{year}_{stat}.csv"
            if os.path.exists(file_name):
                df = pd.read_csv(file_name)

                if 'Player' in df.columns:
                    df.rename(columns={'Player': 'player'}, inplace=True)

                if 'Awards' in df.columns:
                    df.drop(columns=['Awards'], inplace=True)

                rename_dict = {
                    'Yds.2': 'Scrim_Yds',
                    'Avg': 'Scrim_Avg',
                    'TD.2': 'Tot_TD'
                }
                df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)

                if 'player' not in df.columns or 'Rk' not in df.columns:
                    print(f"❌ 'player' or 'Rk' column missing in {file_name}, skipping this file.")
                    continue

                df['Rk'] = pd.to_numeric(df['Rk'], errors='coerce')
                df['season'] = year
                df['stat_type'] = stat
                all_records.append(df)

    if all_records:
        combined_df = pd.concat(all_records, ignore_index=True)

        combined_df['player'] = combined_df['player'].str.replace('*', '', regex=False)
        combined_df['player'] = combined_df['player'].apply(clean_player_name)

        combined_df = combined_df.dropna(subset=['Rk'])

        idx = combined_df.groupby(['player', 'season'])['Rk'].idxmin()
        filtered_df = combined_df.loc[idx].copy()

        cols = list(filtered_df.columns)
        if 'season' in cols:
            cols.remove('season')
        player_idx = cols.index('player')
        cols = cols[:player_idx + 1] + ['season'] + cols[player_idx + 1:]
        filtered_df = filtered_df[cols]

        if 'Rk' in filtered_df.columns:
            filtered_df.drop(columns=['Rk'], inplace=True)

        player_college_dict = {
            name: group.reset_index(drop=True)
            for name, group in filtered_df.groupby('player')
        }

        print(f"✅ Created dictionary for {len(player_college_dict)} players (lowest Rk per season, all seasons included).")
        return player_college_dict
    else:
        print("❌ No data found. Please check your CSV files.")
        return {}
