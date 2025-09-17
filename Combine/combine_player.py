#Example Usage
#from combine_player import run_combine_player
#combine_dict = run_combine_player()

import pandas as pd
import os
import re
from utils import clean_player_name, height_to_inches

pd.set_option('display.max_columns', None)


# Main function to process combine files
def run_combine_player():
    years = list(range(2016, 2026))
    combine_records = []

    for year in years:
        file_name = f"{year}_Combine.csv"
        if os.path.exists(file_name):
            try:
                df = pd.read_csv(
                    file_name,
                    encoding='latin1',
                    dtype=str,
                    parse_dates=False
                )

                if 'Player' not in df.columns:
                    print(f"❌ 'Player' column missing in {file_name}, skipping.")
                    continue

                df.rename(columns={'Player': 'player'}, inplace=True)
                df['Year'] = str(year)

                if 'Yr' in df.columns:
                    df.drop(columns=['Yr'], inplace=True)

                if 'Drafted (tm/rnd/yr)' in df.columns:
                    draft_split = df['Drafted (tm/rnd/yr)'].str.extract(
                        r'^(.*?)\s*/\s*(\d+(?:st|nd|rd|th))\s*/\s*(\d+(?:st|nd|rd|th) pick)'
                    )
                    df['Draft_Team'] = draft_split[0].str.strip()
                    df['Draft_Round'] = draft_split[1].str.strip()
                    df['Draft_Pick'] = draft_split[2].str.strip()
                    df.drop(columns=['Drafted (tm/rnd/yr)'], inplace=True)

                if 'Pos' in df.columns:
                    df = df[df['Pos'].isin(['QB', 'RB', 'WR', 'TE'])]

                if 'Ht' in df.columns:
                    df['Ht'] = df['Ht'].apply(height_to_inches)

                df['player'] = df['player'].apply(clean_player_name)
                df['player'] = df['player'].str.replace('*', '', regex=False)

                combine_records.append(df)
                print(f"✅ Processed {file_name} with {len(df)} rows.")
            except Exception as e:
                print(f"❌ Error reading {file_name}: {e}")
        else:
            print(f"🚫 File not found: {file_name}")

    if combine_records:
        combined_df = pd.concat(combine_records, ignore_index=True)

        # Move 'Year' after 'player'
        cols = list(combined_df.columns)
        if 'Year' in cols:
            cols.remove('Year')
        if 'player' in cols:
            player_index = cols.index('player')
            cols = cols[:player_index + 1] + ['Year'] + cols[player_index + 1:]
            combined_df = combined_df[cols]

        player_combine_dict = {
            name: group.reset_index(drop=True)
            for name, group in combined_df.groupby('player')
        }

        print(f"✅ Created player_combine_dict with {len(player_combine_dict)} players.")
        return player_combine_dict
    else:
        print("❌ No combine data found.")
        return {}
