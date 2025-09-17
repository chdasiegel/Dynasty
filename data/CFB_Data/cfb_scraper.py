#Example Usage
#from cfb_scraper import run_cfb_scraper
#run_cfb_scraper()

import requests
import pandas as pd
import time

def run_cfb_scraper():
    years = list(range(2016, 2025))  # change end bound to upcoming season
    stats = ["passing", "rushing", "receiving"]

    for year in years:
        for stat in stats:
            url = f"https://www.sports-reference.com/cfb/years/{year}-{stat}.html"
            try:
                response = requests.get(url)
                response.raise_for_status()
                dfs = pd.read_html(response.text)
                if dfs:
                    df = dfs[0]
                    df.to_csv(f"{year}_{stat}.csv", index=False)
                    print(f"Saved {year}_{stat}.csv")
                else:
                    print(f"No tables found for {url}")
            except Exception as e:
                print(f"Failed for {url}: {e}")
            time.sleep(1)  # be polite to the server
