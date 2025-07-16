#Example usage
#from ktc_scraper import load_ktc_dynasty_rankings
#ktc_df = load_ktc_dynasty_rankings(save_csv=True)


import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

def clean_ktc_name(name):
    # Add your suffix/punctuation cleanup here if needed
    return ' '.join(name.strip().split())

def load_ktc_dynasty_rankings(save_csv=False):
    """
    Scrapes KeepTradeCut Dynasty Rankings and returns a DataFrame.
    Optionally saves to CSV.
    """
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        url = "https://keeptradecut.com/dynasty-rankings"
        driver.get(url)
        time.sleep(5)

        for _ in range(10):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        player_cards = driver.find_elements(By.CLASS_NAME, 'rankings-page__list-item')
        records = []

        for card in player_cards:
            try:
                name = clean_ktc_name(card.find_element(By.CLASS_NAME, 'rankings-page__name').text)
                pos_team = card.find_element(By.CLASS_NAME, 'rankings-page__position-team').text
                value = card.find_element(By.CLASS_NAME, 'rankings-page__value').text
                rank = card.find_element(By.CLASS_NAME, 'rankings-page__rank').text

                records.append({
                    'player': name,
                    'pos_team': pos_team,
                    'ktc_value': value,
                    'ktc_rank': rank
                })

            except Exception:
                continue

        ktc_df = pd.DataFrame(records)

        if save_csv:
            ktc_df.to_csv("ktc_dynasty_rankings.csv", index=False)
            print("✅ Saved to ktc_dynasty_rankings.csv")

        return ktc_df

    finally:
        driver.quit()
