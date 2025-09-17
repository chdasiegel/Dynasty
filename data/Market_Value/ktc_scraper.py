#Example usage
#from ktc_scraper import load_ktc_dynasty_rankings
#ktc_df = load_ktc_dynasty_rankings(save_csv=True)

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def clean_ktc_name(name):
    return ' '.join(name.strip().split())

def load_ktc_dynasty_rankings(save_csv=False):
    """
    Scrapes KeepTradeCut Dynasty Superflex Rankings.
    Requires user to manually close any popups/modals.
    """
    options = Options()
    # options.add_argument('--headless')  # Headless off for manual interaction
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        url = "https://keeptradecut.com/dynasty-rankings"
        driver.get(url)

        print("⚠️ If a popup appears (e.g. 'Your Thoughts?'), please close it manually.")
        input("⏸️ Press ENTER after closing any popups to continue scraping...")

        # Scroll several times to load full content
        for _ in range(12):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)

        # Wait until at least 10 player cards are loaded
        WebDriverWait(driver, 15).until(
            lambda d: len(d.find_elements(By.CLASS_NAME, 'rankings-page__list-item')) > 10
        )

        player_cards = driver.find_elements(By.CLASS_NAME, 'rankings-page__list-item')
        print(f"🔢 Found {len(player_cards)} player cards")

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
            except Exception as e:
                continue

        ktc_df = pd.DataFrame(records)

        if ktc_df.empty:
            print("⚠️ No data scraped. The page structure may have changed or the content didn't load fully.")
        elif save_csv:
            ktc_df.to_csv("ktc_dynasty_rankings.csv", index=False)
            print("✅ Saved to ktc_dynasty_rankings.csv")

        return ktc_df

    finally:
        driver.quit()