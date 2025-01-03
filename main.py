import sys
import asyncio
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import time
import os

SEASONS = list(range(2019, 2025))

DATA_DIR = "data2"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings") # Creating a path from data2 folder to standings folder inside data2
SCORES_DIR = os.path.join(DATA_DIR, "scores")

def get_html(url, selector, sleep=4, retries=5):
    html = None
    for i in range(1, retries+1): # Looping for a select number of retries in case of error
        time.sleep(sleep * i) # Sleeping (waiting) in case the server has banned us temporarily
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch() # essentially means to launch the browser and to wait until it's done launching
                page = browser.new_page() # new tab
                page.goto(url)
                print(page.title())
                html = page.inner_html(selector) # selector only grabs a piece of the html
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue # if the scraping fails we will go to the top of the loop and try again
        else: # else block will run if everything was successful
            break
    return html # if 3 retries fail none will be returned. Otherwise the html of the page will be returned. Webscraping can be unreliable, this is why we have retries

def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"

    # Function to run the async function from the synchronous context
    html = get_html(url, "#content .filter")

    # Execute the async function
    # print(html)

    soup = BeautifulSoup(html, features="html.parser")
    links = soup.find_all("a")
    href = [l["href"] for l in links]
    standings_pages = [f"https://basketball-reference.com{l}" for l in href]

    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = get_html(url, "#all_schedule")
        with open(save_path, "w+") as f:
            f.write(html)


#for season in SEASONS:
#    scrape_season(season)

standings_files = os.listdir(STANDINGS_DIR)

#saving all of the box scores within a single month of a single season. The single months are saved in the standings file
def scrape_game(standings_file):
    with open(standings_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html,  features="html.parser")
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]
    box_scores = [l for l in hrefs if l and "boxscore" in l and "html" in l]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = get_html(url, "#content")
        if not html:
            continue
        with open(save_path, "w+", encoding='utf-8') as f:
            f.write(html)

standings_files = [s for s in standings_files if ".html in s"]

for f in standings_files:
    filepath = os.path.join(STANDINGS_DIR, f)
    scrape_game(filepath)