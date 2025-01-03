import os
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

SCORE_DIR = "data2/scores"
box_scores = os.listdir(SCORE_DIR)
box_scores = [os.path.join(SCORE_DIR, f) for f in box_scores if f.endswith(".html")]


def parse_html(box_score):
    with open(box_score, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, features="html.parser")
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup


def read_line_score(soup):
    html_string = str(soup)
    try:
        line_score = pd.read_html(StringIO(html_string), attrs={"id": "line_score"})[0]
        cols = list(line_score.columns)
        cols[0] = "team"
        cols[-1] = "total"
        line_score.columns = cols
        line_score = line_score[["team", "total"]]
        return line_score
    except (ValueError, IndexError) as e:
        print(f"Error reading line score: {e}")
        return None


def read_stats(soup, team, stat):
    html_string = str(soup)
    try:
        df = pd.read_html(StringIO(html_string), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[0]
        df = df.apply(pd.to_numeric, errors="coerce")
        return df
    except (ValueError, IndexError) as e:
        print(f"Error reading stats for {team} {stat}: {e}")
        return None


def read_season_info(soup):
    try:
        nav = soup.select("#bottom_nav_container")[0]
        hrefs = [a["href"] for a in nav.find_all("a")]
        season = os.path.basename(hrefs[1]).split("_")[0]
        return season
    except (IndexError, ValueError) as e:
        print(f"Error reading season info: {e}")
        return None


base_cols = None
games = []

for idx, box_score in enumerate(box_scores):
    # Print the index and box score file name
    print(f"Processing file {idx}: {box_score}")

    try:
        soup = parse_html(box_score)
        line_score = read_line_score(soup)
        if line_score is None:
            print(f"Skipping file {box_score} due to line score error.")
            continue

        teams = list(line_score["team"])
        summaries = []

        for team in teams:
            basic = read_stats(soup, team, "basic")
            advanced = read_stats(soup, team, "advanced")
            if basic is None or advanced is None:
                print(f"Skipping file {box_score} due to stats error for team {team}.")
                continue

            totals = pd.concat([basic.iloc[-1, :], advanced.iloc[-1, :]])
            totals.index = totals.index.str.lower()

            maxes = pd.concat([basic.iloc[:-1].max(), advanced.iloc[:-1].max()])
            maxes.index = maxes.index.str.lower() + "_max"

            summary = pd.concat([totals, maxes])

            if base_cols is None:
                base_cols = list(summary.index.drop_duplicates(keep="first"))
                base_cols = [b for b in base_cols if "bpm" not in b]

            summary = summary[base_cols]
            summaries.append(summary)

        summary = pd.concat(summaries, axis=1).T
        game = pd.concat([summary, line_score], axis=1)

        game["home"] = [0, 1]
        game_opp = game.iloc[::-1].reset_index()
        game_opp.columns += "_opp"
        full_game = pd.concat([game, game_opp], axis=1)

        season_info = read_season_info(soup)
        if season_info is None:
            print(f"Skipping file {box_score} due to season info error.")
            continue
        full_game["season"] = season_info

        full_game["date"] = os.path.basename(box_score)[:8]
        full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")

        full_game["won"] = full_game["total"] > full_game["total_opp"]
        games.append(full_game)

        if len(games) % 100 == 0:
            print(f"{len(games)} / {len(box_scores)}")

    except Exception as e:
        print(f"An error occurred while processing file {box_score}: {e}")
        continue

games_df = pd.concat(games, ignore_index=True)
games_df.to_csv("nba_games.csv")