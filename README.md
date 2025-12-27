# NBAmatchPredictor

NBAmatchPredictor looks at past NBA games, builds rolling stats, and uses a Ridge classifier to estimate which team is more likely to win a matchup.

You can use it in two ways:

- a Streamlit web app  
- local scripts (CLI)

> Data is currently up to the **2023–24 NBA season**.

---

## Web App

Use the model in the browser:

`https://nbamatchpredictor.streamlit.app`

Select:

- **Team 1**
- **Team 2**
- who’s at **home**

The app shows:

- predicted winner  
- win probability for each team  
- a raw “confidence” score from the model

---

## What it does

- Scrapes game data from Basketball Reference (optional, local only)
- Parses box scores and builds a dataset
- Adds rolling averages for recent performance
- Trains a Ridge classifier
- Uses the saved model to predict matchups

---

## Project structure

```text
data2/
 ├── standings/
 └── scores/

nba_games.csv        # processed box score data
final_data.csv       # final feature dataset
ridge_model.pkl      # trained model
predictors.pkl       # list of feature columns
app.py               # Streamlit UI
main.py              # scraping
parsed_data.py       # parsing / feature building
predictlive.py       # model training
predictMatch.py      # CLI prediction script
```

## Clone the repo 

``` bash
git clone https://github.com/yourusername/NBAmatchPredictor.git
cd NBAmatchPredictor
```

## Run things locally

## Clone the repo

``` bash
git clone https://github.com/yourusername/NBAmatchPredictor.git
cd NBAmatchPredictor
```

## Set up a virtual environment

``` bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

## Install dependencies

``` bash
pip install -r requirements.txt
```

## Optional: scrape + rebuild the dataset

if using playwright for the first time:

```bash
playwright install
```

then 

```bash
python main.py           # scrape standings + game pages
python parsed_data.py    # parse box scores into a CSV
python predictlive.py    # train the model and save ridge_model.pkl / predictors.pkl
```

## Predict via CLI

```bash
python predictMatch.py
```

## Streamlit app locally

```bash
streamlit run app.py
```

## Streamlit Cloud setup

runs with python 3.11

Minimal requirements.txt for the UI:

streamlit
pandas
numpy
joblib
scikit-learn==1.4.2

