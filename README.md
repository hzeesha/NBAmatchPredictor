# NBAmatchPredictor

## NBA game outcome predictor: scrapes box scores from Basketball Reference, compiles them, adds rolling averages, trains a Ridge classifier, then offers a CLI to predict winners. 

## Features
- **Data Scraping**: Uses Playwright and BeautifulSoup to scrape data from Basketball Reference.
- **Data Parsing**: Extracts game statistics and processes them into structured CSV files.
- **Machine Learning**: Ridge Classifier trained on historical and rolling average features to predict game outcomes.
- **Prediction**: Input two teams to predict the winner based on the model.

