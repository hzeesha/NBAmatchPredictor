# NBAmatchPredictor

## NBA game outcome predictor: scrapes box scores from Basketball Reference, compiles them, adds rolling averages, trains a Ridge classifier, then offers a CLI to predict winners. 

## Features
- **Data Scraping**: Uses Playwright and BeautifulSoup to scrape data from Basketball Reference.
- **Data Parsing**: Extracts game statistics and processes them into structured CSV files.
- **Machine Learning**: Ridge Classifier trained on historical and rolling average features to predict game outcomes.
- **Prediction**: Input two teams to predict the winner based on the model.

## Installation
1. **Clone the repository**
2. **Set up the Python environment**
   - Install Python 3.8 or later
   - Install required dependencies: pip install -r requirements.txt
3. **Set up data directories**
   - Ensure that the following directory structure exists:
     data2/
     ├── standings/
     └── scores/
4. **Optional Playwright Setup**
   - If using Playwright for the first time, run: playwright install

## Usage
- Run main.py to scrape standings and game data 
- Use parsed_data.py to process the raw scraped data into structured CSV files
- Run predictlive.py to train the model
- Run predictMatch.py to predict the winner between two teams. Running predictMatch.py will allow you to input teams
- If you wish to skip the scraping, parsing, and model training you can go straight to predicting games as the required files to run the predictMatch script are given. 

## Files

### Code Files
- main.py: Scrapes game data from Basketball Reference (currently, the script scrapes NBA seasons from 2019 - 2024).
- parsed_data.py: Processes raw data into structured CSV files.
- predictlive.py: Trains the Ridge Classifier model.
- predictMatch.py: Predicts the winner of a game between two user-input teams.
  
### Data Files
- nba_games.csv: Processed game data with advanced statistics.
- final_data.csv: Final dataset used for model training and predictions.

### Model Files
- ridge_model.pkl: Trained Ridge Classifier model.
- predictors.pkl: List of feature columns used by the model.

## Notes
- The requirements.txt file includes all libraries used during development. Some may not be critical to the project but are included to ensure no missing dependencies.
- Ensure internet access for scraping data (this process may take a while).
- Use Python 3.8 or later for compatibility.

## Contributing
- Feel free to submit issues or create pull requests for improvements.
