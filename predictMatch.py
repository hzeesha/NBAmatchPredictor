import joblib
import pandas as pd

# Load final data and model
full = pd.read_csv("final_data.csv")

model = joblib.load("ridge_model.pkl")
predictors = joblib.load("predictors.pkl")

# Helper functions
def get_latest_team_features(full_df, team_abbrev):
    """
    Return the most recent row for `team_abbrev` from 'full_df'.
    Must contain the columns your model expects (including rolling).
    """
    team_data = full_df[full_df["team"] == team_abbrev].copy()
    if team_data.empty:
        return None  # No data found for this team

    # Sort so the newest game is first
    team_data = team_data.sort_values("date", ascending=False)
    # Return the row as a Series
    return team_data.iloc[0]


def create_matchup_row(full_df, team_abbrev, opp_abbrev, home=1):
    row_team = get_latest_team_features(full_df, team_abbrev)
    row_opp = get_latest_team_features(full_df, opp_abbrev)

    if row_team is None or row_opp is None:
        # Not enough data for at least one team
        return None

    rename_dict = {}
    for col in row_opp.index:
        if not col.endswith("_opp"):
            rename_dict[col] = col + "_opp"
        else:
            # If it already ends with "_opp", keep it as is (or rename further, your choice)
            rename_dict[col] = col

    row_opp_renamed = row_opp.rename(rename_dict)

    combined = pd.concat([row_team, row_opp_renamed])

    duplicates = combined.index[combined.index.duplicated()].unique()
    if len(duplicates) > 0:
        # We'll keep the first occurrence, drop subsequent duplicates
        combined = combined[~combined.index.duplicated(keep='first')]

    combined["home"] = home
    combined["home_opp"] = 1 - home

    if "team_opp" not in combined.index:
        combined["team_opp"] = opp_abbrev


    combined["prediction_date"] = pd.Timestamp.today()

    matchup_df = pd.DataFrame([combined])
    return matchup_df


def predict_winner(team1, team2, home_team=1):
    matchup_df = create_matchup_row(full, team1, team2, home=home_team)
    if matchup_df is None:
        print(f"Not enough data to build a matchup row for {team1} vs {team2}.")
        return

    # Make sure matchup_df has all the predictor columns; fill missing with 0
    missing_cols = [col for col in predictors if col not in matchup_df.columns]
    for col in missing_cols:
        matchup_df[col] = 0

    matchup_df = matchup_df[predictors]

    # Predict (assuming binary target: 1=win, 0=loss)
    pred = model.predict(matchup_df)[0]
    if pred == 1:
        print(f"The model predicts {team1} will WIN over {team2}.")
    else:
        print(f"The model predicts {team1} will LOSE to {team2}.")


# Main CLI
def main():
    # Prompt user for the teams
    print("Enter team abbreviations as they appear in your data (e.g. 'LAL', 'BOS', 'GSW'):")
    team1 = input("Enter first team abbreviation: ")
    team2 = input("Enter second team abbreviation: ")

    # Decide which is home
    home_team_input = input("Enter '1' if the first team is home, or '0' if the second team is home: ")
    try:
        home_team = int(home_team_input)
    except ValueError:
        home_team = 1  # fallback default if user typed something invalid

    # Make the prediction
    predict_winner(team1, team2, home_team)

if __name__ == "__main__":
    main()
