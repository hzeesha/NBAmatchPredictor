import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

########################################
# 1) LOAD & BASIC CLEANUP
########################################
df = pd.read_csv("nba_games.csv", index_col=0)
df = df.sort_values("date").reset_index(drop=True)

# Remove unneeded columns if they exist
for col in ["mp.1", "mp_opp.1", "index_opp"]:
    if col in df.columns:
        del df[col]

# Create 'target' in one shot (avoid repeated insert warnings)
df["target"] = df.groupby("team")["won"].shift(-1)
df["target"] = df["target"].fillna(2).astype(int, errors="ignore")

# Drop rows missing key stats
for col in ["ft%", "ft%_max", "+/-_max", "ft%_opp", "ft%_max_opp", "+/-_max_opp"]:
    if col in df.columns:
        df = df.dropna(subset=[col])

# (Optional) remove columns that are entirely NaN
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]  # columns with missing values
valid_columns = df.columns[~df.columns.isin(nulls.index)]
df = df[valid_columns].copy()

########################################
# 2) BASIC MODEL (No Rolling Yet)
########################################
removed_cols_basic = ["season", "date", "won", "target", "team", "team_opp"]
basic_features = df.columns[~df.columns.isin(removed_cols_basic)]

scaler = MinMaxScaler()
df[basic_features] = scaler.fit_transform(df[basic_features])

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=45, direction="forward", cv=split)

sfs.fit(df[basic_features], df["target"])
predictors_basic = list(basic_features[sfs.get_support()])
print("Basic model predictors:", predictors_basic)

def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []
    seasons = sorted(data["season"].unique())
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        all_predictions.append(combined)
    return pd.concat(all_predictions)

preds_basic = backtest(df, rr, predictors_basic)
preds_basic = preds_basic[preds_basic["actual"] != 2]
basic_acc = accuracy_score(preds_basic["actual"], preds_basic["prediction"])
print("Basic model accuracy:", basic_acc)

########################################
# 3) ADD ROLLING FEATURES
########################################
desired_columns = [
    "won", "team", "season", "ftr", "trb%", "usg%", "fg%_max",
    "3pa_max", "orb_max", "pf_max", "orb%_max", "stl%_max",
    "3pa_opp", "orb_opp", "stl_opp", "trb%_opp", "ast%_opp",
    "stl%_opp", "usg%_opp", "orb_max_opp", "pf_max_opp"
]

missing_cols = [c for c in desired_columns if c not in df.columns]
if missing_cols:
    print("Missing columns for rolling:", missing_cols)
else:
    df_rolling = df[desired_columns].copy()

    def find_team_averages(group):
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        return group[numeric_cols].rolling(20).mean()

    non_numeric_cols = df_rolling.select_dtypes(exclude=[np.number]).columns
    df_numeric_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)
    df_numeric_rolling = df_numeric_rolling.reset_index(drop=True)

    df_rolling = pd.concat([
        df_rolling[non_numeric_cols].reset_index(drop=True),
        df_numeric_rolling
    ], axis=1)

    # Rename columns to indicate they're rolling
    rolling_cols = [f"{col}_5" for col in df_rolling.columns]
    df_rolling.columns = rolling_cols

    # Concat once
    df = pd.concat([df, df_rolling], axis=1)

df = df.dropna()

########################################
# 4) SHIFT & MERGE OPPONENT STATS
########################################
def shift_col(subdf, col):
    return subdf[col].shift(-1)

# Collect next-game columns in a dict
next_cols = {}
next_cols["home_next"] = df.groupby("team", group_keys=False).apply(lambda g: shift_col(g, "home"))
next_cols["team_opp_next"] = df.groupby("team", group_keys=False).apply(lambda g: shift_col(g, "team_opp"))
next_cols["date_next"] = df.groupby("team", group_keys=False).apply(lambda g: shift_col(g, "date"))

next_cols_df = pd.DataFrame(next_cols)

# One concat to reduce fragmentation
df = pd.concat([df, next_cols_df], axis=1).copy()

# Now do the merge to attach the opponent's rolling stats
all_opp_merge_cols = rolling_cols + ["team_opp_next", "date_next", "team"]
all_opp_merge_cols = [c for c in all_opp_merge_cols if c in df.columns]

# NOTE: We specify suffixes=('', '_opp') so that the left side keeps "team"
final_data = df.merge(
    df[all_opp_merge_cols],
    left_on=["team", "date_next"],
    right_on=["team_opp_next", "date_next"],
    how="left",
    suffixes=('', '_opp')
)

########################################
# 5) DROP ANY REMAINING NaNs BEFORE SFS
########################################
final_data = final_data.dropna().copy()

########################################
# 6) FINAL MODEL TRAINING
########################################
removed_cols_final = ["season", "date", "won", "target", "team", "team_opp"]
# Also remove any object columns
removed_cols_final += list(final_data.select_dtypes(include=["object"]).columns)

final_features = final_data.columns[~final_data.columns.isin(removed_cols_final)]

rr_final = RidgeClassifier(alpha=1)
sfs_final = SequentialFeatureSelector(rr_final, n_features_to_select=45, direction="forward", cv=split)

sfs_final.fit(final_data[final_features], final_data["target"])
predictors_final = list(final_features[sfs_final.get_support()])

predictions_final = backtest(final_data, rr_final, predictors_final)
predictions_final = predictions_final[predictions_final["actual"] != 2]
final_acc = accuracy_score(predictions_final["actual"], predictions_final["prediction"])
print("Final model accuracy:", final_acc)
print("Final predictors:", predictors_final)

########################################
# 7) SAVE THE FINAL DF & MODEL
########################################
# Save to final_data.csv
final_data.to_csv("final_data.csv", index=False)
print("Saved final DataFrame to 'final_data.csv' with shape:", final_data.shape)

joblib.dump(rr_final, "ridge_model.pkl")
joblib.dump(predictors_final, "predictors.pkl")
print("Model & predictors saved successfully.")
