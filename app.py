import joblib
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV


# ---------------- CACHING / LOAD ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("ridge_model.pkl")
    predictors = joblib.load("predictors.pkl")
    full = pd.read_csv("final_data.csv")
    return model, predictors, full


@st.cache_resource
def build_calibrator(_model, predictors, full_df: pd.DataFrame):
    # Calibrate so we can display probabilities.
    # Needs "target" in final_data.csv.
    if "target" not in full_df.columns:
        return None

    train_df = full_df[full_df["target"] != 2].copy()
    if train_df.empty:
        return None

    for col in predictors:
        if col not in train_df.columns:
            train_df[col] = 0

    X_cal = train_df[predictors]
    y_cal = train_df["target"].astype(int)

    calibrator = CalibratedClassifierCV(_model, method="sigmoid", cv=3)
    calibrator.fit(X_cal, y_cal)
    return calibrator


# ---------------- THEME ----------------
def apply_theme(dark: bool):
    if dark:
        css = """
        <style>
        header[data-testid="stHeader"] { display: none; }
        footer { display: none; }
        .block-container { padding-top: 1.2rem !important; }

        .stApp {
            background: radial-gradient(circle at 30% 10%, rgba(40,60,120,0.35), transparent 35%),
                        radial-gradient(circle at 70% 0%, rgba(0,120,255,0.18), transparent 40%),
                        #0b0f17;
            color: #e9eef7;
        }

        /* Base text */
        label, .stMarkdown, .stText, .stCaption, p, span, div {
            color: #e9eef7;
        }

        /* Selectbox (collapsed) */
        div[data-baseweb="select"] > div {
            background-color: #151b26 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 10px !important;
            color: #e9eef7 !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: #e9eef7 !important;
            -webkit-text-fill-color: #e9eef7 !important;
        }

        /* Dropdown menu */
        div[data-baseweb="popover"] * { color: #e9eef7 !important; }
        ul[role="listbox"] {
            background-color: #151b26 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }
        li[role="option"] {
            background-color: #151b26 !important;
            color: #e9eef7 !important;
        }
        li[role="option"]:hover { background-color: rgba(255,255,255,0.08) !important; }

        /* Button */
        div.stButton > button {
            background: #ff4b4b !important;
            color: white !important;
            border: 0 !important;
            border-radius: 10px !important;
            padding: 0.6rem 1rem !important;
            font-weight: 700 !important;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px;
        }
        div[data-testid="stMetricLabel"] > div { color: #cfd8e6 !important; }
        div[data-testid="stMetricValue"] { color: #ffffff !important; }

        div[data-testid="stAlert"] { border-radius: 12px; }
        .stCaption { color: #cfd8e6 !important; }
        </style>
        """
    else:
        css = """
        <style>
        header[data-testid="stHeader"] { display: none; }
        footer { display: none; }
        .block-container { padding-top: 1.2rem !important; }

        .stApp { background: #f7f8fb; color: #0b0f17; }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 14px;
            padding: 14px;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# ---------------- FEATURES ----------------
def get_latest_team_features(full_df: pd.DataFrame, team_abbrev: str):
    team_data = full_df[full_df["team"] == team_abbrev].copy()
    if team_data.empty:
        return None
    team_data = team_data.sort_values("date", ascending=False)
    return team_data.iloc[0]


def create_matchup_row(full_df: pd.DataFrame, team_abbrev: str, opp_abbrev: str, home: int):
    row_team = get_latest_team_features(full_df, team_abbrev)
    row_opp = get_latest_team_features(full_df, opp_abbrev)
    if row_team is None or row_opp is None:
        return None

    # Rename opponent columns to *_opp to match training
    rename_dict = {}
    for col in row_opp.index:
        col_str = str(col)
        rename_dict[col_str] = col_str if col_str.endswith("_opp") else (col_str + "_opp")
    row_opp = row_opp.rename(rename_dict)

    combined = pd.concat([row_team, row_opp])

    # Remove duplicate indices if any
    dupes = combined.index[combined.index.duplicated()].unique()
    if len(dupes) > 0:
        combined = combined[~combined.index.duplicated(keep="first")]

    combined["home"] = int(home)
    combined["home_opp"] = 1 - int(home)

    # Metadata (not used unless it ends up in predictors)
    combined["team_opp"] = opp_abbrev
    combined["prediction_date"] = pd.Timestamp.today()

    return pd.DataFrame([combined])


def predict(model, calibrator, predictors, full_df, team1, team2, home_team1: bool):
    home = 1 if home_team1 else 0
    matchup_df = create_matchup_row(full_df, team1, team2, home=home)
    if matchup_df is None:
        return None, "Couldn't build a matchup row (missing team data)."

    for col in predictors:
        if col not in matchup_df.columns:
            matchup_df[col] = 0

    X = matchup_df[predictors]

    pred = int(model.predict(X)[0])

    score = None
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])

    prob_team1_win = None
    if calibrator is not None and hasattr(calibrator, "predict_proba"):
        prob_team1_win = float(calibrator.predict_proba(X)[0][1])

    winner = team1 if pred == 1 else team2
    loser = team2 if pred == 1 else team1
    return (winner, loser, score, prob_team1_win), None


# ---------------- UI ----------------
st.set_page_config(
    page_title="NBA Match Predictor",
    page_icon="🏀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Header row (simple + human wording)
left, right = st.columns([0.78, 0.22], vertical_alignment="center")
with left:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="font-size:34px;">🏀</div>
            <div style="font-size:44px; font-weight:900; letter-spacing:0.5px;">NBA Match Predictor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.session_state.dark_mode = st.toggle("Dark", value=st.session_state.dark_mode)

apply_theme(st.session_state.dark_mode)

model, predictors, full = load_artifacts()
calibrator = build_calibrator(model, predictors, full)

teams = sorted(full["team"].dropna().unique().tolist())

# Short, natural disclaimer (no ML label talk)
st.caption(
    "DISCLAIMER: this uses historical results + recent game stats through the 2023–24 season."
)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams, index=teams.index("LAL") if "LAL" in teams else 0)
with col2:
    team2 = st.selectbox("Team 2", teams, index=teams.index("GSW") if "GSW" in teams else 1)

home_team1 = st.toggle("Team 1 is home", value=True)

if st.button("Predict", type="primary"):
    if team1 == team2:
        st.error("Pick two different teams.")
    else:
        result, err = predict(model, calibrator, predictors, full, team1, team2, home_team1)
        if err:
            st.error(err)
        else:
            winner, loser, score, prob = result

            st.success(f"Prediction: **{winner}** over **{loser}**")

            if prob is not None:
                st.metric(label=f"{team1} win chance", value=f"{prob * 100:.1f}%")
                st.metric(label=f"{team2} win chance", value=f"{(1 - prob) * 100:.1f}%")
            else:
                st.warning("Win % isn’t available because the probability calibrator couldn’t be built.")

            if score is not None:
                st.info(f"Model score: **{score:.3f}** (closer to 0 = closer matchup)")
