import streamlit as st
import pandas as pd
import joblib
import os

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Bad Debt Prediction", layout="wide")

st.markdown("""
<style>

/* Page background — light blue gradient */
.stApp {
    background: linear-gradient(to bottom, #eef5ff 0%, #ffffff 60%);
}

/* Main headers */
h1, h2, h3 {
    color: #1f3a5f;
    font-weight: 600;
}

/* Sub text */
p, label {
    color: #2c2c2c !important;
    font-weight: 500 !important;
}

/* Section divider line */
hr {
    border-top: 2px solid #d6e4ff;
}

/* Buttons — professional blue */
.stButton>button {
    background-color: #2f6fed;
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 600;
    padding: 0.6rem 1rem;
}

.stButton>button:hover {
    background-color: #1f4fbf;
    color: white;
}

/* Metric result cards */
[data-testid="metric-container"] {
    background-color: #f4f8ff;
    border: 1px solid #dbe7ff;
    padding: 14px;
    border-radius: 8px;
}

/* Inputs */
.stNumberInput input {
    background-color: white !important;
    border-radius: 6px !important;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 6px !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 style='margin-bottom:0'>Bad Debt Prediction</h1>
<p style='color:gray;margin-top:0'>Credit Risk Evaluation Interface</p>
""", unsafe_allow_html=True)

st.caption("All inputs follow credit bureau risk-score logic")
st.divider()

# =====================================
# LOAD MODEL
# =====================================
BASE_DIR = os.path.dirname(__file__)
model_data = joblib.load(os.path.join(BASE_DIR, "model.joblib"))

model = model_data["model"]
features = model_data["features"]
scaler = model_data["scaler"]
cols_to_scale = model_data["cols_to_scale"]

BAD_CLASS = 1
THRESHOLD = 0.3

# =====================================
# FUNCTIONS
# =====================================
def prepare_input(user_input: dict):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df, drop_first=True)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df


def predict_risk(user_input: dict):
    df = prepare_input(user_input)
    prob_bad = float(model.predict_proba(df)[0][BAD_CLASS])
    decision = "Bad" if prob_bad >= THRESHOLD else "Good"
    return prob_bad, decision


def cr22_risk_band(score):
    if score <= 500:
        return "Very High Risk"
    elif score <= 607:
        return "High Risk"
    elif score <= 715:
        return "Medium Risk"
    else:
        return "Low Risk"


# =====================================
# NUMERIC INPUTS — CENTER + 3x3 GRID
# =====================================

st.subheader("Credit Risk Indicators")

center_col = st.columns([1,2,1])
with center_col[1]:
    SCORE_CR22 = st.number_input("Credit Score", -300, 1200, 650)

row1 = st.columns(3)
with row1[0]:
    DEROGATORIES = st.number_input("Derogatory Records", 0, 20, 0)
with row1[1]:
    CREDIT_CARD_CR22 = st.number_input("Active Cards", 0, 20, 1)
with row1[2]:
    DEFAULT_CNT_CR22 = st.number_input("Number of Defaults", 0, 20, 0)

row2 = st.columns(3)
with row2[0]:
    Late_Payment_30DPD_Last_12M = st.number_input("Late Payment 30DPD last 12M", 0, 50, 0)
with row2[1]:
    Late_Payment_30DPD_Last_24M = st.number_input("Late Payment 30 DPD last 24M", 0, 100, 0)
with row2[2]:
    DEFAULT_OPEN_CNT_CR22 = st.number_input("Number Open Defaults", 0, 20, 0)

row3 = st.columns(3)
with row3[0]:
    Credit_Card_Payment_Failure_Count = st.number_input("No of Credit Card Failures", 0, 20, 0)
with row3[1]:
    Recent_Payment_Irregularity_Flag = st.number_input("Recent Payment Irregular Flag", 0, 25, 0)
with row3[2]:
    Long_Term_Payment_Delinquency_Count = st.number_input("Total Delinquency", 0, 100, 0)

st.divider()

# =====================================
# CATEGORICAL — 2 ROWS × 4 COLUMNS
# =====================================

st.subheader("Applicant Profile Attributes")

cat_row1 = st.columns(4)
with cat_row1[0]:
    RESIDENTIAL = st.selectbox("Residential", ["Owned","Rented","Living_With_Family","Missing"])
with cat_row1[1]:
    CD_OCCUPATION = st.selectbox("Occupation", ["employed","self_employed","student","retired","unemployed","Missing"])
with cat_row1[2]:
    DOC_TYPE = st.selectbox("Doc Type", ["AU Passport","AU Driver Licence","Missing"])
with cat_row1[3]:
    EMPLOYED_STATUS = st.selectbox("Employment", ["employed","self_employed","student","retired","unemployed","benefits","Missing"])

cat_row2 = st.columns(4)
with cat_row2[0]:
    APPLICANT_AGE = st.selectbox("Age Band", ["18-24","25 - 29","30-34","35-44","45-54","54+"])
with cat_row2[1]:
    BUREAU_DEFAULT = st.selectbox("Bureau Default", ["Missing","1-1000","1000+"])
with cat_row2[2]:
    SCORECARD = st.selectbox("Scorecard", ["TAR1A","SFJR1A","HSHSOL","CTSDP","INSLV"])
with cat_row2[3]:
    BUREAU_ENQUIRIES_12_MONTHS = st.selectbox("Enquiries", ["1-2","3","4-5","6-7","8-11","12+","14+"])


st.divider()

# =====================================
# BUTTON
# =====================================

btn = st.columns([2,2,2])
predict_click = btn[1].button("Run Risk Prediction", use_container_width=True)

# =====================================
# RESULTS
# =====================================

if predict_click:

    payload = {
        "SCORE_CR22": SCORE_CR22,
        "DEROGATORIES": DEROGATORIES,
        "Late_Payment_30DPD_Last_12M": Late_Payment_30DPD_Last_12M,
        "Credit_Card_Payment_Failure_Count": Credit_Card_Payment_Failure_Count,
        "Recent_Payment_Irregularity_Flag": Recent_Payment_Irregularity_Flag,
        "Late_Payment_30DPD_Last_24M": Late_Payment_30DPD_Last_24M,
        "Long_Term_Payment_Delinquency_Count": Long_Term_Payment_Delinquency_Count,
        "CREDIT_CARD_CR22": CREDIT_CARD_CR22,
        "DEFAULT_CNT_CR22": DEFAULT_CNT_CR22,
        "DEFAULT_OPEN_CNT_CR22": DEFAULT_OPEN_CNT_CR22,
        "RESIDENTIAL": RESIDENTIAL,
        "CD_OCCUPATION": CD_OCCUPATION,
        "DOC_TYPE": DOC_TYPE,
        "EMPLOYED_STATUS": EMPLOYED_STATUS,
        "APPLICANT_AGE": APPLICANT_AGE,
        "BUREAU_DEFAULT": BUREAU_DEFAULT,
        "SCORECARD": SCORECARD,
        "BUREAU_ENQUIRIES_12_MONTHS": BUREAU_ENQUIRIES_12_MONTHS
    }

    prob_bad, decision = predict_risk(payload)

    st.subheader("Prediction Result")

    r1, r2 = st.columns(2)
    r1.metric("Decision", decision)

    st.subheader("Score Interpretation")
    st.write("Credit Score Band:", cr22_risk_band(SCORE_CR22))

