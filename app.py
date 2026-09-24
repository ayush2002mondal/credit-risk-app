import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Minimal / aesthetic styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117;
        }
        .main .block-container {
            max-width: 640px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        h1 {
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 0.1rem;
        }
        .subtitle {
            color: #9ca3af;
            font-size: 0.95rem;
            margin-bottom: 2.2rem;
        }
        .section-label {
            color: #9ca3af;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 1.4rem 0 0.4rem 0;
            font-weight: 600;
        }
        div[data-testid="stButton"] button {
            width: 100%;
            border-radius: 10px;
            padding: 0.6rem 0;
            font-weight: 600;
            border: 1px solid #2d323c;
            background-color: #161a23;
            transition: 0.15s ease;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #6366f1;
            color: #6366f1;
        }
        .result-card {
            margin-top: 1.6rem;
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            border: 1px solid;
        }
        .result-good {
            background-color: rgba(34, 197, 94, 0.08);
            border-color: rgba(34, 197, 94, 0.4);
            color: #4ade80;
        }
        .result-bad {
            background-color: rgba(239, 68, 68, 0.08);
            border-color: rgba(239, 68, 68, 0.4);
            color: #f87171;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load model + encoders
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("extra_trees_credit_model.pkl")
    encoders = {
        col: joblib.load(f"{col}_encoder.pkl")
        for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
    }
    target_encoder = joblib.load("target_encoder.pkl")
    return model, encoders, target_encoder

try:
    model, encoders, target_encoder = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing file: **{e.filename}**.\n\n"
        "Make sure the model + encoder `.pkl` files are in the same folder as `app.py`."
    )
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("# 💳 Credit Risk Predictor")
st.markdown(
    '<div class="subtitle">Enter applicant details to estimate whether the credit is a good or bad risk.</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
st.markdown('<div class="section-label">Applicant</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
with c2:
    sex = st.selectbox("Sex", ["male", "female"])

st.markdown('<div class="section-label">Background</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
with c4:
    housing = st.selectbox("Housing", ["own", "rent", "free"])

st.markdown('<div class="section-label">Accounts</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    saving_accounts = st.selectbox(
        "Saving accounts", ["little", "moderate", "quite rich", "rich"]
    )
with c6:
    checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])

st.markdown('<div class="section-label">Loan</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7:
    credit_amount = st.number_input(
        "Credit amount", min_value=0, value=1000, step=100
    )
with c8:
    duration = st.number_input("Duration (months)", min_value=1, value=12)

# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
st.write("")
if st.button("Predict Risk"):
    input_df = pd.DataFrame(
        {
            "Age": [age],
            "Sex": encoders["Sex"].transform([sex]),
            "Job": [job],
            "Housing": encoders["Housing"].transform([housing]),
            "Saving accounts": encoders["Saving accounts"].transform([saving_accounts]),
            "Checking account": encoders["Checking account"].transform([checking_account]),
            "Credit amount": [credit_amount],
            "Duration": [duration],
        }
    )

    prediction = model.predict(input_df)[0]
    label = target_encoder.inverse_transform([prediction])[0]  # "good" or "bad"

    if label == "good":
        st.markdown(
            '<div class="result-card result-good">✅ Predicted credit risk: GOOD</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-card result-bad">⚠️ Predicted credit risk: BAD</div>',
            unsafe_allow_html=True,
        )
