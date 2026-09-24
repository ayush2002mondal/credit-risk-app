# 💳 Credit Risk Predictor

A minimal Streamlit app that predicts whether a credit applicant is **good** or **bad** risk, based on the [German Credit Risk dataset](https://www.kaggle.com/datasets/uciml/german-credit).

**Live demo:**[ _add your Streamlit Cloud link here once deployed_
](https://credit-risk-app-aqotsvu47t9e3224ndiuoc.streamlit.app/)
## How it works

- `credit_risk_modeling.ipynb` — data analysis, feature engineering, and model training (Decision Tree, Random Forest, Extra Trees, XGBoost compared via `GridSearchCV`)
- `app.py` — Streamlit interface that loads the trained model and encoders to serve live predictions
- Inputs: age, sex, job category, housing, saving accounts, checking account, credit amount, duration
- Output: predicted risk (**good** / **bad**)

## Run it locally

```bash
git clone https://github.com/ayush2002mondal/credit-risk-app.git
cd credit-risk-app

conda create -n credit-risk python=3.10
conda activate credit-risk

pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project structure

```
├── app.py                              # Streamlit app
├── credit_risk_modeling.ipynb          # Model training notebook
├── requirements.txt                    # Python dependencies
├── extra_trees_credit_model.pkl        # Trained model
├── Sex_encoder.pkl                     # Label encoders
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
└── target_encoder.pkl
```

## Tech stack

Python · pandas · scikit-learn · XGBoost · Streamlit
