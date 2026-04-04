"""
app.py - Heart Disease Risk Predictor
Streamlit web app for interactive heart disease risk prediction.
Loads data directly from UCI ML Repository and trains model on first run.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ──────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]


# ── Data & Model (cached) ─────────────────────────────────
@st.cache_data
def load_data():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(
        StringIO(response.text),
        header=None, names=COLUMNS, na_values="?"
    )
    df = df.dropna().copy()
    df["target"] = (df["target"] > 0).astype(int)
    return df


@st.cache_resource
def train_model(model_name):
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "auc":      round(roc_auc_score(y_test, y_prob), 4),
        "n_train":  len(X_train),
        "n_test":   len(X_test),
        "features": X.columns.tolist()
    }
    return model, scaler, metrics


# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("❤️ Heart Disease Risk")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "Logistic Regression"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Enter Patient Details")

age      = st.sidebar.slider("Age", 20, 80, 50)
sex      = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp       = st.sidebar.selectbox(
    "Chest Pain Type",
    ["0 - Typical Angina", "1 - Atypical Angina",
     "2 - Non-anginal Pain", "3 - Asymptomatic"]
)
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol     = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg  = st.sidebar.selectbox(
    "Resting ECG",
    ["0 - Normal", "1 - ST-T Abnormality", "2 - LV Hypertrophy"]
)
thalach  = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang    = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak  = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
slope    = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["0 - Upsloping", "1 - Flat", "2 - Downsloping"]
)
ca       = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal     = st.sidebar.selectbox(
    "Thalassemia",
    ["1 - Normal", "2 - Fixed Defect", "3 - Reversible Defect"]
)


# ── Build input vector ──────────────────────────────────────
def encode(val, options):
    return options.index(val)

input_data = np.array([[
    age,
    1 if sex == "Male" else 0,
    int(cp[0]),
    trestbps,
    chol,
    1 if fbs == "Yes" else 0,
    int(restecg[0]),
    thalach,
    1 if exang == "Yes" else 0,
    oldpeak,
    int(slope[0]),
    ca,
    int(thal[0])
]])


# ── Main content ────────────────────────────────────────
st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    "An end-to-end ML app using the **UCI Cleveland Heart Disease Dataset** "
    "(303 patients). Adjust patient parameters in the sidebar and click **Predict**."
)
st.markdown("---")

with st.spinner(f"Loading data and training {model_choice}..."):
    model, scaler, metrics = train_model(model_choice)

# Model metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Model", model_choice.split()[0])
col2.metric("📊 Accuracy", f"{metrics['accuracy']}%")
col3.metric("📈 AUC-ROC", str(metrics['auc']))
col4.metric("🧭 Training Samples", metrics['n_train'])

st.markdown("---")

# Predict button
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.markdown("## Prediction Result")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Heart Disease**")
            st.markdown(f"**Risk Probability: {probability*100:.1f}%**")
        else:
            st.success(f"✅ **Low Risk of Heart Disease**")
            st.markdown(f"**Risk Probability: {probability*100:.1f}%**")

    with res_col2:
        # Risk gauge using progress bar
        st.markdown("**Risk Level**")
        st.progress(float(probability))
        risk_label = "High" if probability >= 0.5 else "Low"
        st.markdown(f"Confidence: **{max(probability, 1-probability)*100:.1f}%** ({risk_label} risk)")

    st.markdown("---")
    st.markdown("### Patient Summary")
    summary = pd.DataFrame({
        "Feature": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
                    "Fasting BS", "Resting ECG", "Max HR", "Exercise Angina",
                    "ST Depression", "ST Slope", "Major Vessels", "Thalassemia"],
        "Value":   [age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    st.info("← Set patient parameters in the sidebar, then click **Predict Risk**.")

# Dataset preview
st.markdown("---")
with st.expander("📊 View Dataset Sample"):
    df = load_data()
    st.write(f"**{len(df)} patients | {df.shape[1]} features**")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(f"**Target distribution:** {df['target'].value_counts().to_dict()}")

# Footer
st.markdown("---")
st.markdown(
    "Built by **Rahul Roy** — MS Applied Machine Intelligence @ Northeastern University | "
    "[GitHub](https://github.com/rahul-roy56) | [LinkedIn](https://www.linkedin.com/in/rahulroy0499/)"
)
