"""
app.py - Heart Disease Risk Predictor
Streamlit web app for interactive heart disease risk prediction.
Loads combined data from 4 UCI heart disease datasets (918 samples) and trains model on first run."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from io import StringIO
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

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]
