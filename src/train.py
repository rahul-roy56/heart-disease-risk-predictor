"""
train.py - Heart Disease Risk Predictor
Trains Logistic Regression and Random Forest models on the UCI Cleveland dataset.
Saves the best model and scaler to disk using joblib.
"""

import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO

import requests
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load Data ──────────────────────────────────────────
def load_data():
    print("Fetching dataset from UCI ML Repository...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(
        StringIO(response.text),
        header=None,
        names=COLUMNS,
        na_values="?"
    )
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


# ── 2. Preprocess ─────────────────────────────────────────
def preprocess(df):
    # Drop rows with missing values
    df = df.dropna().copy()

    # Binarise target: 0=no disease, 1=disease
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


# ── 3. EDA plots ──────────────────────────────────────────
def plot_eda(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Target distribution
    df["target_label"] = df["target"].map({0: "No Disease", 1: "Disease"})
    df["target_label"].value_counts().plot(
        kind="bar", ax=axes[0], color=["#4CAF50", "#F44336"], edgecolor="white"
    )
    axes[0].set_title("Target Distribution")
    axes[0].set_xlabel("")
    axes[0].tick_params(rotation=0)

    # Age distribution by target
    for label, grp in df.groupby("target_label"):
        axes[1].hist(grp["age"], bins=15, alpha=0.6, label=label)
    axes[1].set_title("Age Distribution by Target")
    axes[1].set_xlabel("Age")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "eda_plots.png"), dpi=120)
    plt.close()
    print("EDA plots saved to models/eda_plots.png")


# ── 4. Train & Evaluate ───────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, scaler):
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42)
    }

    best_model, best_auc = None, 0.0
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cv  = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc").mean()

        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}")
        print(f"  CV AUC   : {cv:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

        if auc > best_auc:
            best_auc   = auc
            best_model = model
            best_name  = name

    print(f"\nBest model: {best_name} (AUC={best_auc:.4f})")
    return best_model


# ── 5. Feature Importance plot ────────────────────────────
def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title("Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    print("Feature importance plot saved to models/feature_importance.png")


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    plot_eda(df)

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    best_model = train_and_evaluate(X_train, X_test, y_train, y_test, scaler)
    plot_feature_importance(best_model, X.columns.tolist())

    # Save artifacts
    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "features.pkl"))
    print("\nModel, scaler and feature list saved to models/")
