"""
train.py - Heart Disease Risk Predictor
Trains Logistic Regression and Random Forest models on the UCI combined dataset.
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
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

def load_data():
    """
    Load and combine heart disease data from UCI repository.
    Combines Cleveland, Hungary, Switzerland, and VA Long Beach datasets.
    Total: 918 samples
    """
    print("Loading UCI combined heart disease dataset...")
    
    # URLs for the four UCI datasets
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    datasets = {
        'cleveland': 'processed.cleveland.data',
        'hungarian': 'processed.hungarian.data',
        'switzerland': 'processed.switzerland.data',
        'va': 'processed.va.data'
    }
    
    # Column names
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    all_data = []
    
    for name, filename in datasets.items():
        url = base_url + filename
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text), names=columns, na_values='?')
            print(f"Loaded {name}: {len(df)} samples")
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {name} dataset: {e}")
    
    # Combine all datasets
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal samples: {len(df)}")
    
    return df

def preprocess_data(df):
    """
    Preprocess the dataset:
    - Handle missing values
    - Convert target to binary (0: no disease, 1: disease)
    - Select features
    """
    print("\nPreprocessing data...")
    
    # Convert target to binary (0 = no disease, 1-4 = disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Select key features
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Keep only selected features and target
    df = df[features + ['target']]
    
    # Handle missing values
    print(f"Missing values before: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Missing values after: {df.isnull().sum().sum()}")
    print(f"Remaining samples: {len(df)}")
    
    # Split features and target
    X = df[features]
    y = df['target']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_models(X_train, X_test, y_train, y_test):
    """
    Train and compare Logistic Regression and Random Forest models.
    Returns the best model.
    """
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Select best model based on ROC-AUC
    best_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_name]['model']
    
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_name}")
    print(f"ROC-AUC: {results[best_name]['roc_auc']:.4f}")
    print(f"{'='*50}")
    
    return best_model, results

def save_models(model, scaler):
    """
    Save the trained model and scaler to disk.
    """
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/heart_disease_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\nModels saved successfully!")
    print("  - models/heart_disease_model.pkl")
    print("  - models/scaler.pkl")

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    best_model, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save models
    save_models(best_model, scaler)
    
    print("\nTraining completed successfully!")
