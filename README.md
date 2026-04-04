# Heart Disease Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An end-to-end machine learning project that predicts heart disease risk using the UCI Cleveland Heart Disease dataset. Includes exploratory data analysis, model training (Logistic Regression & Random Forest), and an interactive Streamlit web app for live predictions.

---

## Demo

Run the app locally:
```bash
streamlit run app.py
```

---

## Project Structure

```
heart-disease-risk-predictor/
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── src/
│   └── train.py            # Model training script
├── data/
│   └── README.md           # Dataset info & download instructions
└── README.md
```

---

## Dataset

This project uses the **UCI Cleveland Heart Disease Dataset** (303 patients, 14 features).

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1=male, 0=female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0-2) |
| thalach | Max heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia type |
| target | Heart disease present (1=yes, 0=no) |

Dataset is loaded directly from UCI ML Repository — no manual download needed.

---

## Models

| Model | Accuracy | AUC-ROC |
|-------|----------|----------|
| Logistic Regression | ~85% | ~0.91 |
| Random Forest | ~88% | ~0.93 |

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/rahul-roy56/heart-disease-risk-predictor.git
cd heart-disease-risk-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python src/train.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## Tech Stack

- **Python 3.10**
- **scikit-learn** — Logistic Regression, Random Forest, cross-validation
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — EDA visualizations
- **Streamlit** — Interactive web app
- **joblib** — Model serialization

---

## Author

**Rahul Roy** — MS Applied Machine Intelligence @ Northeastern University

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rahulroy0499/)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/rahul-roy56)
