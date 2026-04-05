<div align="center">

# ❤️ Heart Disease Risk Predictor

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:e74c3c,100:c0392b&height=160&section=header&text=Heart%20Disease%20Risk%20Predictor&fontSize=32&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=End-to-End%20ML%20%7C%20Random%20Forest%20%7C%20Streamlit%20App&descSize=15&descAlignY=58" />

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://heart-disease-risk-predictorgitcdheart-disease-risk-predictor.streamlit.app)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-27ae60?style=for-the-badge)](LICENSE)

**An end-to-end machine learning project that predicts heart disease risk using the UCI Cleveland dataset.**
Includes EDA, model training (Logistic Regression & Random Forest), and an interactive Streamlit web app.

[🚀 **Try the Live App →**](https://heart-disease-risk-predictorgitcdheart-disease-risk-predictor.streamlit.app)

</div>

---

## 🌟 Features

✅ **Live Streamlit Web App** — no install needed, runs in your browser  
✅ **Two ML Models** — Random Forest & Logistic Regression, switchable in the UI  
✅ **86.67% Accuracy | 0.9464 AUC-ROC** — trained on UCI Cleveland dataset  
✅ **13 Interactive Input Controls** — sliders & dropdowns for every clinical feature  
✅ **Auto Data Loading** — fetches UCI dataset at runtime, no manual download  
✅ **Risk Probability Score** — see exact % confidence alongside the prediction  
✅ **Clean Project Structure** — modular `src/train.py` + `app.py` separation  

---

## 📊 Model Performance

| Model | Accuracy | AUC-ROC | CV AUC (5-fold) |
|-------|:--------:|:-------:|:---------------:|
| 🤖 Random Forest | **86.67%** | **0.9464** | ~0.93 |
| 📈 Logistic Regression | ~84% | ~0.91 | ~0.90 |

> Trained on 237 samples, tested on 60 samples from the UCI Cleveland Heart Disease Dataset (303 patients total).

---

## 🚀 Live Demo

**[https://heart-disease-risk-predictorgitcdheart-disease-risk-predictor.streamlit.app](https://heart-disease-risk-predictorgitcdheart-disease-risk-predictor.streamlit.app)**

| Feature | Description |
|---------|-------------|
| 🧠 Model selector | Switch between Random Forest and Logistic Regression |
| 🎯 Predict button | Get instant heart disease risk with probability % |
| 📊 Metric cards | Live accuracy, AUC-ROC and training sample count |
| 🔍 Patient summary | Full table of entered parameters after prediction |
| 🗂️ Dataset viewer | Expandable sample of the raw UCI data |

---

## 🏗️ Project Structure

```
📂 heart-disease-risk-predictor/
├── 📝 app.py               # Streamlit web app (main entry point)
├── 📝 requirements.txt     # Python dependencies
├── 📂 src/
│   └── 📝 train.py         # Model training: EDA + LR + RF + joblib export
├── 📂 data/
│   └── 📝 README.md        # Dataset info & feature descriptions
└── 📝 README.md            # You are here
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10 |
| ML | scikit-learn (Random Forest, Logistic Regression) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Web App | Streamlit |
| Model Persistence | joblib |
| Dataset | UCI Cleveland Heart Disease (via requests) |

---

## ⚡ Quick Start

### 1️⃣ Clone the repo
```bash
git clone https://github.com/rahul-roy56/heart-disease-risk-predictor.git
cd heart-disease-risk-predictor
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Train the model locally
```bash
python src/train.py
```

### 4️⃣ Launch the Streamlit app
```bash
streamlit run app.py
```
App opens at `http://localhost:8501` — dataset is fetched automatically, no CSV download needed.

---

## 🧬 Dataset

This project uses the **UCI Cleveland Heart Disease Dataset** — 303 patients, 14 features.

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | int | Age in years |
| 2 | `sex` | binary | 1 = male, 0 = female |
| 3 | `cp` | int | Chest pain type (0–3) |
| 4 | `trestbps` | int | Resting blood pressure (mm Hg) |
| 5 | `chol` | int | Serum cholesterol (mg/dl) |
| 6 | `fbs` | binary | Fasting blood sugar > 120 mg/dl |
| 7 | `restecg` | int | Resting ECG results (0–2) |
| 8 | `thalach` | int | Max heart rate achieved |
| 9 | `exang` | binary | Exercise induced angina |
| 10 | `oldpeak` | float | ST depression (exercise vs rest) |
| 11 | `slope` | int | Slope of peak exercise ST segment |
| 12 | `ca` | int | Major vessels coloured (0–3) |
| 13 | `thal` | int | Thalassemia type |
| 14 | `target` | binary | Heart disease present (1 = yes) |

---

## 📝 License

This project is **open-source** and free to use. Feel free to fork, improve, or contribute! 💡

---

<div align="center">

### 🔗 Connect with me

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rahulroy0499/)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rahul-roy56)
[![Email](https://img.shields.io/badge/-Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:roy.rah@northeastern.edu)

**Rahul Roy** — MS Applied Machine Intelligence @ Northeastern University, Boston

💚 Stay heart-healthy and keep building! 🚀

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:e74c3c,100:c0392b&height=100&section=footer" />

</div>
