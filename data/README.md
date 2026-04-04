# Data

This project uses the **UCI Cleveland Heart Disease Dataset**.

## Source

The dataset is automatically downloaded from the UCI ML Repository at runtime:

```
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
```

No manual download is needed. The `src/train.py` and `app.py` scripts fetch it directly.

## Features

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | age | int | Age in years |
| 2 | sex | binary | 1=male, 0=female |
| 3 | cp | int | Chest pain type (0-3) |
| 4 | trestbps | int | Resting blood pressure (mm Hg) |
| 5 | chol | int | Serum cholesterol (mg/dl) |
| 6 | fbs | binary | Fasting blood sugar > 120 mg/dl |
| 7 | restecg | int | Resting ECG results (0-2) |
| 8 | thalach | int | Maximum heart rate achieved |
| 9 | exang | binary | Exercise induced angina |
| 10 | oldpeak | float | ST depression induced by exercise |
| 11 | slope | int | Slope of peak exercise ST segment |
| 12 | ca | int | Number of major vessels (0-3) |
| 13 | thal | int | Thalassemia (1=normal, 2=fixed defect, 3=reversible defect) |
| 14 | target | binary | Heart disease present (1=yes, 0=no) |

## License

UCI Machine Learning Repository datasets are freely available for research purposes.
Citation: Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease.
