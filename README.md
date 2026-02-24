# Amazon Product Rating Prediction Engine

![Python](https://img.shields.io/badge/Python-3.10-blue)
![pandas](https://img.shields.io/badge/pandas-Data%20Processing-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Regression](https://img.shields.io/badge/Task-Regression-green)

## Overview
End-to-end machine learning pipeline predicting Amazon product star ratings using price-based features and product metadata — without leveraging review text.

This project evaluates how much predictive signal exists in observable pricing variables alone and compares linear vs nonlinear regression models using structured preprocessing pipelines to prevent data leakage.

Dataset: ~1,462 Amazon products across electronics and home categories.

---

## Benchmark Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.1877 | 0.2555 | 0.20 |
| Random Forest | **0.1592** | **0.2348** | **0.33** |
| KNN Regression | 0.1879 | 0.2544 | 0.21 |

Random Forest achieved the best performance, explaining ~33% of rating variance.

---

## Architecture

- **Pipeline:** scikit-learn `Pipeline` + `ColumnTransformer`
- **Numeric Processing:** StandardScaler
- **Categorical Processing:** OneHotEncoder
- **Train/Test Split:** 80/20
- **Target:** Continuous rating (1–5 scale)

### Models Evaluated
- Linear Regression (baseline)
- Random Forest Regressor (nonlinear ensemble)
- K-Nearest Neighbors Regressor

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² (Coefficient of Determination)

---

## Feature Engineering

### Inputs
- discounted_price
- actual_price
- discount_percentage
- rating_count
- category (one-hot encoded)

### Data Cleaning
- Removed currency symbols and commas
- Converted percentage strings to numeric
- Dropped invalid rating entries
- Ensured numeric type casting
- Prevented data leakage using preprocessing pipelines

---

## Results

Random Forest captured nonlinear interactions between pricing, popularity (rating count), and category effects.

### Key Findings
- **Rating count** was the strongest predictor
- Discount percentage and pricing contributed moderate signal
- Category effects influenced baseline rating distributions
- Price-based features alone explain only one-third of rating variance

Conclusion: Pricing contains predictive signal, but textual sentiment and product quality likely drive the majority of rating behavior.

---

## Visualizations

### Model Comparison
![Model Comparison](results/model_comparison.png)

### Feature Importance (Random Forest)
![Feature Importance](results/feature_importance.png)

### Residual Distribution
![Residuals](results/residuals.png)

---

## Tech Stack
- Python
- pandas
- numpy
- scikit-learn
- matplotlib

---

## Project Structure
```
├── amazon_rating_prediction.ipynb   # End-to-end regression pipeline
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
├── data/
│   └── amazon.csv                   # Dataset
└── results/
    ├── model_comparison.png
    ├── feature_importance.png
    └── residuals.png
```
