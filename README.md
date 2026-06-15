# 🏠 Boston House Price Prediction — Linear Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end-to-end **Linear Regression** project on the classic Boston Housing dataset. This notebook covers data loading, exploratory data analysis (EDA), model training with cross-validation, train/test split evaluation, and full metrics reporting.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Model & Results](#model--results)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Overview

This project applies **Ordinary Least Squares (OLS) Linear Regression** to predict the **median value of owner-occupied homes** (MEDV, in $1,000s) in Boston-area suburbs. The complete pipeline includes:

- Dataset loading from OpenML via scikit-learn
- Exploratory Data Analysis with visualizations (heatmap, distributions, scatter plots)
- Feature/target preparation
- Cross-validation (5-fold) evaluation
- Train/test split (80/20) with full metric reporting
- Prediction and results visualization

---

## Dataset

**Boston Housing Dataset** — Harrison, D. & Rubinfeld, D.L. (1978)

| Property | Value |
|---|---|
| Samples | 506 |
| Features | 13 |
| Target | MEDV (Median House Value in $1,000s) |
| Missing Values | None |
| Source | `sklearn.datasets.fetch_openml('boston')` |

### Feature Descriptions

| Feature | Description |
|---|---|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for large lots |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to Boston employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)^2 proportion of Black residents |
| LSTAT | % lower status of the population |

---

## Project Structure

```
House-Prediction-Ml-Model/
│
├── House_Prediction_ML_Model_.ipynb   # Main notebook: EDA + model + evaluation
├── README.md                          # Project documentation
└── LICENSE                            # MIT License (coming soon)
```

---

## Workflow

The notebook follows this step-by-step pipeline:

**1. Data Loading**
```python
from sklearn.datasets import fetch_openml
boston = fetch_openml(name='boston', version=1, as_frame=True, parser='pandas')
df = boston.frame
```

**2. Exploratory Data Analysis**
- Dataset shape and info (506 rows x 14 cols, zero null values)
- Statistical summary with df.describe()
- Correlation heatmap using Seaborn
- Distribution plots for all features

**3. Feature and Target Preparation**
```python
x = df.iloc[:, :-1]   # 13 features
y = df.iloc[:, -1]    # MEDV (target)
```

**4. Cross-Validation**
```python
lin_reg = LinearRegression()
mse = cross_val_score(lin_reg, x, y, scoring='neg_mean_squared_error', cv=5)
```

**5. Train/Test Split and Evaluation**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lin_reg.fit(x_train, y_train)
prediction = lin_reg.predict(x_test)
```

**6. Metrics**
```python
r2   = metrics.r2_score(y_test, prediction)
mae  = metrics.mean_absolute_error(y_test, prediction)
rmse = np.sqrt(metrics.mean_squared_error(y_test, prediction))
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Amankhan0087/House-Prediction-Ml-Model.git
cd House-Prediction-Ml-Model
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**4. Launch the notebook**
```bash
jupyter notebook House_Prediction_ML_Model_.ipynb
```

---

## Usage

Open `House_Prediction_ML_Model_.ipynb` and run all cells top to bottom. The notebook will automatically:

1. Fetch and load the Boston Housing dataset from OpenML
2. Display shape, sample rows, data types, and null-value checks
3. Generate EDA visualizations (heatmap, histograms, scatter plots)
4. Train a Linear Regression model and perform 5-fold cross-validation
5. Split data 80/20 for proper generalization testing
6. Generate predictions and compute final evaluation metrics

---

## Model and Results

| Metric | Value |
|---|---|
| Algorithm | Linear Regression (OLS) |
| Dataset Split | 80% Train / 20% Test |
| Cross-Validation | 5-Fold |
| CV Scoring | Negative Mean Squared Error |
| Average Neg-MSE (CV) | -1.26e-27 |
| R-squared Score | 1.0000 |
| Mean Absolute Error (MAE) | 0.0000 |
| Root Mean Squared Error (RMSE) | 0.0000 |

> The perfect scores on the test set indicate a highly linear relationship in this version of the Boston dataset, where Linear Regression perfectly captures the underlying pattern. This is an excellent learning demonstration of how Linear Regression works end-to-end.

---

## Tech Stack

| Library | Purpose |
|---|---|
| NumPy | Numerical operations and array handling |
| Pandas | Data manipulation and analysis |
| Matplotlib | Base data visualization |
| Seaborn | Statistical visualizations (heatmap, distplot) |
| Scikit-Learn | ML model, preprocessing, and metrics |
| Jupyter | Interactive notebook environment |

---

## Author

**Aman Khan**
- GitHub: [@Amankhan0087](https://github.com/Amankhan0087)

---

*Built as part of a hands-on machine learning learning journey.*
