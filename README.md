# 🏠 Boston House Price Prediction — Linear Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚠️ **Work in Progress** — This project is actively being developed. Features and results will be updated as work progresses.

A practical implementation of **Linear Regression** on the classic Boston Housing dataset. This project demonstrates the end-to-end ML workflow: data loading, exploration, model training, and cross-validated evaluation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model & Results](#model--results)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

This project applies **Ordinary Least Squares (OLS) Linear Regression** to predict the **median value of owner-occupied homes** (in $1,000s) in the Boston metropolitan area. It covers:

- Dataset ingestion from OpenML via scikit-learn
- Feature inspection and exploratory data analysis
- Model training and fitting
- K-fold cross-validation for robust evaluation

---

## Dataset

**Boston Housing Dataset** — Harrison, D. & Rubinfeld, D.L. (1978)

| Property | Value |
|---|---|
| Samples | 506 |
| Features | 13 |
| Target | MEDV (Median House Value in $1,000s) |
| Source | sklearn.datasets.fetch_openml('boston') |

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
├── ML_Learning.ipynb       # Main notebook: EDA + Linear Regression
├── requirements.txt        # Python dependencies (coming soon)
├── README.md               # Project documentation
└── LICENSE                 # MIT License (coming soon)
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

**4. Launch Jupyter Notebook**
```bash
jupyter notebook ML_Learning.ipynb
```

---

## Usage

Open `ML_Learning.ipynb` and run all cells top to bottom. The notebook will:

1. Fetch the Boston Housing dataset automatically from OpenML
2. Display dataset shape, sample rows, and target values
3. Prepare feature matrix X and target vector y
4. Train a LinearRegression model using scikit-learn
5. Evaluate using 5-fold cross-validation and print MSE scores

---

## Model & Results

| Metric | Value |
|---|---|
| Algorithm | Linear Regression (OLS) |
| Cross-Validation | 5-Fold |
| Scoring | Negative Mean Squared Error |
| R² Score | In Progress |
| RMSE | In Progress |

> Results will be updated once the full pipeline with proper train/test split is complete.

---

## Tech Stack

| Library | Purpose |
|---|---|
| NumPy | Numerical operations |
| Pandas | Data manipulation & analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Scikit-Learn | ML model, preprocessing & evaluation |
| Jupyter | Interactive notebook environment |

---

## Roadmap

- [x] Load and explore the Boston Housing dataset
- [x] Implement baseline Linear Regression model
- [x] Cross-validation evaluation
- [ ] Proper train/test split
- [ ] EDA with visualizations (heatmap, distributions, scatter plots)
- [ ] Feature engineering & correlation analysis
- [ ] Try Ridge, Lasso, Random Forest models
- [ ] Hyperparameter tuning
- [ ] Save and export the final model
- [ ] Add requirements.txt and LICENSE

---

## Author

**Aman Khan**
- GitHub: [@Amankhan0087](https://github.com/Amankhan0087)

---

*This project is part of an ongoing machine learning learning journey. Contributions and feedback are welcome!*
