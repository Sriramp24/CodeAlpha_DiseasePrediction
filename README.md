# CodeAlpha Task 4: Advanced Disease Prediction & Diagnostic ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-latest-green.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/lightgbm-latest-brightgreen.svg)](https://lightgbm.readthedocs.io/)

## Project Overview
This repository delivers an end-to-end clinical machine learning diagnostic pipeline for cardiovascular and metabolic disease risk prediction. The project implements comprehensive data science best practices including missing value imputation, outlier winsorization, physiological feature engineering, mutual information feature selection, multi-algorithm benchmarking with Stratified 5-Fold Cross-Validation, and an interactive diagnostic inference engine.

---

## 🔬 End-to-End Machine Learning Architecture

```mermaid
graph LR
    A[Clinical Records Dataset] --> B[Data Preprocessing & KNN Imputation]
    B --> C[Outlier Winsorization & Robust Scaling]
    C --> D[Domain Feature Engineering]
    D --> E[Mutual Information Feature Selection]
    E --> F[Multi-Model Benchmarking]
    F --> G[XGBoost / LightGBM / SVM / RF]
    G --> H[Model Diagnostics & ROC/PR Curves]
    H --> I[Production Inference Engine]
```

### 1. Data Ingestion & Preprocessing
- **Dataset**: `clinical_cardiovascular_disease_dataset.csv` (1,500 patient records, 15 clinical features).
- **Imputation**: K-Nearest Neighbors (KNN) Imputer (k=5) for missing clinical values (`resting_bp`, `cholesterol`, `bmi`).
- **Outlier Remediation**: Interquartile Range (IQR) boundary clipping to handle physiological anomalies.
- **Categorical Encoding**: One-Hot Encoding for multi-level nominal variables.

### 2. Domain Feature Engineering
- `heart_rate_reserve`: Theoretical maximum heart rate minus peak exercise rate (220 - Age - MaxHR).
- `bp_age_ratio`: Clinical vascular stiffness ratio (RestingBP / Age).
- `cardiometabolic_risk_index`: Combined metabolic burden metric ((Cholesterol * BMI) / 1000).
- `st_angina_interaction`: Myocardial ischemia stress interaction term.

### 3. Feature Selection & Analysis
- Statistical relevance ranking using **Mutual Information Score** (`mutual_info_classif`).
- Top discriminative feature extraction.

### 4. Multi-Model Benchmarking & Hyperparameter Optimization
- **Stratified 5-Fold Cross-Validation** across 5 classifiers:
  1. **XGBoost Classifier** (Gradient Boosted Decision Trees)
  2. **LightGBM Classifier** (Histogram-based GBDT)
  3. **Random Forest Classifier** (Bagged Trees)
  4. **Support Vector Machine (SVM)** (RBF Kernel)
  5. **Logistic Regression** (L2 Regularized Baseline)

---

## 📁 Repository Structure
```
.
├── disease_prediction_medical_data.ipynb      # Complete Step-by-Step Jupyter Notebook
├── clinical_cardiovascular_disease_dataset.csv  # Full Clinical Patient Records Dataset
├── README.md                                  # Technical Documentation
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
```

### Running the Notebook
Launch Jupyter Notebook to execute all cells sequentially:
```bash
jupyter notebook disease_prediction_medical_data.ipynb
```
