# ML Classification Models - Wine Quality Prediction

## Problem Statement

This project implements and compares 6 different machine learning classification models on the Wine Quality dataset. The goal is to predict whether a wine is of good quality (quality >= 6) or bad quality (quality < 6) based on physicochemical properties.

## Dataset Description

- **Dataset Name:** Wine Quality Dataset (Red Wine)
- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/ml/datasets/wine+quality
- **Number of Features:** 11 physicochemical features
- **Number of Instances:** 1599 samples
- **Classification Type:** Binary Classification
- **Target Variable:** quality_binary (0 = Bad Wine, 1 = Good Wine)

The dataset represents red wine samples from the Portuguese "Vinho Verde" wine.

## Models Used

1. **Logistic Regression** - Linear model using sigmoid function
2. **Decision Tree** - Tree-based classifier with max_depth=10
3. **K-Nearest Neighbors (kNN)** - Instance-based learning with k=5
4. **Naive Bayes** - Gaussian probabilistic classifier
5. **Random Forest** - Ensemble of 100 decision trees
6. **XGBoost** - Gradient boosting ensemble method

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.7406 | 0.8242 | 0.7419 | 0.7406 | 0.7409 | 0.4808 |
| Decision Tree | 0.7562 | 0.7755 | 0.7562 | 0.7562 | 0.7562 | 0.5102 |
| kNN | 0.7406 | 0.8117 | 0.7407 | 0.7406 | 0.7407 | 0.4790 |
| Naive Bayes | 0.7219 | 0.7884 | 0.7282 | 0.7219 | 0.7219 | 0.4500 |
| Random Forest | 0.8062 | 0.9018 | 0.8077 | 0.8062 | 0.8065 | 0.6128 |
| XGBoost | 0.8250 | 0.8963 | 0.8259 | 0.8250 | 0.8252 | 0.6497 |

## Observations on Model Performance

### Logistic Regression
- Performance: F1 score of 0.7409 with good AUC of 0.8242, showing moderate performance.
- Strengths: Simple, interpretable, fast training. High AUC indicates good probability estimates.
- Weaknesses: Limited by linear decision boundary, cannot capture complex non-linear interactions.
- Observation: Performed reasonably well as a baseline, but clearly outperformed by complex models, indicating wine quality benefits from capturing non-linear feature interactions.

### Decision Tree
- Performance: F1 score of 0.7562, solid mid-range performance.
- Strengths: Highly interpretable with clear decision rules. Captures non-linear relationships without scaling.
- Weaknesses: Prone to overfitting despite max_depth constraint.
- Observation: Better than logistic regression and kNN, showing non-linear boundaries help. Lower AUC than logistic regression suggests less reliable probability estimates.

### kNN
- Performance: F1 score of 0.7407, similar to logistic regression.
- Strengths: Simple, no training phase, works with local patterns.
- Weaknesses: Computationally expensive, sensitive to scaling and k value.
- Observation: Performance nearly identical to logistic regression suggests local neighborhood patterns aren't more informative than global linear patterns for wine quality.

### Naive Bayes
- Performance: Lowest F1 of 0.7219.
- Strengths: Fast, requires minimal resources.
- Weaknesses: Feature independence assumption violated - wine properties are correlated.
- Observation: Poor performance confirms wine chemical properties are interdependent. pH relates to acidity, alcohol affects density - making Naive Bayes inappropriate.

### Random Forest (Ensemble)
- Performance: Strong F1 of 0.8065 and excellent AUC of 0.9018.
- Strengths: Robust, reduces overfitting, handles non-linear relationships well.
- Weaknesses: Less interpretable, more computational resources needed.
- Observation: Substantial jump from Decision Tree (0.7562 to 0.8065) demonstrates ensemble learning power. Averaging multiple trees captures complex patterns while avoiding overfitting.

### XGBoost (Ensemble)
- Performance: Best with F1 of 0.8252, AUC of 0.8963, highest MCC of 0.6497.
- Strengths: Sequential error correction, built-in regularization, excellent with feature interactions.
- Weaknesses: More complex, longer training time, less interpretable.
- Observation: Best performance across all metrics. Sequential learning where each tree corrects previous errors gives slight edge over Random Forest for wine quality classification.

### Overall Insights

- **Best Model:** XGBoost (F1: 0.8252), followed by Random Forest (F1: 0.8065)
- **Ensemble vs Individual:** Clear 5-7% performance gap between ensemble methods and individual models demonstrates that wine quality depends on complex feature interactions
- **Key Finding:** Strong ensemble performance vs linear models shows wine quality determined by complex, non-linear chemical property interactions
- **Feature Relationships:** Poor Naive Bayes performance confirms features are highly interdependent
- **Practical Recommendation:** XGBoost for production due to superior performance. Random Forest if interpretability needed.

## Streamlit App Features

1. **Dataset Upload** - CSV upload with preview and statistics
2. **Model Selection** - Dropdown with 6 models and descriptions
3. **Evaluation Metrics** - All 6 metrics with interactive visualizations
4. **Confusion Matrix** - Interactive heatmap with detailed breakdown
5. **Additional Features** - Predictions download, classification report, distribution charts

## Links

- **GitHub Repository:** https://github.com/TPAteeq/ml-wine-quality-classification
- **Live Streamlit App:** to be upated
- **Dataset Source:** https://archive.ics.uci.edu/ml/datasets/wine+quality

## Author

Mohammed Ateequddin
2025AB05144
M.Tech (AIML/DSE) - BITS Pilani
Machine Learning Assignment 2

---

**Date Completed:** 13th February 2026