# 📊 Model Comparison – Smart Lender

**Project:** Smart Lender – Loan Eligibility Prediction  
**Prepared By:** Testing & QA Team  
**Date:** July 2026  

---

## Overview

During the development of Smart Lender, four machine learning classification models were trained
and evaluated on the same dataset split (80% training / 20% testing). This document compares
their performance and explains why Random Forest was selected as the production model.

---

## Dataset Split

| Split | Records | Percentage |
|-------|---------|------------|
| Training Set | ~491 rows | 80% |
| Test Set | ~123 rows | 20% |
| **Total** | **614 rows** | **100%** |

---

## Model Accuracy Comparison

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|--------|----------|
| 🥇 1st | **Random Forest** | **82.1%** | 83.5% | 90.1% | 86.7% |
| 🥈 2nd | XGBoost | 80.5% | 81.8% | 89.0% | 85.2% |
| 🥉 3rd | KNN | 76.4% | 77.9% | 86.4% | 81.9% |
| 4th | Decision Tree | 73.2% | 74.1% | 84.0% | 78.7% |

> *Note: Accuracy values are based on the 20% test split with `random_state=42`.*

---

## Detailed Model Analysis

### 1. Decision Tree Classifier
**Accuracy: ~73.2%**

| Property | Detail |
|----------|--------|
| Algorithm | Recursive binary splitting (CART) |
| Parameters | `random_state=42` |
| Pros | Highly interpretable, fast to train, no scaling required |
| Cons | Prone to overfitting, sensitive to small data changes |
| Why Not Selected | Lowest accuracy; overfits on training data |

**How it works:**  
A Decision Tree builds a flowchart-like structure where each internal node tests a feature,
each branch represents a decision, and each leaf gives the final prediction (Approved/Rejected).
For loan prediction, it splits on features like Credit_History and Income thresholds.

---

### 2. K-Nearest Neighbors (KNN)
**Accuracy: ~76.4%**

| Property | Detail |
|----------|--------|
| Algorithm | Instance-based, distance metric (Euclidean) |
| Parameters | `n_neighbors=5` (default) |
| Scaling | Requires feature scaling (StandardScaler applied) |
| Pros | Simple to understand, no training phase |
| Cons | Slow on large datasets, sensitive to irrelevant features |
| Why Not Selected | Requires scaling, performance degrades with scale |

**How it works:**  
KNN predicts by finding the K closest training examples (measured by Euclidean distance)
and returning the majority class. For loan prediction, it checks "what did applicants with
similar income, loan amount, and credit history get?" KNN required StandardScaler
preprocessing since income values are much larger than binary flags.

---

### 3. XGBoost (Extreme Gradient Boosting)
**Accuracy: ~80.5%**

| Property | Detail |
|----------|--------|
| Algorithm | Gradient boosting (ensemble of weak learners) |
| Parameters | `eval_metric='logloss'`, `random_state=42` |
| Pros | High accuracy, handles missing values, regularisation built-in |
| Cons | More complex to tune, slower training |
| Why Not Selected | Slightly lower accuracy than Random Forest on this dataset |

**How it works:**  
XGBoost builds an ensemble of decision trees **sequentially** – each new tree corrects the
errors of the previous one. It uses gradient descent to minimise prediction loss. It's one
of the most powerful algorithms in tabular data competitions (Kaggle, etc.) and performed
strongly here at 80.5%.

---

### 4. ✅ Random Forest (SELECTED MODEL)
**Accuracy: ~82.1%**

| Property | Detail |
|----------|--------|
| Algorithm | Bagging ensemble of decision trees |
| Parameters | `n_estimators=200`, `random_state=42` |
| Saved As | `models/loan_model.pkl` |
| Pros | High accuracy, robust to overfitting, handles missing data |
| Cons | Less interpretable than single tree, larger model file |
| **Why Selected** | **Highest accuracy + best generalisation on test data** |

**How it works:**  
Random Forest trains **200 independent decision trees**, each on a random bootstrap sample
of the training data and a random subset of features. Final prediction is determined by
**majority vote** across all 200 trees. This "wisdom of the crowd" approach dramatically
reduces overfitting compared to a single Decision Tree.

---

## Visual Comparison

```
Accuracy (%)
90 |
85 |                                          ██████
80 |                             ██████       ██████
75 |              ██████         ██████       ██████
70 |   ██████     ██████         ██████       ██████
   |   Decision   KNN            XGBoost      Random
   |   Tree       76.4%          80.5%        Forest
   |   73.2%                                  82.1%
```

---

## Why Random Forest is the Best Choice for Loan Prediction

1. **Highest Accuracy (82.1%)** – Outperforms all other models on the holdout test set.

2. **Handles Class Imbalance** – Loan datasets often have more approvals than rejections.
   Random Forest's ensemble averaging naturally handles this better than a single tree.

3. **No Feature Scaling Required** – Unlike KNN, Random Forest works directly with raw
   feature values, simplifying the prediction pipeline in `app.py`.

4. **Robust to Outliers** – High-income outliers (some applicants earn ₹81,000/month)
   do not destabilise the predictions because the ensemble averages out extreme splits.

5. **Missing Value Tolerance** – Combined with our preprocessing pipeline (median/mode
   imputation), the model handles missing Credit_History and LoanAmount values gracefully.

6. **Production Proven** – Random Forests are widely used in credit scoring and financial
   services industry for exactly this type of tabular classification problem.

---

## Recommendation

> ✅ **Keep Random Forest as the production model.**  
> If higher accuracy is needed in future, consider XGBoost with hyperparameter tuning
> (GridSearchCV) or a stacked ensemble combining RF + XGBoost predictions.

---
*Document prepared as part of the Smart Lender internship project – Testing & Documentation Phase*
