# LoanLens 💰🏦
## Comparative Analysis of Machine Learning Algorithms for Personal Loan Prediction 

> A supervised machine learning project that compares **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)** for predicting whether a bank customer will accept a personal loan offer.

**Python** | **Scikit-learn** | **Imbalanced-Learn** | **Supervised Learning** | **Binary Classification**

---

## Overview 🚀

In the banking sector, identifying customers who are likely to accept a personal loan is important for improving targeted marketing, reducing unnecessary campaign costs, and supporting better decision-making.

This project applies and compares four supervised machine learning algorithms on the `Bank_Personal_Loan_Modelling` dataset:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**

The objective is to classify whether a customer will accept a personal loan offer.

Target variable:
- `Personal Loan = 0` -> customer did **not** accept the loan
- `Personal Loan = 1` -> customer **accepted** the loan

---

## Dataset 📊

**Dataset:** `Bank_Personal_Loan_Modelling`  
**Total Records:** `5000`  
**Problem Type:** Binary Classification  
**Target Column:** `Personal Loan`

### Main Features ⛑️
- `Age`
- `Experience`
- `Income`
- `Family`
- `CCAvg`
- `Education`
- `Mortgage`
- `Securities Account`
- `CD Account`
- `Online`
- `CreditCard`

### Notes 📋
- `ID` was removed because it is only an identifier
- the dataset is slightly imbalanced
- multiple evaluation metrics were used instead of relying only on accuracy

---

## Preprocessing 🧹

The following preprocessing steps were applied before model training:

- corrected invalid negative values in `Experience` by clipping them to `0`
- removed non-informative columns such as `ID`
- used an `80/20` train-test split with stratification
- applied algorithm-specific preprocessing where necessary
- used `StandardScaler` for algorithms sensitive to feature magnitude, especially SVM
- used `5-fold cross-validation` during hyperparameter tuning

To address class imbalance, different approaches were used depending on the algorithm:
- threshold tuning and calibration for Logistic Regression
- class balancing for SVM
- `SMOTE + SVM` as an additional imbalance-aware experiment
- `SMOTE + Random Forest` as an additional experiment for comparison

---

## Algorithms Used ✨

### 1. Logistic Regression
Logistic Regression was used as a baseline linear classifier. It is simple, efficient, and highly interpretable. Different variants such as calibration and threshold tuning were tested to improve predictive balance.

### 2. Decision Tree
Decision Tree was used because of its interpretability and ability to capture nonlinear decision rules. It provides a simple tree-based representation of the classification process.

### 3. Random Forest
Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve robustness and reduce overfitting. It also provides feature importance values for interpretation.

### 4. Support Vector Machine (SVM)
SVM was used as a margin-based classifier for binary classification. It is effective for structured datasets and was combined with preprocessing, scaling, and SMOTE to improve minority-class handling.

---

## Comparative Performance 💹

| Algorithm | Final Variant Used for Comparison | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Calibrated LR (Isotonic) | 0.971 | 0.868 | 0.822 | 0.844 | 0.988 |
| Decision Tree | Final tuned model | 0.992 | 0.968 | 0.948 | 0.958 | 0.993 |
| Random Forest | Final tuned model | 0.990 | 0.9574 | 0.9375 | 0.9474 | 0.9977 |
| Support Vector Machine | SMOTE + SVM | 0.983 | 0.8911 | 0.9375 | 0.9137 | 0.9943 |

---

## Key Findings 🗝️*️⃣

- **Decision Tree** achieved the highest **accuracy** and **F1-score** on this dataset.
- **Random Forest** provided the strongest overall **robustness and generalization**, with the highest **ROC-AUC**.
- **Support Vector Machine**, especially with SMOTE, produced strong and balanced classification performance.
- **Logistic Regression** gave stable and interpretable results and served as a strong baseline model.

---


