# LoanLens 🏦💳
## Personal Loan Acceptance Prediction with SVM 💹💰

Supervised binary classification on the `Bank_Personal_Loan_Modelling` dataset using two SVM-based pipelines:

- `Base SVM`
- `SMOTE + SVM`

**Python** | **Scikit-learn** | **Imbalanced-Learn** | **Binary Classification** | **SVM**

---

## Project Snapshot 👩‍💻

This project predicts whether a bank customer will accept a personal loan offer.

Target variable: ⚙️

- `Personal Loan = 0` -> customer did not accept the loan
- `Personal Loan = 1` -> customer accepted the loan

Two versions are included in the project:

- `Base SVM` using class balancing
- `SMOTE + SVM` for improved minority-class detection

<p align="center">
  <img src="./outputs/analysis_helpers/data-transformed-svm.jpg" alt="SVM Data Transformation Preview" width="760">
</p>

---

## Why SVM 🤔

Support Vector Machine, or SVM, is a supervised learning algorithm mainly used for classification.

Its main idea is to find the best separating boundary between classes. This boundary is called a **hyperplane**. SVM tries to maximize the **margin**, which is the distance between the boundary and the closest data points from each class. Those closest points are called **support vectors**, and they are the most important points in defining the classifier.

In this project, SVM is used to classify customers into:

- customers who will not accept the loan
- customers who will accept the loan

Because SVM is sensitive to feature scale, the model pipeline first preprocesses the data and standardizes the numeric features before training.

---

## Dataset 📊

| Item | Details |
|---|---|
| Dataset | `Bank_Personal_Loan_Modelling` |
| Total records | `5000` |
| Problem type | Binary classification |
| Target | `Personal Loan` |

Main input features used:

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
- `ZIP Code`

Dropped column:

- `ID`

---

## Preprocessing And Modeling Decisions 🧹

The following decisions were made specifically to suit SVM:

- Negative values in `Experience` were corrected by clipping them to `0`
- `ID` was removed because it is only an identifier
- Numerical features were standardized using `StandardScaler`
- Categorical and discrete features were handled through the preprocessing pipeline
- The dataset was split using an `80/20` train-test split with `stratify=y`
- `5-fold cross-validation` was used during hyperparameter tuning
- `F1-score` was used as the main tuning metric because the dataset is imbalanced

Two modeling strategies were implemented:

### 1. Base SVM
- Uses an SVM classifier with hyperparameter tuning
- Includes `class_weight='balanced'` to reduce bias toward the majority class

### 2. SMOTE + SVM
- Uses `SMOTE` only on the training portion inside the pipeline
- Helps improve minority-class recall without leaking synthetic data into the test set

---

## Project Structure 🧱

- `dataset/` : source dataset
- `src/svm_loan/` : reusable project code
- `scripts/` : training and prediction scripts
- `notebooks/` : notebook version of the implementation
- `outputs/base_svm/` : normal SVM figures and reports
- `outputs/smote_svm/` : SMOTE + SVM figures and reports
- `outputs/analysis_helpers/` : presentation visuals
- `artifacts/base_svm/` : saved base SVM model and metadata
- `artifacts/smote_svm/` : saved SMOTE + SVM model and metadata
- `samples/` : sample prediction input

---

## Run The Project 🙆‍♀️

Train base SVM:

```bash
python scripts/run_training.py
```

Train SMOTE + SVM:

```bash
python scripts/run_smote_svm_experiment.py
```

Predict with base SVM:

```bash
python scripts/predict_svm.py
```

Predict with SMOTE + SVM:

```bash
python scripts/predict_smote_svm.py
```

---

## Results ®️

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Base SVM | `0.983` | `0.9247` | `0.8958` | `0.9101` | `0.9942` |
| SMOTE + SVM | `0.983` | `0.8911` | `0.9375` | `0.9137` | `0.9943` |

Interpretation:

- `Base SVM` gives higher precision
- `SMOTE + SVM` gives higher recall and slightly better F1-score
- `SMOTE + SVM` is the better choice when detecting more positive loan cases is important

---

## Saved Outputs 🚀

### Base SVM

- Model: `artifacts/base_svm/model/svm_model.joblib`
- Metrics: `outputs/base_svm/reports/metrics.json`
- Report: `outputs/base_svm/reports/classification_report.txt`
- Figures:
  - `outputs/base_svm/figures/confusion_matrix.png`
  - `outputs/base_svm/figures/roc_curve.png`
  - `outputs/base_svm/figures/learning_curve_svm.png`
  - `outputs/base_svm/figures/validation_curve_svm.png`
  - `outputs/base_svm/figures/train_test_metrics_svm.png`

### SMOTE + SVM

- Model: `artifacts/smote_svm/model/smote_svm_model.joblib`
- Metrics: `outputs/smote_svm/reports/metrics_smote_svm.json`
- Report: `outputs/smote_svm/reports/classification_report_smote_svm.txt`
- Figures:
  - `outputs/smote_svm/figures/confusion_matrix_smote_svm.png`
  - `outputs/smote_svm/figures/train_test_metrics_smote_svm.png`

---

## My Contribution 👏

My contribution to this project was the complete SVM implementation, including:

- preprocessing for SVM-ready input
- scaling and pipeline setup
- train-test splitting and cross-validation
- hyperparameter tuning
- imbalance handling with both balanced SVM and SMOTE + SVM
- evaluation using multiple metrics
- generating plots and saved artifacts
- reusable prediction scripts

---
By Gamage S S J - IT22607232 🐰✨