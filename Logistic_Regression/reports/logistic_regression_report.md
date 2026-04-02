# Advanced Supervised Learning Report: Personal Loan Prediction (Logistic Regression Only)

## Introduction
This study develops an advanced Logistic Regression pipeline for predicting whether a bank customer will accept a personal loan offer (`Personal Loan`: 1/0). The objective is to improve predictive performance while maintaining model interpretability and to present a clear precision-recall trade-off analysis.

## Dataset Description
- Dataset: Bank Personal Loan Modelling
- File: `dataset/Bank_Personal_Loan_Modelling.csv`
- Size: 5,000 records
- Class distribution: 4,520 negatives, 480 positives

The class imbalance motivates the use of recall-sensitive evaluation and imbalance-aware training strategies.

## Methodology

### 1. Advanced Feature Engineering
Added engineered features:
- `income_per_family = Income / Family`
- `income_x_education = Income * Education`
- `ccavg_x_income = CCAvg * Income`
- `mortgage_to_income = Mortgage / (Income + 1)`
- `age_bin` (binned age category)
- `income_bin` (binned income category)

Removed:
- `ID`, `ZIP Code`

### 2. Multicollinearity Handling
- High correlation detected between `Age` and `Experience` (`|r| = 0.9942`)
- `Age` was dropped (we retained the more target-relevant feature)
- VIF-based filtering then removed `ccavg_x_income` (VIF 12.2742)

Final VIF diagnostics are available in `outputs/final_vif_table.csv`.

### 3. Logistic Regression Variants Evaluated
1. Baseline LR (`L2`, `C=1`, threshold 0.5)
2. Tuned LR (`GridSearchCV`: `C`, `penalty`, `solver`)
3. Tuned LR with optimized threshold
4. Balanced LR (`class_weight='balanced'`)
5. SMOTE + LR
6. Calibrated LR (Platt vs Isotonic, selected by validation Brier score)

Best tuned parameters:
- `C = 2`, `penalty = l1`, `solver = liblinear`

### 4. Threshold Optimization
Validation threshold sweep (`0.05` to `0.95`) used:
- F1-score maximization
- Youden’s J maximization

Results:
- Best F1 threshold: `0.35`
- Best Youden’s J threshold: `0.23`

### 5. Calibration
Compared:
- Platt scaling (`sigmoid`)
- Isotonic regression

Chosen calibrated model: **Calibrated LR (Isotonic)**.

## Results and Discussion

### Logistic Regression Metrics Comparison

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated LR (Isotonic) | 0.50 | 0.971 | 0.868 | 0.823 | 0.845 | 0.988 | 0.932 |
| Tuned LR (Best F1 Thr) | 0.35 | 0.969 | 0.828 | 0.854 | 0.841 | 0.989 | 0.933 |
| Tuned LR (0.50) | 0.50 | 0.969 | 0.874 | 0.792 | 0.831 | 0.989 | 0.933 |
| Baseline LR | 0.50 | 0.967 | 0.889 | 0.750 | 0.814 | 0.986 | 0.924 |
| SMOTE LR | 0.50 | 0.933 | 0.596 | 0.938 | 0.729 | 0.987 | 0.923 |
| Balanced LR | 0.50 | 0.920 | 0.548 | 0.958 | 0.697 | 0.986 | 0.915 |

### Trade-off Interpretation
- Baseline LR gives stronger precision but lower recall.
- Tuned LR with threshold `0.35` improves recall with moderate precision drop.
- Balanced LR and SMOTE aggressively increase recall but reduce precision substantially.
- Calibrated LR (Isotonic) provides the best overall F1 among LR variants with strong ROC-AUC.

### Best Model (Within Logistic Regression Family)
- **Best overall LR variant:** `Calibrated LR (Isotonic)`
- **Best threshold for recall-sensitive tuned LR:** `0.35`

## Visualizations (LR-only)
Generated files:
- `figures/confusion_matrices_lr_variants.png`
- `figures/roc_curves_lr_variants.png`
- `figures/pr_curves_lr_variants.png`
- `figures/threshold_vs_f1_youden.png`
- `figures/calibration_curves.png`
- `figures/lr_feature_importance.png`

## Critical Analysis
1. False negatives were reduced by threshold tuning and imbalance-aware strategies.
2. Multicollinearity control (correlation + VIF) improved coefficient stability.
3. Performance gains are mainly due to combined effects of feature engineering, regularization tuning, and threshold calibration rather than a single change.

## Limitations
1. Evaluation is on one dataset split (no external dataset).
2. Engineered features are domain-guided but not exhaustive.
3. Business cost matrix was not explicitly optimized.

## Future Improvements
1. Optimize threshold using explicit business costs.
2. Use repeated/nested CV for more stable estimates.
3. Add model monitoring and calibration drift checks for deployment.

## Reproducibility
Run command:
```bash
MPLCONFIGDIR=/tmp/matplotlib python3 src/logistic_regression_loan.py
```

Outputs:
- `outputs/all_model_metrics.csv`
- `outputs/threshold_analysis.csv`
- `outputs/final_vif_table.csv`
- `outputs/advanced_run_summary.json`
