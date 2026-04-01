# LoanLens
Comparative Analysis of Machine Learning Algorithms for Personal Loan Prediction

## Decision Tree Example

This folder contains a self-contained Decision Tree classification workflow for personal loan prediction.

### Files

- `Bank_Personal_Loan_Modelling.csv`
- `decision_tree_loan.py`
- `decision_tree_loan.ipynb`
- `requirements.txt`

### Install dependencies

```bash
python3 -m pip install -r decision_tree/requirements.txt
```

### Run the model

```bash
python3 decision_tree/decision_tree_loan.py
```

### What the script does

- Loads the dataset
- Drops `ID` and `ZIP Code`
- Uses `Personal Loan` as the target column
- Splits the data into training and testing sets
- Trains a Decision Tree classifier
- Prints training and testing accuracy
- Prints the test confusion matrix and classification report
- Prints top feature importances
- Saves all charts into clustered train/test output folders

### Output structure

Generated files are saved under:

- `output/train/confusion_matrix/`
- `output/train/metrics_summary/`
- `output/test/confusion_matrix/`
- `output/test/metrics_summary/`
- `output/test/feature_importance/`
- `output/test/decision_tree_plot/`
- `output/test/accuracy_vs_max_depth/`
- `output/test/class_distribution/`

### Generated charts

- Train confusion matrix
- Train metrics summary
- Test confusion matrix
- Test metrics summary
- Feature importance bar chart
- Decision tree plot
- Accuracy vs max depth chart
- Target class distribution chart
