# LoanLens
Comparative Analysis of Machine Learning Algorithms for Personal Loan Prediction

## Decision Tree Example

This project includes a basic Decision Tree classification script for the dataset:

- `Bank_Personal_Loan_Modelling.csv`
- `decision_tree/Bank_Personal_Loan_Modelling.csv`
- `decision_tree/decision_tree_loan.py`
- `decision_tree/decision_tree_loan.ipynb`

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
- Prints accuracy, confusion matrix, classification report, and top feature importances
