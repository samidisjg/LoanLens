import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'smote_svm'
FIGURES_DIR = OUTPUT_DIR / 'figures'
REPORTS_DIR = OUTPUT_DIR / 'reports'
ARTIFACT_MODEL_DIR = PROJECT_ROOT / 'artifacts' / 'smote_svm' / 'model'
ARTIFACT_METADATA_DIR = PROJECT_ROOT / 'artifacts' / 'smote_svm' / 'metadata'
for p in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, ARTIFACT_MODEL_DIR, ARTIFACT_METADATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
    }


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / 'dataset' / 'Bank_Personal_Loan_Modelling.csv')
    df['Experience'] = df['Experience'].clip(lower=0)
    X = df.drop(columns=['Personal Loan', 'ID'])
    y = df['Personal Loan']
    numeric_features = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']
    categorical_features = ['ZIP Code', 'Family', 'Education', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    preprocessor = ColumnTransformer(transformers=[
        ('num', ImbPipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = ImbPipeline(steps=[('preprocessor', preprocessor), ('smote', SMOTE(random_state=42)), ('classifier', SVC(probability=True))])
    param_grid = {
        'classifier__kernel': ['rbf'],
        'classifier__C': [0.5, 1, 2],
        'classifier__gamma': ['scale', 0.01],
        'smote__k_neighbors': [3],
    }
    search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='f1', n_jobs=1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_proba_train = best_model.predict_proba(X_train)[:, 1]
    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    train_metrics = compute_metrics(y_train, y_pred_train, y_proba_train)
    test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test)

    (REPORTS_DIR / 'metrics_smote_svm.json').write_text(json.dumps({'best_params': search.best_params_, 'train_metrics': train_metrics, 'test_metrics': test_metrics}, indent=2), encoding='utf-8')
    (REPORTS_DIR / 'classification_report_smote_svm.txt').write_text(classification_report(y_test, y_pred_test), encoding='utf-8')
    joblib.dump(best_model, ARTIFACT_MODEL_DIR / 'smote_svm_model.joblib')
    (ARTIFACT_METADATA_DIR / 'feature_columns_smote_svm.json').write_text(json.dumps({'feature_columns': list(X.columns)}, indent=2), encoding='utf-8')
    (ARTIFACT_METADATA_DIR / 'best_params_smote_svm.json').write_text(json.dumps(search.best_params_, indent=2), encoding='utf-8')
    (ARTIFACT_METADATA_DIR / 'run_summary_smote_svm.json').write_text(json.dumps({
        'target_column': 'Personal Loan', 'drop_columns': ['ID'], 'feature_columns': list(X.columns),
        'test_size': 0.2, 'random_state': 42, 'best_params': search.best_params_,
        'train_metrics': train_metrics, 'test_metrics': test_metrics,
    }, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, cmap='Oranges', ax=ax)
    ax.set_title('Confusion Matrix - SMOTE + SVM')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'confusion_matrix_smote_svm.png', dpi=200); plt.close(fig)

    compare_df = pd.DataFrame({'Metric': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'] * 2, 'Value': [train_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']] + [test_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']], 'Split': ['Train'] * 5 + ['Test'] * 5})
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=compare_df, x='Metric', y='Value', hue='Split', palette='Set2', ax=ax)
    ax.set_title('Train vs Test Metrics - SMOTE + SVM'); ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'train_test_metrics_smote_svm.png', dpi=200); plt.close(fig)

    print('Best params:', search.best_params_)
    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)
    print('Saved to:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
