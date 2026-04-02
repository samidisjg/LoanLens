import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.rf_loan.analysis import save_dataset_analysis
from src.rf_loan.config import RandomForestConfig
from src.rf_loan.data import load_dataset, split_features_target
from src.rf_loan.reporting import save_dataset_summary
from src.rf_loan.visualization import save_confusion_matrix, save_precision_recall_curve, save_roc_curve


OUTPUT_DIR = Path('outputs/smote_rf')
FIGURES_DIR = OUTPUT_DIR / 'figures'
REPORTS_DIR = OUTPUT_DIR / 'reports'
ARTIFACTS_DIR = Path('artifacts/smote_rf')
MODEL_DIR = ARTIFACTS_DIR / 'model'
METADATA_DIR = ARTIFACTS_DIR / 'metadata'


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if 'Experience' in cleaned.columns:
        cleaned['Experience'] = cleaned['Experience'].clip(lower=0)
    return cleaned


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'average_precision': float(average_precision_score(y_true, y_proba)),
    }


def main() -> None:
    for path in [FIGURES_DIR, REPORTS_DIR, MODEL_DIR, METADATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    config = RandomForestConfig()
    df = load_dataset(config)
    X, y = split_features_target(df, config)
    X = clean_dataset(X)

    save_dataset_summary(df=df, target_column=config.target_column, output_path=REPORTS_DIR / 'dataset_summary_smote_rf.txt')
    save_dataset_analysis(df=df, target_column=config.target_column, output_dir=FIGURES_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[('passthrough', 'passthrough', list(X.columns))],
        remainder='drop',
    )

    pipeline = ImbPipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=config.random_state)),
            ('classifier', RandomForestClassifier(random_state=config.random_state, n_jobs=1)),
        ]
    )

    param_grid = {
        'smote__k_neighbors': [3, 5],
        'classifier__n_estimators': [200],
        'classifier__max_depth': [6, 7],
        'classifier__min_samples_split': [15, 20],
        'classifier__min_samples_leaf': [6, 8],
        'classifier__max_features': ['sqrt'],
        'classifier__class_weight': [None, 'balanced'],
    }

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=config.cv_folds,
        scoring=config.scoring,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    best_model = model.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_proba_train = best_model.predict_proba(X_train)[:, 1]
    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)[:, 1]

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba_train)
    test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test)

    payload = {
        'best_params': model.best_params_,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }
    (REPORTS_DIR / 'metrics_smote_rf.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (REPORTS_DIR / 'classification_report_smote_rf.txt').write_text(classification_report(y_test, y_pred_test), encoding='utf-8')
    (REPORTS_DIR / 'train_test_comparison_smote_rf.json').write_text(
        json.dumps({'train_metrics': train_metrics, 'test_metrics': test_metrics}, indent=2),
        encoding='utf-8',
    )

    save_confusion_matrix(y_test, y_pred_test, FIGURES_DIR / 'confusion_matrix_smote_rf.png')
    save_roc_curve(y_test, y_proba_test, FIGURES_DIR / 'roc_curve_smote_rf.png')
    save_precision_recall_curve(y_test, y_proba_test, FIGURES_DIR / 'precision_recall_curve_smote_rf.png')

    joblib.dump(best_model, MODEL_DIR / 'smote_random_forest_model.joblib')
    (METADATA_DIR / 'feature_columns_smote_rf.json').write_text(
        json.dumps({'feature_columns': list(X.columns)}, indent=2), encoding='utf-8'
    )
    (METADATA_DIR / 'best_params_smote_rf.json').write_text(json.dumps(model.best_params_, indent=2), encoding='utf-8')
    (METADATA_DIR / 'run_summary_smote_rf.json').write_text(
        json.dumps(
            {
                'target_column': config.target_column,
                'drop_columns': config.drop_columns,
                'feature_columns': list(X.columns),
                'test_size': config.test_size,
                'random_state': config.random_state,
                'best_params': model.best_params_,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    print('SMOTE + Random Forest training complete.')
    print(f'Best parameters: {model.best_params_}')
    print(f'Train metrics: {train_metrics}')
    print(f'Test metrics: {test_metrics}')
    print(f'Outputs saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
