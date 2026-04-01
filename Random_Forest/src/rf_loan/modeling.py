import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from .config import RandomForestConfig
from .preprocessing import clean_dataset


def build_model(config: RandomForestConfig) -> GridSearchCV:
    estimator = RandomForestClassifier(random_state=config.random_state, n_jobs=1)
    return GridSearchCV(estimator=estimator, param_grid=config.param_grid, cv=config.cv_folds, scoring=config.scoring, n_jobs=1)


def _compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'average_precision': float(average_precision_score(y_true, y_proba)),
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, model: GridSearchCV, config: RandomForestConfig) -> dict:
    X = clean_dataset(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    model.fit(X_train, y_train)

    best_model = model.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_proba_train = best_model.predict_proba(X_train)[:, 1]
    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)[:, 1]

    train_metrics = _compute_metrics(y_train, y_pred_train, y_proba_train)
    test_metrics = _compute_metrics(y_test, y_pred_test, y_proba_test)

    reports_payload = {
        'best_params': model.best_params_,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }
    (config.reports_dir / 'metrics_rf.json').write_text(json.dumps(reports_payload, indent=2), encoding='utf-8')
    (config.reports_dir / 'classification_report_rf.txt').write_text(classification_report(y_test, y_pred_test), encoding='utf-8')
    (config.reports_dir / 'train_test_comparison_rf.json').write_text(
        json.dumps({'train_metrics': train_metrics, 'test_metrics': test_metrics}, indent=2),
        encoding='utf-8',
    )

    joblib.dump(best_model, config.model_dir / 'random_forest_model.joblib')
    (config.metadata_dir / 'feature_columns.json').write_text(json.dumps({'feature_columns': list(X.columns)}, indent=2), encoding='utf-8')
    (config.metadata_dir / 'best_params_rf.json').write_text(json.dumps(model.best_params_, indent=2), encoding='utf-8')
    (config.metadata_dir / 'run_summary_rf.json').write_text(
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

    return {
        'best_model': best_model,
        'best_params': model.best_params_,
        'metrics': test_metrics,
        'train_metrics': train_metrics,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_test,
        'y_proba': y_proba_test,
    }
