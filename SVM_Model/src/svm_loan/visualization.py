from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import learning_curve, validation_curve


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix - SVM')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_curve(y_true, y_score, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title('ROC Curve - SVM')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_feature_importance(model, X_test: pd.DataFrame, y_test, output_path: Path, random_state: int) -> None:
    result = permutation_importance(estimator=model, X=X_test, y=y_test, n_repeats=10, random_state=random_state, scoring='f1', n_jobs=1)
    importance_df = pd.DataFrame({'feature': X_test.columns, 'importance_mean': result.importances_mean}).sort_values('importance_mean', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance_mean', y='feature', hue='feature', dodge=False, legend=False, palette='viridis', ax=ax)
    ax.set_title('Permutation Importance of Original Features')
    ax.set_xlabel('Mean Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_learning_curve_plot(model, X_train: pd.DataFrame, y_train, output_path: Path, random_state: int) -> None:
    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy',
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=1,
        shuffle=True,
        random_state=random_state,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='Training Accuracy')
    ax.plot(train_sizes, valid_scores.mean(axis=1), marker='o', label='Validation Accuracy')
    ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1), train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(train_sizes, valid_scores.mean(axis=1) - valid_scores.std(axis=1), valid_scores.mean(axis=1) + valid_scores.std(axis=1), alpha=0.15)
    ax.set_title('Learning Curve - SVM')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_validation_curve_plot(model, X_train: pd.DataFrame, y_train, output_path: Path) -> None:
    param_range = [0.1, 1, 10, 50]
    train_scores, valid_scores = validation_curve(
        model,
        X_train,
        y_train,
        param_name='classifier__C',
        param_range=param_range,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(param_range, train_scores.mean(axis=1), marker='o', label='Training Accuracy')
    ax.plot(param_range, valid_scores.mean(axis=1), marker='o', label='Validation Accuracy')
    ax.fill_between(param_range, train_scores.mean(axis=1) - train_scores.std(axis=1), train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(param_range, valid_scores.mean(axis=1) - valid_scores.std(axis=1), valid_scores.mean(axis=1) + valid_scores.std(axis=1), alpha=0.15)
    ax.set_title('Validation Curve - SVM (C)')
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_train_test_metrics_plot(train_metrics: dict, test_metrics: dict, output_path: Path) -> None:
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    plot_df = pd.DataFrame({
        'Metric': metric_names * 2,
        'Value': [train_metrics[name] for name in metric_names] + [test_metrics[name] for name in metric_names],
        'Split': ['Train'] * len(metric_names) + ['Test'] * len(metric_names),
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Metric', y='Value', hue='Split', palette='Set2', ax=ax)
    ax.set_title('Train vs Test Metrics - SVM')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
