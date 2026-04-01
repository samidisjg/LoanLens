from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import learning_curve, validation_curve


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Greens', ax=ax)
    ax.set_title('Confusion Matrix - Random Forest')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_roc_curve(y_true, y_score, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title('ROC Curve - Random Forest')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_precision_recall_curve(y_true, y_score, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title('Precision-Recall Curve - Random Forest')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_target_correlation_bar(df: pd.DataFrame, target_column: str, output_path: Path) -> None:
    numeric_df = df.copy()
    numeric_df['Experience'] = numeric_df['Experience'].clip(lower=0)
    corr = numeric_df.select_dtypes(include='number').corr(numeric_only=True)[target_column].drop(target_column).sort_values(key=lambda s: s.abs(), ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette('crest', n_colors=len(corr))
    sns.barplot(x=corr.values, y=corr.index, hue=corr.index, dodge=False, palette=palette, legend=False, ax=ax)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title('Feature Correlation with Personal Loan')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_feature_importance_plots(model, X_test: pd.DataFrame, y_test, output_dir: Path, random_state: int) -> None:
    importance_df = pd.DataFrame({'feature': X_test.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), x='importance', y='feature', hue='feature', dodge=False, palette='viridis', legend=False, ax=ax)
    ax.set_title('Top 10 Feature Importances - Random Forest')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(output_dir / 'feature_importance_rf.png', dpi=220)
    plt.close(fig)

    perm = permutation_importance(estimator=model, X=X_test, y=y_test, n_repeats=10, random_state=random_state, scoring='f1', n_jobs=1)
    perm_df = pd.DataFrame({'feature': X_test.columns, 'importance': perm.importances_mean}).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=perm_df.head(10), x='importance', y='feature', hue='feature', dodge=False, palette='magma', legend=False, ax=ax)
    ax.set_title('Top 10 Permutation Importances - Random Forest')
    ax.set_xlabel('Mean Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(output_dir / 'permutation_importance_rf.png', dpi=220)
    plt.close(fig)


def save_learning_curve_plot(model, X: pd.DataFrame, y, output_path: Path, random_state: int) -> None:
    estimator = clone(model)
    train_sizes = np.linspace(0.2, 1.0, 5)
    train_sizes_abs, train_scores, valid_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        scoring='f1',
        train_sizes=train_sizes,
        n_jobs=1,
        shuffle=True,
        random_state=random_state,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes_abs, train_scores.mean(axis=1), marker='o', label='Training F1')
    ax.plot(train_sizes_abs, valid_scores.mean(axis=1), marker='o', label='Validation F1')
    ax.fill_between(train_sizes_abs, train_scores.mean(axis=1) - train_scores.std(axis=1), train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(train_sizes_abs, valid_scores.mean(axis=1) - valid_scores.std(axis=1), valid_scores.mean(axis=1) + valid_scores.std(axis=1), alpha=0.15)
    ax.set_title('Learning Curve - Random Forest')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('F1 Score')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_validation_curve_plot(model, X: pd.DataFrame, y, output_path: Path) -> None:
    estimator = clone(model)
    if hasattr(estimator, 'class_weight'):
        estimator.set_params(class_weight=None)
    param_range = [50, 100, 150, 200, 300]
    train_scores, valid_scores = validation_curve(
        estimator,
        X,
        y,
        param_name='n_estimators',
        param_range=param_range,
        cv=5,
        scoring='f1',
        n_jobs=1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_range, train_scores.mean(axis=1), marker='o', label='Training F1')
    ax.plot(param_range, valid_scores.mean(axis=1), marker='o', label='Validation F1')
    ax.fill_between(param_range, train_scores.mean(axis=1) - train_scores.std(axis=1), train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(param_range, valid_scores.mean(axis=1) - valid_scores.std(axis=1), valid_scores.mean(axis=1) + valid_scores.std(axis=1), alpha=0.15)
    ax.set_title('Validation Curve - Random Forest (n_estimators)')
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('F1 Score')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
