import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = "dataset/Bank_Personal_Loan_Modelling.csv"
TARGET = "Personal Loan"
FIGURES_DIR = Path("figures")

# Use a non-interactive backend so plots can be generated in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ModelResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray
    report: str


def load_and_preprocess(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Domain fix: negative years of experience are not realistic.
    df["Experience"] = df["Experience"].clip(lower=0)

    return df


def build_pipeline() -> Pipeline:
    numeric_features = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]
    categorical_features = [
        "Family",
        "Education",
        "Securities Account",
        "CD Account",
        "Online",
        "CreditCard",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=3000,
        class_weight=None,
        C=1.0,
        solver="liblinear",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_predictions(
    y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray
) -> ModelResult:
    return ModelResult(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
        confusion=confusion_matrix(y_true, y_pred),
        report=classification_report(y_true, y_pred, digits=4, zero_division=0),
    )


def evaluate_model(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> ModelResult:
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return evaluate_predictions(y_test, y_pred, y_prob)


def run_setting_experiments(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    settings = [
        {"name": "baseline", "C": 1.0, "class_weight": None, "threshold": 0.50},
        {
            "name": "stronger_reg",
            "C": 0.1,
            "class_weight": None,
            "threshold": 0.50,
        },
        {"name": "weaker_reg", "C": 5.0, "class_weight": None, "threshold": 0.50},
        {
            "name": "balanced_weight",
            "C": 1.0,
            "class_weight": "balanced",
            "threshold": 0.50,
        },
        {
            "name": "threshold_0.35",
            "C": 1.0,
            "class_weight": None,
            "threshold": 0.35,
        },
    ]

    rows = []
    for setting in settings:
        pipeline = build_pipeline()
        pipeline.set_params(
            model__C=setting["C"], model__class_weight=setting["class_weight"]
        )
        pipeline.fit(x_train, y_train)
        result = evaluate_model(
            pipeline, x_test, y_test, threshold=float(setting["threshold"])
        )
        rows.append(
            {
                "setting": setting["name"],
                "C": setting["C"],
                "class_weight": setting["class_weight"] or "None",
                "threshold": setting["threshold"],
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )

    return pd.DataFrame(rows).sort_values(by="f1", ascending=False)


def run_training_size_experiments(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[TARGET, "ID", "ZIP Code"])
    y = df[TARGET]
    train_sizes = [0.5, 0.6, 0.7, 0.8]
    rows = []

    for train_ratio in train_sizes:
        test_size = 1.0 - train_ratio
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        pipeline = build_pipeline()
        pipeline.fit(x_train, y_train)
        result = evaluate_model(pipeline, x_test, y_test, threshold=0.5)
        rows.append(
            {
                "train_ratio": train_ratio,
                "train_samples": len(x_train),
                "test_samples": len(x_test),
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )

    return pd.DataFrame(rows)


def save_correlation_heatmap(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.drop(columns=["ID"]).corr(numeric_only=True)
    out_path = FIGURES_DIR / "correlation_heatmap.png"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.2,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_confusion_matrix_heatmap(confusion: np.ndarray) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "confusion_matrix_heatmap.png"
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    annotated = np.array(
        [[f"{labels[i, j]}\n{confusion[i, j]}" for j in range(2)] for i in range(2)]
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=annotated,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_settings_metrics_heatmap(settings_df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "settings_metrics_heatmap.png"
    heatmap_df = settings_df.set_index("setting")[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ]

    plt.figure(figsize=(8, 4.8))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.2,
        cbar_kws={"shrink": 0.9},
    )
    plt.title("Metrics Across Logistic Regression Settings")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_training_size_lineplot(training_size_df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "training_size_performance.png"

    plot_df = training_size_df.sort_values(by="train_ratio")
    plt.figure(figsize=(8, 4.8))
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        plt.plot(plot_df["train_ratio"], plot_df[metric], marker="o", label=metric)
    plt.xlabel("Train Ratio")
    plt.ylabel("Score")
    plt.title("Performance vs Training Set Ratio")
    plt.ylim(0.45, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = load_and_preprocess(DATA_PATH)

    x = df.drop(columns=[TARGET, "ID", "ZIP Code"])
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    baseline_result = evaluate_model(pipeline, x_test, y_test, threshold=0.5)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(pipeline, x, y, cv=cv, scoring="roc_auc")
    cv_f1 = cross_val_score(pipeline, x, y, cv=cv, scoring="f1")
    settings_df = run_setting_experiments(x_train, x_test, y_train, y_test)
    training_size_df = run_training_size_experiments(df)
    corr_path = save_correlation_heatmap(df)
    cm_path = save_confusion_matrix_heatmap(baseline_result.confusion)
    settings_path = save_settings_metrics_heatmap(settings_df)
    training_path = save_training_size_lineplot(training_size_df)

    print("=== Logistic Regression (Personal Loan Prediction) ===")
    print(f"Dataset shape: {df.shape}")
    print("Target distribution:")
    print(y.value_counts().to_string())
    print("\nTest metrics:")
    print(f"Accuracy : {baseline_result.accuracy:.4f}")
    print(f"Precision: {baseline_result.precision:.4f}")
    print(f"Recall   : {baseline_result.recall:.4f}")
    print(f"F1-score : {baseline_result.f1:.4f}")
    print(f"ROC-AUC  : {baseline_result.roc_auc:.4f}")
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(baseline_result.confusion)
    print("\nClassification report:")
    print(baseline_result.report)
    print(f"5-fold CV ROC-AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
    print(f"5-fold CV F1-score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

    print("\n=== Metric Changes Across Training Settings ===")
    print(
        settings_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
            },
        )
    )

    print("\n=== Effect of Training Set Size (Baseline Settings) ===")
    print(
        training_size_df.to_string(
            index=False,
            formatters={
                "train_ratio": "{:.2f}".format,
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
            },
        )
    )
    print("\n=== Saved Visualizations ===")
    print(corr_path.as_posix())
    print(cm_path.as_posix())
    print(settings_path.as_posix())
    print(training_path.as_posix())


if __name__ == "__main__":
    main()
