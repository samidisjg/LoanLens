from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Bank_Personal_Loan_Modelling.csv"
TARGET_COLUMN = "Personal Loan"
OUTPUT_DIR = BASE_DIR / "output"
GENERATED_PLOTS: list[tuple[str, np.ndarray]] = []


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Remove columns that identify a customer rather than describe loan behavior.
    drop_columns = ["ID", "ZIP Code"]
    df = df.drop(columns=drop_columns, errors="ignore")

    return df


def train_decision_tree(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    GENERATED_PLOTS.clear()

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    baseline_metrics = build_metrics(y_test, y_test_pred)

    print("Decision Tree Results")
    print("-" * 50)
    print(f"Training samples : {len(x_train)}")
    print(f"Testing samples  : {len(x_test)}")
    print(f"Train accuracy   : {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test accuracy    : {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nTest Confusion Matrix")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nTest Classification Report")
    print(classification_report(y_test, y_test_pred, digits=4))

    feature_importance = (
        pd.Series(model.feature_importances_, index=x.columns)
        .sort_values(ascending=False)
    )
    print("Top 10 Important Features")
    print(feature_importance.head(10).to_string())

    plot_class_distribution(y)
    plot_split_metrics(y_train, y_train_pred, "train")
    plot_split_metrics(y_test, y_test_pred, "test")
    plot_accuracy_vs_max_depth(x_train, x_test, y_train, y_test)
    plot_feature_importance(feature_importance)
    plot_confusion_matrix(y_train, y_train_pred, "train")
    plot_confusion_matrix(y_test, y_test_pred, "test")
    plot_decision_tree(model, x.columns)
    plot_cross_validation_summary(model, x, y, "baseline")
    compare_model_variants(x_train, x_test, y_train, y_test, baseline_metrics, y_test_pred)
    create_plot_overview()


def build_model() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )


def build_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )["1"]["precision"],
        "Recall": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )["1"]["recall"],
        "F1 Score": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )["1"]["f1-score"],
    }


def compare_model_variants(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    baseline_metrics: dict[str, float],
    baseline_predictions: pd.Series,
) -> None:
    tuned_model, tuned_params = tune_decision_tree(x_train, y_train)
    tuned_test_pred = tuned_model.predict(x_test)
    tuned_metrics = build_metrics(y_test, tuned_test_pred)

    print("\nTuned Decision Tree Best Parameters")
    print(tuned_params)
    print("\nTuned Test Classification Report")
    print(classification_report(y_test, tuned_test_pred, digits=4))
    print(f"Tuned ROC-AUC            : {roc_auc_score(y_test, tuned_model.predict_proba(x_test)[:, 1]):.4f}")
    print(
        f"Tuned Average Precision : "
        f"{average_precision_score(y_test, tuned_model.predict_proba(x_test)[:, 1]):.4f}"
    )

    smote = SMOTE(random_state=42)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    smote_model = build_model()
    smote_model.fit(x_train_smote, y_train_smote)
    smote_test_pred = smote_model.predict(x_test)
    smote_metrics = build_metrics(y_test, smote_test_pred)

    print("\nSMOTE Test Classification Report")
    print(classification_report(y_test, smote_test_pred, digits=4))

    plot_cross_validation_summary(tuned_model, x_train, y_train, "tuned")
    plot_roc_curve_chart(y_test, tuned_model.predict_proba(x_test)[:, 1])
    plot_precision_recall_curve_chart(y_test, tuned_model.predict_proba(x_test)[:, 1])
    plot_model_comparison(
        baseline_metrics,
        smote_metrics,
        tuned_metrics,
    )
    plot_confusion_matrix_comparison(
        y_test,
        baseline_predictions,
        smote_test_pred,
        tuned_test_pred,
    )


def tune_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[DecisionTreeClassifier, dict[str, object]]:
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [4, 5, 6, 7, 8, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_


def get_next_root_versioned_path(plot_name: str) -> Path:
    subdir = OUTPUT_DIR / plot_name
    subdir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        candidate = subdir / f"{plot_name}_v{version}.png"
        if not candidate.exists():
            return candidate
        version += 1


def capture_plot(label: str, fig: plt.Figure) -> None:
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba()).copy()
    GENERATED_PLOTS.append((label, image))


def create_plot_overview() -> None:
    if not GENERATED_PLOTS:
        return

    num_plots = len(GENERATED_PLOTS)
    cols = 2
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4.5))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (label, image) in zip(axes, GENERATED_PLOTS):
        ax.imshow(image)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    for ax in axes[num_plots:]:
        ax.axis("off")

    plt.suptitle("Decision Tree Plot Overview", fontsize=14)
    plt.tight_layout()
    output_path = get_next_root_versioned_path("all_plots_overview")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined plot overview to: {output_path}")
    plt.close(fig)


def plot_accuracy_vs_max_depth(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    depths = list(range(1, 16))
    train_scores = []
    test_scores = []

    for depth in depths:
        candidate_model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        )
        candidate_model.fit(x_train, y_train)
        train_scores.append(candidate_model.score(x_train, y_train))
        test_scores.append(candidate_model.score(x_test, y_test))

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_scores, marker="o", label="Training Accuracy")
    plt.plot(depths, test_scores, marker="o", label="Testing Accuracy")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Max Depth")
    plt.xticks(depths)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    capture_plot("Accuracy vs Max Depth", fig)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    split: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    plt.colorbar(image, ax=ax)

    ax.set_title(f"{split.title()} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Loan", "Loan"])
    ax.set_yticklabels(["No Loan", "Loan"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    capture_plot(f"{split.title()} Confusion Matrix", fig)
    plt.close(fig)


def plot_split_metrics(y_true: pd.Series, y_pred: pd.Series, split: str) -> None:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
    }

    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"{split.title()} Metrics")

    for index, value in enumerate(metrics.values()):
        plt.text(index, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()
    fig = plt.gcf()
    capture_plot(f"{split.title()} Metrics Summary", fig)
    plt.close(fig)


def plot_cross_validation_summary(
    model: DecisionTreeClassifier,
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    cv_results = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=-1)

    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    metric_keys = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    means = [cv_results[key].mean() for key in metric_keys]
    stds = [cv_results[key].std() for key in metric_keys]

    plt.figure(figsize=(8, 5))
    plt.bar(metric_labels, means, yerr=stds, capsize=5, color="#4C78A8")
    plt.ylim(0, 1.05)
    plt.ylabel("Cross-Validation Score")
    plt.title(f"{model_name.title()} 5-Fold Cross-Validation Summary")

    for index, value in enumerate(means):
        plt.text(index, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()
    fig = plt.gcf()
    capture_plot(f"{model_name.title()} Cross Validation", fig)
    plt.close(fig)


def plot_model_comparison(
    baseline_metrics: dict[str, float],
    smote_metrics: dict[str, float],
    tuned_metrics: dict[str, float],
) -> None:
    metric_names = list(baseline_metrics.keys())
    baseline_values = [baseline_metrics[name] for name in metric_names]
    smote_values = [smote_metrics[name] for name in metric_names]
    tuned_values = [tuned_metrics[name] for name in metric_names]
    positions = range(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_bars = ax.bar(
        [position - width for position in positions],
        baseline_values,
        width=width,
        label="Baseline",
        color="#4C78A8",
    )
    smote_bars = ax.bar(
        list(positions),
        smote_values,
        width=width,
        label="SMOTE",
        color="#F58518",
    )
    tuned_bars = ax.bar(
        [position + width for position in positions],
        tuned_values,
        width=width,
        label="Tuned",
        color="#54A24B",
    )
    ax.set_xticks(list(positions), metric_names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs SMOTE vs Tuned Metrics")
    ax.legend()

    for bars in [baseline_bars, smote_bars, tuned_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height * 100:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plt.tight_layout()
    capture_plot("Model Metrics Comparison", fig)
    plt.close(fig)


def plot_confusion_matrix_comparison(
    y_true: pd.Series,
    baseline_pred: pd.Series,
    smote_pred: pd.Series,
    tuned_pred: pd.Series,
) -> None:
    baseline_cm = confusion_matrix(y_true, baseline_pred)
    smote_cm = confusion_matrix(y_true, smote_pred)
    tuned_cm = confusion_matrix(y_true, tuned_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, matrix, title in zip(
        axes,
        [baseline_cm, smote_cm, tuned_cm],
        [
            "Baseline Confusion Matrix",
            "SMOTE Confusion Matrix",
            "Tuned Confusion Matrix",
        ],
    ):
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Loan", "Loan"])
        ax.set_yticklabels(["No Loan", "Loan"])

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    capture_plot("Model Confusion Matrix Comparison", fig)
    plt.close(fig)


def plot_roc_curve_chart(y_true: pd.Series, y_scores: pd.Series) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Tuned Model (AUC = {auc_score:.3f})", color="#4C78A8")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    capture_plot("ROC Curve", fig)
    plt.close(fig)


def plot_precision_recall_curve_chart(y_true: pd.Series, y_scores: pd.Series) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#F58518", label=f"Tuned Model (AP = {avg_precision:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    capture_plot("Precision-Recall Curve", fig)
    plt.close(fig)


def plot_class_distribution(y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    labels = ["No Loan", "Loan"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts.values, color=["#4C78A8", "#F58518"])
    plt.ylabel("Number of Customers")
    plt.title("Target Class Distribution")

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 25,
            str(value),
            ha="center",
        )

    plt.tight_layout()
    fig = plt.gcf()
    capture_plot("Class Distribution", fig)
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.Series) -> None:
    top_features = feature_importance.head(10).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(top_features.index, top_features.values, color="steelblue")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()

    fig = plt.gcf()
    capture_plot("Feature Importance", fig)
    plt.close(fig)


def plot_decision_tree(model: DecisionTreeClassifier, feature_names: pd.Index) -> None:
    plt.figure(figsize=(22, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No Loan", "Loan"],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Decision Tree for Personal Loan Prediction")
    plt.tight_layout()
    fig = plt.gcf()
    capture_plot("Decision Tree Plot", fig)
    plt.close(fig)


def main() -> None:
    df = load_data(DATASET_PATH)
    train_decision_tree(df)


if __name__ == "__main__":
    main()