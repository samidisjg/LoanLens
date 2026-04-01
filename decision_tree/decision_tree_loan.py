from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Bank_Personal_Loan_Modelling.csv"
TARGET_COLUMN = "Personal Loan"
OUTPUT_DIR = BASE_DIR / "output"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Remove columns that identify a customer rather than describe loan behavior.
    drop_columns = ["ID", "ZIP Code"]
    df = df.drop(columns=drop_columns, errors="ignore")

    return df


def train_decision_tree(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

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


def get_next_versioned_path(split: str, plot_name: str) -> Path:
    subdir = OUTPUT_DIR / split / plot_name
    subdir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        candidate = subdir / f"{plot_name}_v{version}.png"
        if not candidate.exists():
            return candidate
        version += 1


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

    output_path = get_next_versioned_path("test", "accuracy_vs_max_depth")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved max depth comparison chart to: {output_path}")
    plt.close()


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
    output_path = get_next_versioned_path(split, "confusion_matrix")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {split} confusion matrix chart to: {output_path}")
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
    output_path = get_next_versioned_path(split, "metrics_summary")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {split} metrics summary chart to: {output_path}")
    plt.close()


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
    output_path = get_next_versioned_path("test", "class_distribution")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved class distribution chart to: {output_path}")
    plt.close()


def plot_feature_importance(feature_importance: pd.Series) -> None:
    top_features = feature_importance.head(10).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(top_features.index, top_features.values, color="steelblue")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()

    output_path = get_next_versioned_path("test", "feature_importance")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved feature importance chart to: {output_path}")
    plt.close()


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
    output_path = get_next_versioned_path("test", "decision_tree_plot")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved decision tree plot to: {output_path}")
    plt.close()


def main() -> None:
    df = load_data(DATASET_PATH)
    train_decision_tree(df)


if __name__ == "__main__":
    main()
