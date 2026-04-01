from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Bank_Personal_Loan_Modelling.csv"
TARGET_COLUMN = "Personal Loan"
OUTPUT_DIR = BASE_DIR / "outputs"


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
    y_pred = model.predict(x_test)

    print("Decision Tree Results")
    print("-" * 50)
    print(f"Training samples : {len(x_train)}")
    print(f"Testing samples  : {len(x_test)}")
    print(f"Accuracy         : {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred, digits=4))

    feature_importance = (
        pd.Series(model.feature_importances_, index=x.columns)
        .sort_values(ascending=False)
    )
    print("Top 10 Important Features")
    print(feature_importance.head(10).to_string())

    plot_confusion_matrix(y_test, y_pred)
    plot_decision_tree(model, x.columns)


def get_next_versioned_path(prefix: str) -> Path:
    version = 1
    while True:
        candidate = OUTPUT_DIR / f"{prefix}_v{version}.png"
        if not candidate.exists():
            return candidate
        version += 1


def plot_confusion_matrix(y_test: pd.Series, y_pred: pd.Series) -> None:
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    plt.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
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
    output_path = get_next_versioned_path("confusion_matrix")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved confusion matrix chart to: {output_path}")
    plt.show()


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
    output_path = get_next_versioned_path("decision_tree_plot")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved decision tree plot to: {output_path}")
    plt.show()


def main() -> None:
    df = load_data(DATASET_PATH)
    train_decision_tree(df)


if __name__ == "__main__":
    main()
