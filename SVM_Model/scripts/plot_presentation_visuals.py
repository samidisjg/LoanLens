from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = Path("dataset/Bank_Personal_Loan_Modelling.csv")
OUTPUT_DIR = Path("outputs")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Experience"] = df["Experience"].clip(lower=0)
    return df


def save_target_correlation_bar(df: pd.DataFrame) -> None:
    cols = [
        "Age",
        "Experience",
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "Securities Account",
        "CD Account",
        "Online",
        "CreditCard",
        "Personal Loan",
    ]
    corr = (
        df[cols]
        .corr(numeric_only=True)["Personal Loan"]
        .drop("Personal Loan")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=corr.values, y=corr.index, palette="Blues_r", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Feature Correlation with Personal Loan")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "presentation_correlation_bar.png", dpi=220)
    plt.close(fig)


def save_class_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Personal Loan", palette="Set2", ax=ax)
    ax.set_title("Class Distribution of Personal Loan")
    ax.set_xlabel("Personal Loan")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "presentation_class_distribution.png", dpi=220)
    plt.close(fig)


def save_top_feature_distributions(df: pd.DataFrame) -> None:
    top_features = ["Income", "CCAvg", "Mortgage", "CD Account"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, feature in zip(axes, top_features):
        if df[feature].nunique() <= 5:
            sns.countplot(data=df, x=feature, hue="Personal Loan", palette="coolwarm", ax=ax)
        else:
            sns.kdeplot(data=df, x=feature, hue="Personal Loan", fill=True, common_norm=False, alpha=0.35, ax=ax)
        ax.set_title(f"{feature} by Loan Class")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "presentation_feature_distributions.png", dpi=220)
    plt.close(fig)


def save_clean_margin_plot(df: pd.DataFrame) -> None:
    X = df[["Income", "CCAvg"]].copy()
    y = df["Personal Loan"].copy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=0.2, class_weight="balanced")),
        ]
    )
    model.fit(X, y)

    scaler = model.named_steps["scaler"]
    svm = model.named_steps["svm"]
    X_scaled = scaler.transform(X)

    import numpy as np

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    z = svm.decision_function(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contourf(xx, yy, z, levels=25, cmap="coolwarm", alpha=0.18)
    ax.contour(xx, yy, z, levels=[-1, 0, 1], colors=["black", "navy", "black"], linestyles=["--", "-", "--"])

    sample = df.groupby("Personal Loan", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 180), random_state=42)
    )
    sample_scaled = scaler.transform(sample[["Income", "CCAvg"]])

    scatter = ax.scatter(
        sample_scaled[:, 0],
        sample_scaled[:, 1],
        c=sample["Personal Loan"],
        cmap="coolwarm",
        s=45,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.scatter(
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=120,
        facecolors="none",
        edgecolors="gold",
        linewidths=1.8,
        label="Support Vectors",
    )

    handles, _ = scatter.legend_elements()
    legend1 = ax.legend(handles, ["No Loan", "Loan"], title="Class", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(loc="upper right")

    ax.set_title("Linear SVM Margin Demonstration Using Income and CCAvg")
    ax.set_xlabel("Income (standardized)")
    ax.set_ylabel("CCAvg (standardized)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "presentation_margin_plot.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_data()
    save_target_correlation_bar(df)
    save_class_distribution(df)
    save_top_feature_distributions(df)
    save_clean_margin_plot(df)
    print(f"Saved presentation visuals to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
