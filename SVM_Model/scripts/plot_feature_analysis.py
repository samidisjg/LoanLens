from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("dataset/Bank_Personal_Loan_Modelling.csv")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["Experience"] = df["Experience"].clip(lower=0)

    feature_columns = [
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

    corr = df[feature_columns].corr(numeric_only=True)[["Personal Loan"]].sort_values(
        by="Personal Loan", ascending=False
    )

    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation of Features with Personal Loan")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_correlation_heatmap.png", dpi=200)
    plt.close(fig)

    top_features = ["Income", "CCAvg", "Mortgage", "Family"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, feature in zip(axes, top_features):
        sns.boxplot(data=df, x="Personal Loan", y=feature, ax=ax)
        ax.set_title(f"{feature} vs Personal Loan")
        ax.set_xlabel("Personal Loan")
        ax.set_ylabel(feature)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_feature_boxplots.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Income",
        y="CCAvg",
        hue="Personal Loan",
        alpha=0.7,
        palette="coolwarm",
        ax=ax,
    )
    ax.set_title("Income vs CCAvg by Personal Loan")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "income_ccavg_scatter.png", dpi=200)
    plt.close(fig)

    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
