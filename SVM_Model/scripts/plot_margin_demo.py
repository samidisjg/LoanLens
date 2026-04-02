from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = Path("dataset/Bank_Personal_Loan_Modelling.csv")
OUTPUT_PATH = Path("outputs/svm_large_margin_plot.png")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["Experience"] = df["Experience"].clip(lower=0)

    features = ["Income", "CCAvg"]
    X = df[features].copy()
    y = df["Personal Loan"].copy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=0.1, class_weight="balanced")),
        ]
    )
    model.fit(X, y)

    scaler = model.named_steps["scaler"]
    svm = model.named_steps["svm"]
    X_scaled = scaler.transform(X)

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    decision_values = svm.decision_function(grid).reshape(xx.shape)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xx, yy, decision_values, levels=30, cmap="coolwarm", alpha=0.25)
    ax.contour(
        xx,
        yy,
        decision_values,
        levels=[-1, 0, 1],
        colors=["black", "navy", "black"],
        linestyles=["--", "-", "--"],
        linewidths=[1.5, 2.5, 1.5],
    )

    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="k",
        s=40,
        alpha=0.8,
    )
    ax.scatter(
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=140,
        facecolors="none",
        edgecolors="gold",
        linewidths=1.8,
        label="Support Vectors",
    )

    handles, labels = scatter.legend_elements()
    legend1 = ax.legend(handles, ["No Loan (0)", "Loan (1)"], title="Class", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(loc="upper right")

    ax.set_title("Linear SVM Large-Margin Plot Using Income and CCAvg")
    ax.set_xlabel("Income (standardized)")
    ax.set_ylabel("CCAvg (standardized)")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)

    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
