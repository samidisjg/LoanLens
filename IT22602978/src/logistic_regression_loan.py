from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# Headless backend for environments without GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("dataset/Bank_Personal_Loan_Modelling.csv")
TARGET = "Personal Loan"
FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("outputs")


@dataclass
class ModelMetrics:
    model_name: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Experience"] = df["Experience"].clip(lower=0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Remove non-generalizable columns.
    work = work.drop(columns=["ID", "ZIP Code"])

    # Feature engineering.
    work["income_per_family"] = work["Income"] / work["Family"].replace(0, 1)
    work["income_x_education"] = work["Income"] * work["Education"]
    work["ccavg_x_income"] = work["CCAvg"] * work["Income"]
    work["mortgage_to_income"] = work["Mortgage"] / (work["Income"] + 1.0)

    work["age_bin"] = pd.cut(
        work["Age"], bins=[20, 30, 40, 50, 60, 80], labels=["20s", "30s", "40s", "50s", "60+"]
    ).astype(str)
    work["income_bin"] = pd.cut(
        work["Income"],
        bins=[0, 40, 80, 120, 170, 300],
        labels=["low", "lower_mid", "upper_mid", "high", "very_high"],
    ).astype(str)

    return work


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return x, y


def drop_high_corr_pair(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], dict[str, Any]]:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    corr = x_train[numeric_cols].corr().abs()

    # Explicitly handle strongly redundant pair Age/Experience.
    dropped = []
    reason = []
    if "Age" in corr.columns and "Experience" in corr.columns:
        age_exp_corr = float(corr.loc["Age", "Experience"])
        if age_exp_corr > 0.95:
            age_target = abs(np.corrcoef(x_train["Age"], y_train)[0, 1])
            exp_target = abs(np.corrcoef(x_train["Experience"], y_train)[0, 1])
            to_drop = "Age" if age_target < exp_target else "Experience"
            dropped.append(to_drop)
            reason.append(
                {
                    "pair": ["Age", "Experience"],
                    "corr": round(age_exp_corr, 4),
                    "dropped": to_drop,
                    "rule": "drop feature with lower |corr with target|",
                }
            )

    kept = [c for c in x_train.columns if c not in dropped]
    return kept, {"manual_drop": dropped, "details": reason}


def compute_vif(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        y_col = frame[col].values
        x_cols = [c for c in cols if c != col]
        if not x_cols:
            rows.append({"feature": col, "vif": 1.0})
            continue
        x_mat = frame[x_cols].values
        model = LinearRegression()
        model.fit(x_mat, y_col)
        r2 = model.score(x_mat, y_col)
        if r2 >= 0.999999:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)
        rows.append({"feature": col, "vif": float(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def vif_filter(
    x_train: pd.DataFrame,
    vif_threshold: float = 10.0,
    protected: set[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]], pd.DataFrame]:
    protected = protected or set()
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    active = numeric_cols.copy()
    dropped_log: list[dict[str, Any]] = []

    while True:
        vif_df = compute_vif(x_train, active)
        worst = vif_df.iloc[0]
        if worst["vif"] <= vif_threshold:
            break
        feature = str(worst["feature"])
        if feature in protected:
            candidates = vif_df[~vif_df["feature"].isin(protected)]
            if candidates.empty or float(candidates.iloc[0]["vif"]) <= vif_threshold:
                break
            feature = str(candidates.iloc[0]["feature"])
            worst_val = float(candidates.iloc[0]["vif"])
        else:
            worst_val = float(worst["vif"])

        active.remove(feature)
        dropped_log.append({"feature": feature, "vif": round(worst_val, 4)})

    kept_cols = [c for c in x_train.columns if c not in {d["feature"] for d in dropped_log}]
    final_vif = compute_vif(x_train, active) if active else pd.DataFrame(columns=["feature", "vif"])
    return kept_cols, dropped_log, final_vif


def build_preprocessor(x_train: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    num_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_transform, numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )


def evaluate_predictions(model_name: str, y_true: pd.Series, probs: np.ndarray, threshold: float) -> ModelMetrics:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    p_curve, r_curve, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(r_curve, p_curve)

    return ModelMetrics(
        model_name=model_name,
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, preds)),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        f1_score=float(f1_score(y_true, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probs)),
        pr_auc=float(pr_auc),
        tn=int(cm[0, 0]),
        fp=int(cm[0, 1]),
        fn=int(cm[1, 0]),
        tp=int(cm[1, 1]),
    )


def optimize_thresholds(y_val: pd.Series, probs_val: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        fpr = ((preds == 1) & (y_val.values == 0)).sum() / max((y_val.values == 0).sum(), 1)
        youden_j = r - fpr
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1, "youden_j": youden_j})

    table = pd.DataFrame(rows)
    best_f1_t = float(table.sort_values(["f1", "recall"], ascending=False).iloc[0]["threshold"])
    best_j_t = float(table.sort_values("youden_j", ascending=False).iloc[0]["threshold"])
    return table, best_f1_t, best_j_t


def plot_confusion_matrices(metrics_list: list[ModelMetrics]) -> Path:
    cols = 3
    rows = int(np.ceil(len(metrics_list) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.0))
    axes = np.array(axes).reshape(-1)

    for ax in axes[len(metrics_list) :]:
        ax.axis("off")

    for ax, m in zip(axes, metrics_list):
        cm = np.array([[m.tn, m.fp], [m.fn, m.tp]])
        ann = np.array([[f"TN\n{m.tn}", f"FP\n{m.fp}"], [f"FN\n{m.fn}", f"TP\n{m.tp}"]])
        sns.heatmap(cm, annot=ann, fmt="", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{m.model_name}\n(thr={m.threshold:.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    out = FIGURES_DIR / "confusion_matrices_lr_variants.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> Path:
    out = FIGURES_DIR / "roc_curves_lr_variants.png"
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_val) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Logistic Regression Variants")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> Path:
    out = FIGURES_DIR / "pr_curves_lr_variants.png"
    plt.figure(figsize=(8, 6))
    for name, (recall_arr, precision_arr, auc_val) in curves.items():
        plt.plot(recall_arr, precision_arr, label=f"{name} (PR-AUC={auc_val:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves: Logistic Regression Variants")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_threshold_curve(table: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "threshold_vs_f1_youden.png"
    plot_df = table.sort_values("threshold")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["threshold"], plot_df["f1"], label="F1-score")
    plt.plot(plot_df["threshold"], plot_df["youden_j"], label="Youden's J")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Optimization Curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_calibration_curves(y_true: pd.Series, prob_map: dict[str, np.ndarray]) -> Path:
    out = FIGURES_DIR / "calibration_curves.png"
    plt.figure(figsize=(8, 6))
    for name, probs in prob_map.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    return preprocessor.get_feature_names_out()


def plot_lr_coefficients(lr_pipe: Pipeline, top_n: int = 15) -> Path:
    out = FIGURES_DIR / "lr_feature_importance.png"

    lr_names = get_feature_names(lr_pipe.named_steps["preprocessor"])
    lr_coef = lr_pipe.named_steps["clf"].coef_[0]
    lr_df = pd.DataFrame({"feature": lr_names, "importance": np.abs(lr_coef)})
    lr_df = lr_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.barplot(data=lr_df.sort_values("importance"), x="importance", y="feature", ax=ax, color="#2A7FFF")
    ax.set_title("Logistic Regression |coef|")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    return out


def to_dict(m: ModelMetrics) -> dict[str, Any]:
    return {
        "model": m.model_name,
        "threshold": m.threshold,
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1_score": m.f1_score,
        "roc_auc": m.roc_auc,
        "pr_auc": m.pr_auc,
        "tn": m.tn,
        "fp": m.fp,
        "fn": m.fn,
        "tp": m.tp,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()

    raw_df = load_data()
    df = engineer_features(raw_df)
    x, y = split_xy(df)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
    )

    # Multicollinearity handling: manual age/experience handling + VIF pruning.
    kept_after_manual, manual_info = drop_high_corr_pair(x_train, y_train)
    x_train_m = x_train[kept_after_manual]
    x_val_m = x_val[kept_after_manual]
    x_test_m = x_test[kept_after_manual]

    kept_after_vif, vif_dropped, final_vif = vif_filter(
        x_train_m,
        vif_threshold=10.0,
        protected={"Income", "CCAvg", "Mortgage"},
    )
    x_train_f = x_train_m[kept_after_vif]
    x_val_f = x_val_m[kept_after_vif]
    x_test_f = x_test_m[kept_after_vif]

    # Preprocessor
    pre_lr = build_preprocessor(x_train_f, scale_numeric=True)

    # 1) Baseline Logistic Regression
    baseline_lr = Pipeline(
        steps=[
            ("preprocessor", pre_lr),
            (
                "clf",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=3000,
                    solver="liblinear",
                    penalty="l2",
                    C=1.0,
                ),
            ),
        ]
    )
    baseline_lr.fit(x_train_f, y_train)
    baseline_probs = baseline_lr.predict_proba(x_test_f)[:, 1]
    baseline_metrics = evaluate_predictions("Baseline LR", y_test, baseline_probs, 0.5)

    # 2) Tuned Logistic Regression
    tuned_pipe = Pipeline(
        steps=[("preprocessor", pre_lr), ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]
    )
    grid_params = [
        {
            "clf__solver": ["liblinear"],
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
        },
        {
            "clf__solver": ["lbfgs"],
            "clf__penalty": ["l2"],
            "clf__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
        },
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
        },
    ]

    tuned_search = GridSearchCV(
        tuned_pipe,
        param_grid=grid_params,
        scoring="roc_auc",
        cv=5,
        n_jobs=1,
        verbose=0,
    )
    tuned_search.fit(x_train_f, y_train)
    tuned_lr = tuned_search.best_estimator_

    tuned_val_probs = tuned_lr.predict_proba(x_val_f)[:, 1]
    threshold_table, best_f1_threshold, best_youden_threshold = optimize_thresholds(y_val, tuned_val_probs)

    tuned_test_probs = tuned_lr.predict_proba(x_test_f)[:, 1]
    tuned_default_metrics = evaluate_predictions("Tuned LR (0.50)", y_test, tuned_test_probs, 0.5)
    tuned_best_metrics = evaluate_predictions(
        "Tuned LR (Best F1 Thr)", y_test, tuned_test_probs, best_f1_threshold
    )

    # 3) class_weight='balanced'
    best_params = tuned_search.best_params_.copy()
    best_params["clf__class_weight"] = "balanced"
    balanced_lr = Pipeline(
        steps=[
            ("preprocessor", pre_lr),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    solver=best_params["clf__solver"],
                    penalty=best_params["clf__penalty"],
                    C=best_params["clf__C"],
                    class_weight="balanced",
                ),
            ),
        ]
    )
    balanced_lr.fit(x_train_f, y_train)
    balanced_probs = balanced_lr.predict_proba(x_test_f)[:, 1]
    balanced_metrics = evaluate_predictions("Balanced LR", y_test, balanced_probs, 0.5)

    # 4) SMOTE Logistic Regression (optional)
    smote_metrics = None
    smote_probs = None
    smote_note = "SMOTE skipped: imbalanced-learn not installed."
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        smote_lr = ImbPipeline(
            steps=[
                ("preprocessor", pre_lr),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        solver=best_params["clf__solver"],
                        penalty=best_params["clf__penalty"],
                        C=best_params["clf__C"],
                    ),
                ),
            ]
        )
        smote_lr.fit(x_train_f, y_train)
        smote_probs = smote_lr.predict_proba(x_test_f)[:, 1]
        smote_metrics = evaluate_predictions("SMOTE LR", y_test, smote_probs, 0.5)
        smote_note = "SMOTE executed successfully."
    except Exception:
        pass

    # 5) Calibrated model (Platt vs Isotonic; select by validation Brier score)
    sigmoid_cal = CalibratedClassifierCV(estimator=tuned_lr, method="sigmoid", cv=5)
    isotonic_cal = CalibratedClassifierCV(estimator=tuned_lr, method="isotonic", cv=5)
    sigmoid_cal.fit(x_train_f, y_train)
    isotonic_cal.fit(x_train_f, y_train)

    sig_val = sigmoid_cal.predict_proba(x_val_f)[:, 1]
    iso_val = isotonic_cal.predict_proba(x_val_f)[:, 1]
    brier_sig = float(np.mean((sig_val - y_val.values) ** 2))
    brier_iso = float(np.mean((iso_val - y_val.values) ** 2))

    calibrated_model = sigmoid_cal if brier_sig <= brier_iso else isotonic_cal
    calibrated_name = "Calibrated LR (Platt)" if brier_sig <= brier_iso else "Calibrated LR (Isotonic)"
    calibrated_probs = calibrated_model.predict_proba(x_test_f)[:, 1]
    calibrated_metrics = evaluate_predictions(calibrated_name, y_test, calibrated_probs, 0.5)

    # Assemble LR-only metrics table
    metrics_all = [
        baseline_metrics,
        tuned_default_metrics,
        tuned_best_metrics,
        balanced_metrics,
        calibrated_metrics,
    ]
    if smote_metrics is not None:
        metrics_all.append(smote_metrics)

    metrics_df = pd.DataFrame([to_dict(m) for m in metrics_all]).sort_values(
        by=["f1_score", "roc_auc", "recall"], ascending=False
    )
    best_model_name = str(metrics_df.iloc[0]["model"])

    # Curves and plots (LR variants only)
    roc_inputs = {
        baseline_metrics.model_name: roc_curve(y_test, baseline_probs),
        tuned_best_metrics.model_name: roc_curve(y_test, tuned_test_probs),
        balanced_metrics.model_name: roc_curve(y_test, balanced_probs),
        calibrated_metrics.model_name: roc_curve(y_test, calibrated_probs),
    }
    if smote_probs is not None:
        roc_inputs["SMOTE LR"] = roc_curve(y_test, smote_probs)

    roc_curves = {
        name: (fpr, tpr, roc_auc_score(y_test, probs))
        for name, (fpr, tpr, thr) in roc_inputs.items()
        for probs in [
            baseline_probs
            if name == baseline_metrics.model_name
            else tuned_test_probs
            if name == tuned_best_metrics.model_name
            else balanced_probs
            if name == balanced_metrics.model_name
            else calibrated_probs
            if name == calibrated_metrics.model_name
            else smote_probs
        ]
    }

    pr_curves = {}
    prob_map_for_pr = {
        baseline_metrics.model_name: baseline_probs,
        tuned_best_metrics.model_name: tuned_test_probs,
        balanced_metrics.model_name: balanced_probs,
        calibrated_metrics.model_name: calibrated_probs,
    }
    if smote_probs is not None:
        prob_map_for_pr["SMOTE LR"] = smote_probs

    for name, probs in prob_map_for_pr.items():
        p, r, _ = precision_recall_curve(y_test, probs)
        pr_curves[name] = (r, p, auc(r, p))

    cm_path = plot_confusion_matrices(metrics_all)
    roc_path = plot_roc_curves(roc_curves)
    pr_path = plot_pr_curves(pr_curves)
    th_path = plot_threshold_curve(threshold_table)
    cal_path = plot_calibration_curves(
        y_test,
        {
            tuned_default_metrics.model_name: tuned_test_probs,
            calibrated_metrics.model_name: calibrated_probs,
            baseline_metrics.model_name: baseline_probs,
        },
    )
    fi_path = plot_lr_coefficients(tuned_lr)

    # Save artifacts
    metrics_df.to_csv(OUTPUT_DIR / "all_model_metrics.csv", index=False)
    threshold_table.sort_values("threshold").to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)
    final_vif.to_csv(OUTPUT_DIR / "final_vif_table.csv", index=False)

    summary = {
        "best_model": best_model_name,
        "best_threshold_f1": best_f1_threshold,
        "best_threshold_youden_j": best_youden_threshold,
        "best_tuned_params": tuned_search.best_params_,
        "manual_multicollinearity_handling": manual_info,
        "vif_dropped_features": vif_dropped,
        "smote_note": smote_note,
        "calibration_choice": calibrated_name,
        "brier_sigmoid": brier_sig,
        "brier_isotonic": brier_iso,
        "saved_figures": [
            cm_path.as_posix(),
            roc_path.as_posix(),
            pr_path.as_posix(),
            th_path.as_posix(),
            cal_path.as_posix(),
            fi_path.as_posix(),
        ],
    }
    with open(OUTPUT_DIR / "advanced_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("=== Advanced Logistic Regression Pipeline ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Train/Val/Test sizes: {len(x_train_f)}/{len(x_val_f)}/{len(x_test_f)}")
    print("\nBest tuned LR params:")
    print(tuned_search.best_params_)
    print(f"Best threshold (F1): {best_f1_threshold:.2f}")
    print(f"Best threshold (Youden J): {best_youden_threshold:.2f}")
    print(f"Calibration selected: {calibrated_name}")
    print(f"SMOTE status: {smote_note}")

    print("\nTop model ranking (by F1, ROC-AUC, Recall):")
    print(metrics_df[["model", "threshold", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]].to_string(index=False))

    print("\nSaved output files:")
    print("- outputs/all_model_metrics.csv")
    print("- outputs/threshold_analysis.csv")
    print("- outputs/final_vif_table.csv")
    print("- outputs/advanced_run_summary.json")
    for p in summary["saved_figures"]:
        print(f"- {p}")


if __name__ == "__main__":
    main()
