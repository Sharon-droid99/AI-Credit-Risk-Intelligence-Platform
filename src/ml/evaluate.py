import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("models/application_features.csv")
MODEL_PATH = Path("models/credit_risk_lightgbm.pkl")
FEATURE_PATH = Path("models/model_features.json")

OUTPUT_DIR = Path("documents/model_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

print("=" * 80)
print("CREDIT RISK MODEL EVALUATION")
print("=" * 80)

print("\nLoading data...")

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ============================================================
# PREPARE FEATURES
# ============================================================

y = df["TARGET"]

X = df.drop(columns=["TARGET"])

X = X.drop(
    columns=["SK_ID_CURR", "CODE_GENDER"],
    errors="ignore"
)

categorical_features = metadata["categorical_features"]

for col in categorical_features:

    if col in X.columns:

        X[col] = X[col].fillna("Missing")
        X[col] = X[col].astype("category")


# Keep exact training feature order
X = X[metadata["features"]]


# ============================================================
# SAME VALIDATION SPLIT
# ============================================================

print("\nRecreating validation split...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"Validation applicants: "
    f"{len(X_valid):,}"
)


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

print("\nGenerating probabilities...")

probabilities = model.predict_proba(
    X_valid
)[:, 1]


# ============================================================
# OVERALL METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_valid,
    probabilities
)

average_precision = average_precision_score(
    y_valid,
    probabilities
)

print("\n" + "=" * 80)
print("OVERALL MODEL PERFORMANCE")
print("=" * 80)

print(f"\nROC-AUC             : {roc_auc:.4f}")
print(f"Average Precision   : {average_precision:.4f}")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]

threshold_results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_valid,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    })


threshold_df = pd.DataFrame(
    threshold_results
)

print(
    "\n"
    + threshold_df.round(4).to_string(
        index=False
    )
)


threshold_df.to_csv(
    OUTPUT_DIR / "threshold_analysis.csv",
    index=False
)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_valid,
    probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"LightGBM (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Credit Default Model")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_valid,
        probabilities
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"AP = {average_precision:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(
    "Precision-Recall Curve — Credit Default Model"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# RISK BANDS
# ============================================================

print("\n" + "=" * 80)
print("RISK BAND ANALYSIS")
print("=" * 80)

risk_band = pd.cut(
    probabilities,
    bins=[
        -np.inf,
        0.30,
        0.60,
        np.inf
    ],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

risk_analysis = pd.DataFrame({
    "risk_band": risk_band,
    "actual_default": y_valid.values,
    "probability": probabilities
})


band_summary = (
    risk_analysis
    .groupby(
        "risk_band",
        observed=False
    )
    .agg(
        applicants=(
            "actual_default",
            "count"
        ),

        actual_defaults=(
            "actual_default",
            "sum"
        ),

        observed_default_rate=(
            "actual_default",
            "mean"
        ),

        average_predicted_probability=(
            "probability",
            "mean"
        )
    )
)

band_summary["observed_default_rate"] *= 100
band_summary["average_predicted_probability"] *= 100

print(
    "\n"
    + band_summary.round(2).to_string()
)


band_summary.to_csv(
    OUTPUT_DIR / "risk_band_analysis.csv"
)


# ============================================================
# RISK BAND CHART
# ============================================================

plt.figure(figsize=(8, 5))

band_summary[
    "observed_default_rate"
].plot(kind="bar")

plt.title(
    "Observed Default Rate by Risk Band"
)

plt.xlabel("Risk Band")
plt.ylabel("Observed Default Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "default_rate_by_risk_band.png",
    dpi=200
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print(
    f"""
ROC-AUC           : {roc_auc:.4f}
Average Precision : {average_precision:.4f}

Evaluation files saved to:
{OUTPUT_DIR.resolve()}
"""
)

print("Generated files:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(f"  ✓ {file.name}")