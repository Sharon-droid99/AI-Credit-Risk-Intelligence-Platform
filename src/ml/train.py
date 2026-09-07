import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("models/application_features.csv")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "credit_risk_lightgbm.pkl"
FEATURE_PATH = MODEL_DIR / "model_features.json"


RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("CREDIT RISK MODEL TRAINING")
print("=" * 80)

print("\nLoading feature dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

TARGET = "TARGET"

y = df[TARGET]

X = df.drop(columns=[TARGET])


# ============================================================
# REMOVE IDENTIFIER
# ============================================================

if "SK_ID_CURR" in X.columns:
    X = X.drop(columns=["SK_ID_CURR"])


# ============================================================
# REMOVE SENSITIVE FEATURE FROM FIRST MODEL
# ============================================================

if "CODE_GENDER" in X.columns:
    print("\nRemoving CODE_GENDER from model features")
    X = X.drop(columns=["CODE_GENDER"])


# ============================================================
# IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nFeature information:")
print(f"Total features      : {X.shape[1]}")
print(f"Numeric features    : {len(numeric_columns)}")
print(f"Categorical features: {len(categorical_columns)}")


# ============================================================
# CLEAN CATEGORICAL FEATURES
# ============================================================

for col in categorical_columns:

    # Convert missing values to explicit category
    X[col] = X[col].fillna("Missing")

    # LightGBM categorical features
    X[col] = X[col].astype("category")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating stratified train/validation split...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining rows  : {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")

print("\nTraining target distribution:")
print(
    y_train.value_counts(normalize=True)
    .round(4)
)

print("\nValidation target distribution:")
print(
    y_valid.value_counts(normalize=True)
    .round(4)
)


# ============================================================
# LIGHTGBM MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING LIGHTGBM")
print("=" * 80)

model = LGBMClassifier(
    objective="binary",

    n_estimators=500,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=0.1,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_columns
)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

y_probability = model.predict_proba(
    X_valid
)[:, 1]

# Default classification threshold
THRESHOLD = 0.50

y_prediction = (
    y_probability >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

roc_auc = roc_auc_score(
    y_valid,
    y_probability
)

accuracy = accuracy_score(
    y_valid,
    y_prediction
)

precision = precision_score(
    y_valid,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_valid,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_valid,
    y_prediction,
    zero_division=0
)

print(f"\nROC-AUC  : {roc_auc:.4f}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_valid,
    y_prediction
)

print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(
    classification_report(
        y_valid,
        y_prediction,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("TOP 20 FEATURES")
print("=" * 80)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"Model saved to:\n"
    f"{MODEL_PATH.resolve()}"
)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_metadata = {
    "features": X.columns.tolist(),
    "categorical_features": categorical_columns,
    "numeric_features": numeric_columns,
    "threshold": THRESHOLD,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE
}

with open(
    FEATURE_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_metadata,
        f,
        indent=2
    )

print(
    f"Feature metadata saved to:\n"
    f"{FEATURE_PATH.resolve()}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

print(f"""
Model              : LightGBM
Training rows      : {len(X_train):,}
Validation rows    : {len(X_valid):,}
Features           : {X.shape[1]}
Categorical        : {len(categorical_columns)}
Class weighting    : balanced

ROC-AUC            : {roc_auc:.4f}
Accuracy           : {accuracy:.4f}
Precision          : {precision:.4f}
Recall             : {recall:.4f}
F1 Score           : {f1:.4f}
""")

print("Next step: inspect evaluation results and feature importance.")