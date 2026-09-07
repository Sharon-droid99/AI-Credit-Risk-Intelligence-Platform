import json
import joblib
import pandas as pd

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/credit_risk_lightgbm.pkl")
FEATURE_PATH = Path("models/model_features.json")


# Initial business risk thresholds
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = joblib.load(MODEL_PATH)

    with open(
        FEATURE_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    return model, metadata


# ============================================================
# RISK BAND
# ============================================================

def assign_risk_band(probability):

    if probability < LOW_RISK_THRESHOLD:
        return "Low"

    elif probability < HIGH_RISK_THRESHOLD:
        return "Medium"

    else:
        return "High"


# ============================================================
# PREDICT
# ============================================================

def predict_risk(applicant):

    """
    Predict default probability and risk band
    for a single applicant.

    applicant:
        Dictionary or pandas Series containing
        the same features used during training.
    """

    model, metadata = load_model()

    # Convert input into DataFrame
    if isinstance(applicant, pd.Series):
        applicant = applicant.to_dict()

    X = pd.DataFrame([applicant])

    # --------------------------------------------------------
    # Remove columns that were not used by the model
    # --------------------------------------------------------

    X = X.drop(
        columns=["TARGET", "SK_ID_CURR", "CODE_GENDER"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Ensure all expected features exist
    # --------------------------------------------------------

    expected_features = metadata["features"]

    missing_features = [
        col
        for col in expected_features
        if col not in X.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features[:10])
        )

    # Keep exact training feature order
    X = X[expected_features]

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    categorical_features = (
        metadata["categorical_features"]
    )

    for col in categorical_features:

        X[col] = X[col].fillna("Missing")

        X[col] = X[col].astype("category")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = model.predict_proba(X)[0, 1]

    risk_score = probability * 100

    risk_band = assign_risk_band(
        probability
    )

    return {
        "default_probability": round(
            float(probability),
            4
        ),

        "risk_score": round(
            float(risk_score),
            2
        ),

        "risk_band": risk_band
    }


# ============================================================
# TEST WITH AN ACTUAL APPLICANT
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("CREDIT RISK PREDICTION TEST")
    print("=" * 80)

    # Load feature dataset
    df = pd.read_csv(
        "models/application_features.csv"
    )

    # Take the first applicant as a test case
    applicant = df.iloc[0]

    print("\nApplicant ID:")
    print(applicant["SK_ID_CURR"])

    print("\nActual TARGET:")
    print(applicant["TARGET"])

    result = predict_risk(
        applicant
    )

    print("\nPrediction:")
    print(
        f"Default probability: "
        f"{result['default_probability'] * 100:.2f}%"
    )

    print(
        f"Risk score: "
        f"{result['risk_score']:.2f}/100"
    )

    print(
        f"Risk band: "
        f"{result['risk_band']}"
    )