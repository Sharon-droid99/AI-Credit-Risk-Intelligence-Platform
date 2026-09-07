import json
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("models/credit_risk_lightgbm.pkl")
FEATURE_METADATA_PATH = Path("models/model_features.json")
DATA_PATH = Path("models/application_features.csv")

OUTPUT_DIR = Path("documents/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL AND FEATURE METADATA
# ============================================================

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)

    with open(FEATURE_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model, metadata


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, metadata):
    """
    Reproduce the feature preparation used during training.
    """

    df = df.copy()

    # Remove columns that were not used by the model
    df = df.drop(
        columns=[
            "TARGET",
            "SK_ID_CURR",
            "CODE_GENDER"
        ],
        errors="ignore"
    )

    expected_features = metadata["features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # Keep exactly the training feature order
    df = df[expected_features]

    # Convert categorical columns exactly as in training
    categorical_features = metadata["categorical_features"]

    for column in categorical_features:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("Missing")
                .astype("category")
            )

    return df


# ============================================================
# BUSINESS-FRIENDLY FEATURE NAMES
# ============================================================

FEATURE_LABELS = {
    "EXT_SOURCE_1": "External Credit Score 1",
    "EXT_SOURCE_2": "External Credit Score 2",
    "EXT_SOURCE_3": "External Credit Score 3",
    "AGE_YEARS": "Age",
    "EMPLOYMENT_YEARS": "Employment Duration",
    "CREDIT_INCOME_RATIO": "Credit-to-Income Ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-Income Ratio",
    "CREDIT_GOODS_RATIO": "Credit-to-Goods Ratio",
    "BUREAU_TOTAL_CREDIT": "Historical Credit Amount",
    "BUREAU_TOTAL_DEBT": "Historical Debt",
    "BUREAU_TOTAL_OVERDUE": "Historical Overdue Amount",
    "BUREAU_AVG_CREDIT_DAYS": "Average Credit History Duration",
    "PREV_APPLICATION_COUNT": "Previous Application Count",
    "PREV_APPROVED_COUNT": "Previous Approved Applications",
    "PREV_REFUSED_COUNT": "Previous Refused Applications",
    "PREV_TOTAL_CREDIT": "Previous Credit Amount",
    "PREV_AVG_CREDIT": "Average Previous Credit",
    "PREV_APPROVAL_RATE": "Previous Approval Rate",
    "LATE_PAYMENT_COUNT": "Late Payment Count",
    "AVG_PAYMENT_DELAY": "Average Payment Delay",
    "AVG_PAYMENT_RATIO": "Average Payment Ratio",
    "LATE_PAYMENT_RATE": "Late Payment Rate",
    "POS_LATE_RECORD_COUNT": "POS Late Payment Count",
    "POS_DEFAULT_DPD_COUNT": "POS Default-DPD Count",
    "POS_AVG_DPD": "Average POS Days Past Due",
    "POS_MAX_DPD": "Maximum POS Days Past Due",
    "POS_LATE_RATE": "POS Late Payment Rate",
    "CC_AVG_UTILIZATION": "Credit Card Utilization",
    "CC_LATE_RECORD_COUNT": "Credit Card Late Payment Count",
    "CC_AVG_DPD": "Average Credit Card Days Past Due",
    "CC_MAX_DPD": "Maximum Credit Card Days Past Due",
    "CC_LATE_RATE": "Credit Card Late Payment Rate",
    "HAS_BUREAU_HISTORY": "Has Bureau History",
    "HAS_PREVIOUS_APPLICATION": "Has Previous Application",
    "HAS_INSTALLMENT_HISTORY": "Has Installment History",
    "HAS_POS_HISTORY": "Has POS History",
    "HAS_CREDIT_CARD_HISTORY": "Has Credit Card History",
}


def readable_feature_name(feature):
    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title()
    )


# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

def explain_applicant(applicant_id=None):

    print("=" * 80)
    print("SHAP EXPLAINABLE AI")
    print("=" * 80)

    print("\nLoading model...")

    model, metadata = load_model()

    print("Loading feature dataset...")

    data = pd.read_csv(DATA_PATH)

    if applicant_id is None:
        applicant_id = int(data.iloc[0]["SK_ID_CURR"])

    applicant = data[
        data["SK_ID_CURR"] == applicant_id
    ].copy()

    if applicant.empty:
        raise ValueError(
            f"Applicant {applicant_id} not found."
        )

    print(f"\nExplaining applicant: {applicant_id}")

    X = prepare_features(
        applicant,
        metadata
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = model.predict_proba(X)[:, 1][0]

    print("\n" + "-" * 80)
    print("RISK PREDICTION")
    print("-" * 80)

    print(
        f"Default probability : "
        f"{probability:.2%}"
    )

    if probability < 0.30:
        risk_band = "Low"
    elif probability < 0.60:
        risk_band = "Medium"
    else:
        risk_band = "High"

    print(f"Risk band            : {risk_band}")

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Binary LightGBM models may return either:
    #   array
    # or
    #   list of arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values[0]

    feature_values = X.iloc[0]

    explanation = pd.DataFrame({
        "feature": X.columns,
        "feature_value": feature_values.values,
        "shap_value": shap_values
    })

    explanation["abs_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation["direction"] = explanation[
        "shap_value"
    ].apply(
        lambda x:
        "Increases risk"
        if x > 0
        else "Decreases risk"
    )

    explanation["business_feature"] = (
        explanation["feature"]
        .apply(readable_feature_name)
    )

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False
    )

    # ============================================================
    # BUSINESS-READABLE SHAP INTERPRETATION
    # ============================================================

    def create_business_explanation(explanation, probability, risk_band):
        """
        Convert technical SHAP contributions into a
        business-readable credit-risk explanation.
        """

        # Features with positive SHAP values increase
        # the model's predicted risk.
        risk_increasing = (
            explanation[explanation["shap_value"] > 0]
            .sort_values("abs_shap", ascending=False)
            .head(5)
        )

        # Features with negative SHAP values decrease
        # the model's predicted risk.
        risk_decreasing = (
            explanation[explanation["shap_value"] < 0]
            .sort_values("abs_shap", ascending=False)
            .head(5)
        )

        print("\n" + "=" * 80)
        print("BUSINESS-READABLE RISK EXPLANATION")
        print("=" * 80)

        print(
            f"\nThe applicant is classified as {risk_band} Risk "
            f"with an estimated default probability of "
            f"{probability:.2%}."
        )

        print("\nMAIN FACTORS INCREASING MODEL-PREDICTED RISK")
        print("-" * 80)

        if risk_increasing.empty:
            print("No major risk-increasing factors identified.")
        else:
            for i, (_, row) in enumerate(
                risk_increasing.iterrows(),
                start=1
            ):
                print(
                    f"{i}. {row['business_feature']}"
                )
                print(
                    "   This feature contributed to a "
                    "higher model-predicted risk."
                )

        print("\nMAIN FACTORS DECREASING MODEL-PREDICTED RISK")
        print("-" * 80)

        if risk_decreasing.empty:
            print("No major risk-decreasing factors identified.")
        else:
            for i, (_, row) in enumerate(
                risk_decreasing.iterrows(),
                start=1
            ):
                print(
                    f"{i}. {row['business_feature']}"
                )
                print(
                    "   This feature contributed to a "
                    "lower model-predicted risk."
                )

        print("\nBUSINESS SUMMARY")
        print("-" * 80)

        top_risk = risk_increasing["business_feature"].head(3).tolist()
        top_positive = risk_decreasing["business_feature"].head(2).tolist()

        if top_risk:
            risk_text = ", ".join(top_risk)
            summary = (
                f"The prediction is primarily influenced by "
                f"{risk_text}."
            )
        else:
            summary = (
                "No dominant risk-increasing factors were identified."
            )

        if top_positive:
            positive_text = ", ".join(top_positive)
            summary += (
                f" Factors contributing in the opposite direction "
                f"include {positive_text}."
            )

        print(summary)

        print(
            "\nNote: SHAP explanations describe how features "
            "influenced the model prediction. They do not establish "
            "causal relationships or guarantee that an applicant "
            "will or will not default."
        )

    create_business_explanation(
        explanation,
        probability,
        risk_band
    )

    # --------------------------------------------------------
    # Display top contributors
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP SHAP CONTRIBUTORS")
    print("=" * 80)

    top_features = explanation.head(15)

    display_columns = [
        "business_feature",
        "feature_value",
        "shap_value",
        "direction"
    ]

    print(
        top_features[display_columns].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save explanation
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR /
        f"applicant_{applicant_id}_shap.csv"
    )

    explanation.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # SHAP bar plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 8))

    shap.plots.bar(
        shap.Explanation(
            values=shap_values,
            base_values=(
                explainer.expected_value[1]
                if isinstance(
                    explainer.expected_value,
                    (list, tuple)
                )
                else explainer.expected_value
            ),
            data=X.iloc[0].values,
            feature_names=X.columns
        ),
        max_display=15,
        show=False
    )

    plt.title(
        f"Top SHAP Features — Applicant {applicant_id}"
    )

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR /
        f"applicant_{applicant_id}_shap_bar.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved explanation:")
    print(csv_path.resolve())

    print("\nSaved SHAP plot:")
    print(plot_path.resolve())

    print("\n" + "=" * 80)
    print("SHAP EXPLANATION COMPLETE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    explain_applicant()