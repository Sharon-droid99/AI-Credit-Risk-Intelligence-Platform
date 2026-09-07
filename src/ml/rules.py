import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("models/application_features.csv")
OUTPUT_DIR = Path("documents/rules")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 80)
    print("BUSINESS RISK RULE DERIVATION")
    print("=" * 80)

    print("\nLoading applicant-level dataset...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Applicants loaded: {len(df):,}"
    )

    return df


# ============================================================
# CREATE BUSINESS BANDS
# ============================================================

def create_bands(df):

    df = df.copy()

    # --------------------------------------------------------
    # External credit scores
    # --------------------------------------------------------

    for column in [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]:

        if column in df.columns:

            df[f"{column}_BAND"] = pd.qcut(
                df[column],
                q=5,
                labels=[
                    "Very Low",
                    "Low",
                    "Medium",
                    "High",
                    "Very High"
                ],
                duplicates="drop"
            )

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    df["AGE_BAND"] = pd.cut(
        df["AGE_YEARS"],
        bins=[
            0,
            25,
            30,
            35,
            40,
            50,
            60,
            np.inf
        ],
        labels=[
            "<=25",
            "26-30",
            "31-35",
            "36-40",
            "41-50",
            "51-60",
            "60+"
        ]
    )

    # --------------------------------------------------------
    # Employment duration
    # --------------------------------------------------------

    df["EMPLOYMENT_BAND"] = pd.cut(
        df["EMPLOYMENT_YEARS"],
        bins=[
            -np.inf,
            1,
            3,
            5,
            10,
            20,
            np.inf
        ],
        labels=[
            "<=1 year",
            "1-3 years",
            "3-5 years",
            "5-10 years",
            "10-20 years",
            "20+ years"
        ]
    )

    # --------------------------------------------------------
    # Credit-to-income ratio
    # --------------------------------------------------------

    if "CREDIT_INCOME_RATIO" in df.columns:

        df["CREDIT_INCOME_BAND"] = pd.qcut(
            df["CREDIT_INCOME_RATIO"],
            q=5,
            labels=[
                "Very Low",
                "Low",
                "Medium",
                "High",
                "Very High"
            ],
            duplicates="drop"
        )

    # --------------------------------------------------------
    # Late payment rate
    # --------------------------------------------------------

    if "LATE_PAYMENT_RATE" in df.columns:

        df["LATE_PAYMENT_BAND"] = pd.cut(
            df["LATE_PAYMENT_RATE"],
            bins=[
                -np.inf,
                0,
                0.10,
                0.25,
                0.50,
                1.0,
                np.inf
            ],
            labels=[
                "No late payments",
                "Low",
                "Moderate",
                "High",
                "Very High",
                "Extreme"
            ],
            include_lowest=True
        )

    return df


# ============================================================
# DEFAULT RATE ANALYSIS
# ============================================================

def analyze_feature(df, feature, output_name):

    if feature not in df.columns:
        print(
            f"\nSkipping {feature}: column not found."
        )
        return None

    analysis = (
        df.groupby(
            feature,
            observed=False
        )
        .agg(
            applicants=("TARGET", "size"),
            defaults=("TARGET", "sum"),
            default_rate=("TARGET", "mean")
        )
        .reset_index()
    )

    analysis["default_rate"] = (
        analysis["default_rate"] * 100
    ).round(2)

    analysis = analysis[
        analysis["applicants"] >= 100
    ]

    print("\n" + "=" * 80)
    print(f"ANALYSIS: {feature}")
    print("=" * 80)

    print(
        analysis.to_string(index=False)
    )

    output_path = (
        OUTPUT_DIR /
        f"{output_name}.csv"
    )

    analysis.to_csv(
        output_path,
        index=False
    )

    return analysis


# ============================================================
# GENERATE BUSINESS RULES
# ============================================================

def generate_rules(results):

    rules = []

    # --------------------------------------------------------
    # External score rules
    # --------------------------------------------------------

    for feature in [
        "EXT_SOURCE_1_BAND",
        "EXT_SOURCE_2_BAND",
        "EXT_SOURCE_3_BAND"
    ]:

        if feature not in results:
            continue

        analysis = results[feature]

        if len(analysis) >= 2:

            lowest = analysis.iloc[0]
            highest = analysis.iloc[-1]

            rules.append({
                "rule_category": "External Credit Score",
                "rule": (
                    f"Applicants in the lowest observed "
                    f"{feature.replace('_BAND', '')} score band "
                    f"have an observed default rate of "
                    f"{lowest['default_rate']:.2f}%, compared with "
                    f"{highest['default_rate']:.2f}% in the highest band."
                ),
                "business_action": (
                    "Applicants with weaker external credit scores "
                    "should receive additional risk review."
                )
            })

    # --------------------------------------------------------
    # Employment rule
    # --------------------------------------------------------

    if "EMPLOYMENT_BAND" in results:

        analysis = results["EMPLOYMENT_BAND"]

        if len(analysis) >= 2:

            shortest = analysis.iloc[0]
            longest = analysis.iloc[-1]

            rules.append({
                "rule_category": "Employment History",
                "rule": (
                    f"Applicants with {shortest['AGE_BAND'] if 'AGE_BAND' in shortest else 'shorter'} "
                    f"employment history show an observed default rate of "
                    f"{shortest['default_rate']:.2f}%, compared with "
                    f"{longest['default_rate']:.2f}% for the longest employment band."
                ),
                "business_action": (
                    "Limited employment history can be considered "
                    "as an additional risk-review signal."
                )
            })

    # --------------------------------------------------------
    # Age rule
    # --------------------------------------------------------

    if "AGE_BAND" in results:

        analysis = results["AGE_BAND"]

        if len(analysis) >= 2:

            youngest = analysis.iloc[0]
            oldest = analysis.iloc[-1]

            rules.append({
                "rule_category": "Age",
                "rule": (
                    f"The youngest observed age band has a default "
                    f"rate of {youngest['default_rate']:.2f}%, while "
                    f"the oldest observed band has a default rate of "
                    f"{oldest['default_rate']:.2f}%."
                ),
                "business_action": (
                    "Age-related patterns should be treated as "
                    "descriptive portfolio insights rather than "
                    "automatic lending decisions."
                )
            })

    # --------------------------------------------------------
    # Late payment rule
    # --------------------------------------------------------

    if "LATE_PAYMENT_BAND" in results:

        analysis = results["LATE_PAYMENT_BAND"]

        if len(analysis) >= 2:

            highest = analysis.loc[
                analysis["default_rate"].idxmax()
            ]

            lowest = analysis.loc[
                analysis["default_rate"].idxmin()
            ]

            rules.append({
                "rule_category": "Repayment Behaviour",
                "rule": (
                    f"The observed default rate varies from "
                    f"{lowest['default_rate']:.2f}% to "
                    f"{highest['default_rate']:.2f}% across "
                    f"late-payment-rate bands."
                ),
                "business_action": (
                    "Higher observed late-payment behaviour "
                    "should trigger additional repayment-history review."
                )
            })

    return pd.DataFrame(rules)


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    print("\nCreating business feature bands...")

    df = create_bands(df)

    results = {}

    features_to_analyze = [
        ("EXT_SOURCE_1_BAND", "external_score_1"),
        ("EXT_SOURCE_2_BAND", "external_score_2"),
        ("EXT_SOURCE_3_BAND", "external_score_3"),
        ("AGE_BAND", "age"),
        ("EMPLOYMENT_BAND", "employment"),
        ("CREDIT_INCOME_BAND", "credit_income"),
        ("LATE_PAYMENT_BAND", "late_payment")
    ]

    for feature, output_name in features_to_analyze:

        analysis = analyze_feature(
            df,
            feature,
            output_name
        )

        if analysis is not None:
            results[feature] = analysis

    # --------------------------------------------------------
    # Generate rule table
    # --------------------------------------------------------

    rules = generate_rules(results)

    rules_path = (
        OUTPUT_DIR /
        "business_risk_rules.csv"
    )

    rules.to_csv(
        rules_path,
        index=False
    )

    # --------------------------------------------------------
    # Display rules
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DERIVED BUSINESS RISK RULES")
    print("=" * 80)

    if rules.empty:

        print(
            "No rules could be generated."
        )

    else:

        for i, row in rules.iterrows():

            print(
                f"\nRULE {i + 1}: "
                f"{row['rule_category']}"
            )

            print(
                f"Observation:\n"
                f"  {row['rule']}"
            )

            print(
                f"Business interpretation:\n"
                f"  {row['business_action']}"
            )

    print("\n" + "=" * 80)
    print("RULE DERIVATION COMPLETE")
    print("=" * 80)

    print(
        f"\nRule file saved to:\n"
        f"{rules_path.resolve()}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()