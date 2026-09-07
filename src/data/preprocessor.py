import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# APPLICATION FEATURES
# ============================================================

def build_application_features(df):

    df = df.copy()

    # Age
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    # Employment
    df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(
        365243,
        np.nan
    )

    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED_CLEAN"] / 365.25
    )

    df.loc[
        df["EMPLOYMENT_YEARS"] < 0,
        "EMPLOYMENT_YEARS"
    ] = np.nan

    # Financial ratios
    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    df["CREDIT_GOODS_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_GOODS_PRICE"].replace(0, np.nan)
    )

    # Family
    df["CHILDREN_PER_FAMILY"] = (
        df["CNT_CHILDREN"] /
        df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    )

    # Remove raw temporal variables
    df = df.drop(
        columns=[
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_EMPLOYED_CLEAN"
        ],
        errors="ignore"
    )

    return df


# ============================================================
# BUREAU FEATURES
# ============================================================

def build_bureau_features():

    print("\nLoading bureau.csv...")

    bureau = pd.read_csv(
        DATA_DIR / "bureau.csv"
    )

    print(f"Bureau rows: {len(bureau):,}")

    bureau_features = (
        bureau
        .groupby("SK_ID_CURR")
        .agg(
            BUREAU_ACCOUNT_COUNT=(
                "SK_ID_BUREAU",
                "count"
            ),

            BUREAU_ACTIVE_ACCOUNT_COUNT=(
                "CREDIT_ACTIVE",
                lambda x: (x == "Active").sum()
            ),

            BUREAU_TOTAL_CREDIT=(
                "AMT_CREDIT_SUM",
                "sum"
            ),

            BUREAU_TOTAL_DEBT=(
                "AMT_CREDIT_SUM_DEBT",
                "sum"
            ),

            BUREAU_TOTAL_OVERDUE=(
                "AMT_CREDIT_SUM_OVERDUE",
                "sum"
            ),

            BUREAU_AVG_CREDIT_DAYS=(
                "DAYS_CREDIT",
                "mean"
            )
        )
        .reset_index()
    )

    return bureau_features


# ============================================================
# PREVIOUS APPLICATION FEATURES
# ============================================================

def build_previous_application_features():

    print("\nLoading previous_application.csv...")

    previous = pd.read_csv(
        DATA_DIR / "previous_application.csv"
    )

    print(
        f"Previous application rows: "
        f"{len(previous):,}"
    )

    previous_features = (
        previous
        .groupby("SK_ID_CURR")
        .agg(
            PREV_APPLICATION_COUNT=(
                "SK_ID_PREV",
                "count"
            ),

            PREV_APPROVED_COUNT=(
                "NAME_CONTRACT_STATUS",
                lambda x: (x == "Approved").sum()
            ),

            PREV_REFUSED_COUNT=(
                "NAME_CONTRACT_STATUS",
                lambda x: (x == "Refused").sum()
            ),

            PREV_TOTAL_CREDIT=(
                "AMT_CREDIT",
                "sum"
            ),

            PREV_AVG_CREDIT=(
                "AMT_CREDIT",
                "mean"
            )
        )
        .reset_index()
    )

    previous_features["PREV_APPROVAL_RATE"] = (
        previous_features["PREV_APPROVED_COUNT"] /
        previous_features["PREV_APPLICATION_COUNT"]
    )

    return previous_features


# ============================================================
# INSTALLMENT PAYMENT FEATURES
# ============================================================

def build_installment_features():

    print("\nLoading installments_payments.csv...")

    installments = pd.read_csv(
        DATA_DIR / "installments_payments.csv",
        usecols=[
            "SK_ID_PREV",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT"
        ]
    )

    print(
        f"Installment rows: "
        f"{len(installments):,}"
    )

    # --------------------------------------------------------
    # Payment delay
    #
    # Positive value = payment was made late
    # Zero/negative = on time or early
    # --------------------------------------------------------

    installments["PAYMENT_DELAY"] = (
        installments["DAYS_ENTRY_PAYMENT"]
        - installments["DAYS_INSTALMENT"]
    )

    installments["IS_LATE"] = (
        installments["PAYMENT_DELAY"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Payment completion ratio
    # --------------------------------------------------------

    installments["PAYMENT_RATIO"] = (
        installments["AMT_PAYMENT"] /
        installments["AMT_INSTALMENT"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Aggregate by previous application first
    # --------------------------------------------------------

    prev_installments = (
        installments
        .groupby("SK_ID_PREV")
        .agg(
            INSTALLMENT_COUNT=(
                "NUM_INSTALMENT_NUMBER",
                "count"
            ),

            LATE_PAYMENT_COUNT=(
                "IS_LATE",
                "sum"
            ),

            AVG_PAYMENT_DELAY=(
                "PAYMENT_DELAY",
                "mean"
            ),

            AVG_PAYMENT_RATIO=(
                "PAYMENT_RATIO",
                "mean"
            )
        )
        .reset_index()
    )

    prev_installments["LATE_PAYMENT_RATE"] = (
        prev_installments["LATE_PAYMENT_COUNT"] /
        prev_installments["INSTALLMENT_COUNT"]
    )

    # --------------------------------------------------------
    # Map previous applications to applicants
    # --------------------------------------------------------

    previous = pd.read_csv(
        DATA_DIR / "previous_application.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV"
        ]
    )

    prev_installments = prev_installments.merge(
        previous,
        on="SK_ID_PREV",
        how="inner"
    )

    # --------------------------------------------------------
    # Aggregate at applicant level
    # --------------------------------------------------------

    applicant_features = (
        prev_installments
        .groupby("SK_ID_CURR")
        .agg(
            INSTALLMENT_COUNT=(
                "INSTALLMENT_COUNT",
                "sum"
            ),

            LATE_PAYMENT_COUNT=(
                "LATE_PAYMENT_COUNT",
                "sum"
            ),

            AVG_PAYMENT_DELAY=(
                "AVG_PAYMENT_DELAY",
                "mean"
            ),

            AVG_PAYMENT_RATIO=(
                "AVG_PAYMENT_RATIO",
                "mean"
            ),

            LATE_PAYMENT_RATE=(
                "LATE_PAYMENT_RATE",
                "mean"
            )
        )
        .reset_index()
    )

    return applicant_features


# ============================================================
# POS CASH FEATURES
# ============================================================

def build_pos_features():

    print("\nLoading POS_CASH_balance.csv...")

    pos = pd.read_csv(
        DATA_DIR / "POS_CASH_balance.csv",
        usecols=[
            "SK_ID_PREV",
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "SK_DPD",
            "SK_DPD_DEF"
        ]
    )

    print(
        f"POS cash rows: "
        f"{len(pos):,}"
    )

    # DPD = days past due
    pos["IS_LATE"] = (
        pos["SK_DPD"] > 0
    ).astype(int)

    pos["IS_DEFAULT_DPD"] = (
        pos["SK_DPD_DEF"] > 0
    ).astype(int)

    pos_features = (
        pos
        .groupby("SK_ID_CURR")
        .agg(
            POS_RECORD_COUNT=(
                "SK_ID_PREV",
                "count"
            ),

            POS_LATE_RECORD_COUNT=(
                "IS_LATE",
                "sum"
            ),

            POS_DEFAULT_DPD_COUNT=(
                "IS_DEFAULT_DPD",
                "sum"
            ),

            POS_AVG_DPD=(
                "SK_DPD",
                "mean"
            ),

            POS_MAX_DPD=(
                "SK_DPD",
                "max"
            )
        )
        .reset_index()
    )

    pos_features["POS_LATE_RATE"] = (
        pos_features["POS_LATE_RECORD_COUNT"] /
        pos_features["POS_RECORD_COUNT"]
    )

    return pos_features


# ============================================================
# CREDIT CARD FEATURES
# ============================================================

def build_credit_card_features():

    print("\nLoading credit_card_balance.csv...")

    cc = pd.read_csv(
        DATA_DIR / "credit_card_balance.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_TOTAL_RECEIVABLE",
            "SK_DPD",
            "SK_DPD_DEF"
        ]
    )

    print(
        f"Credit card rows: "
        f"{len(cc):,}"
    )

    # Credit utilisation
    cc["CREDIT_UTILIZATION"] = (
        cc["AMT_BALANCE"] /
        cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(
            0,
            np.nan
        )
    )

    cc["IS_LATE"] = (
        cc["SK_DPD"] > 0
    ).astype(int)

    cc_features = (
        cc
        .groupby("SK_ID_CURR")
        .agg(
            CC_RECORD_COUNT=(
                "SK_ID_PREV",
                "count"
            ),

            CC_AVG_BALANCE=(
                "AMT_BALANCE",
                "mean"
            ),

            CC_MAX_BALANCE=(
                "AMT_BALANCE",
                "max"
            ),

            CC_AVG_CREDIT_LIMIT=(
                "AMT_CREDIT_LIMIT_ACTUAL",
                "mean"
            ),

            CC_AVG_UTILIZATION=(
                "CREDIT_UTILIZATION",
                "mean"
            ),

            CC_AVG_RECEIVABLE=(
                "AMT_TOTAL_RECEIVABLE",
                "mean"
            ),

            CC_LATE_RECORD_COUNT=(
                "IS_LATE",
                "sum"
            ),

            CC_AVG_DPD=(
                "SK_DPD",
                "mean"
            ),

            CC_MAX_DPD=(
                "SK_DPD",
                "max"
            )
        )
        .reset_index()
    )

    cc_features["CC_LATE_RATE"] = (
        cc_features["CC_LATE_RECORD_COUNT"] /
        cc_features["CC_RECORD_COUNT"]
    )

    return cc_features


# ============================================================
# MAIN DATASET BUILD
# ============================================================

def build_ml_dataset():

    print("=" * 80)
    print("BUILDING FINAL APPLICANT-LEVEL ML DATASET")
    print("=" * 80)

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    print("\nLoading application_train.csv...")

    application = pd.read_csv(
        DATA_DIR / "application_train.csv"
    )

    print(
        f"Application rows: "
        f"{len(application):,}"
    )

    application = build_application_features(
        application
    )

    # --------------------------------------------------------
    # Historical features
    # --------------------------------------------------------

    bureau_features = build_bureau_features()

    previous_features = (
        build_previous_application_features()
    )

    installment_features = (
        build_installment_features()
    )

    pos_features = build_pos_features()

    cc_features = build_credit_card_features()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\nMerging all applicant-level features...")

    df = application.merge(
        bureau_features,
        on="SK_ID_CURR",
        how="left"
    )

    df = df.merge(
        previous_features,
        on="SK_ID_CURR",
        how="left"
    )

    df = df.merge(
        installment_features,
        on="SK_ID_CURR",
        how="left"
    )

    df = df.merge(
        pos_features,
        on="SK_ID_CURR",
        how="left"
    )

    df = df.merge(
        cc_features,
        on="SK_ID_CURR",
        how="left"
    )

    print(
        f"\nFinal dataset shape: "
        f"{df.shape}"
    )

        # --------------------------------------------------------
    # Historical data availability indicators
    # --------------------------------------------------------
    # These indicators distinguish:
    #   0 = no historical record exists
    #   1 = historical record exists
    #
    # This is important because a value of 0 can otherwise mean
    # either "no history" or a genuine zero-valued measurement.
    # --------------------------------------------------------

    df["HAS_BUREAU_HISTORY"] = (
        df["BUREAU_ACCOUNT_COUNT"].notna()
    ).astype(int)

    df["HAS_PREVIOUS_APPLICATION"] = (
        df["PREV_APPLICATION_COUNT"].notna()
    ).astype(int)

    df["HAS_INSTALLMENT_HISTORY"] = (
        df["INSTALLMENT_COUNT"].notna()
    ).astype(int)

    df["HAS_POS_HISTORY"] = (
        df["POS_RECORD_COUNT"].notna()
    ).astype(int)

    df["HAS_CREDIT_CARD_HISTORY"] = (
        df["CC_RECORD_COUNT"].notna()
    ).astype(int)

    # --------------------------------------------------------
    # Fill historical aggregate features
    # --------------------------------------------------------
    #
    # Count-based features:
    # No history naturally means zero records.
    #
    # Rate/average features:
    # Keep missing values as NaN so LightGBM can distinguish
    # "no historical information" from an actual zero.
    # --------------------------------------------------------

    count_columns = [
        col
        for col in df.columns
        if (
            col.endswith("_COUNT")
            or col.endswith("_RECORD_COUNT")
        )
    ]

    # Counts can safely be zero when no history exists.
    df[count_columns] = df[count_columns].fillna(0)

    print("\nHistorical data availability:")
    print(
        f"  Applicants with bureau history: "
        f"{df['HAS_BUREAU_HISTORY'].sum():,}"
    )
    print(
        f"  Applicants with previous applications: "
        f"{df['HAS_PREVIOUS_APPLICATION'].sum():,}"
    )
    print(
        f"  Applicants with installment history: "
        f"{df['HAS_INSTALLMENT_HISTORY'].sum():,}"
    )
    print(
        f"  Applicants with POS history: "
        f"{df['HAS_POS_HISTORY'].sum():,}"
    )
    print(
        f"  Applicants with credit-card history: "
        f"{df['HAS_CREDIT_CARD_HISTORY'].sum():,}"
    )
    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        "application_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved final dataset to:\n"
        f"{output_path.resolve()}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL DATASET VALIDATION")
    print("=" * 80)

    print(
        f"\nRows    : {len(df):,}"
        f"\nColumns : {df.shape[1]:,}"
    )

    print(
        f"\nUnique applicants: "
        f"{df['SK_ID_CURR'].nunique():,}"
    )

    print("\nTarget distribution:")
    print(
        df["TARGET"]
        .value_counts(normalize=True)
        .round(4)
    )

    print("\nNew repayment features:")

    repayment_columns = [
        col for col in df.columns
        if (
            col.startswith("INSTALLMENT_")
            or col.startswith("LATE_")
            or col.startswith("AVG_PAYMENT_")
            or col.startswith("POS_")
            or col.startswith("CC_")
        )
    ]

    for col in repayment_columns:
        print(f"  ✓ {col}")

    return df


if __name__ == "__main__":
    build_ml_dataset()