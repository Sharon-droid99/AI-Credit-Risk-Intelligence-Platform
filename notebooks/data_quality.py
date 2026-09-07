import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data")


def inspect_application_data():
    print("=" * 80)
    print("APPLICATION TRAIN DATA QUALITY ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(DATA_DIR / "application_train.csv")

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")

    # ------------------------------------------------------------------
    # 1. DUPLICATES
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("1. DUPLICATE ANALYSIS")
    print("-" * 80)

    duplicate_rows = df.duplicated().sum()
    duplicate_ids = df["SK_ID_CURR"].duplicated().sum()

    print(f"Duplicate rows       : {duplicate_rows:,}")
    print(f"Duplicate SK_ID_CURR : {duplicate_ids:,}")

    # ------------------------------------------------------------------
    # 2. MISSING VALUES
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("2. MISSING VALUE ANALYSIS")
    print("-" * 80)

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_report = missing_report[
        missing_report["missing_count"] > 0
    ].sort_values("missing_pct", ascending=False)

    print("\nTop 20 columns by missingness:\n")
    print(missing_report.head(20).to_string())

    # ------------------------------------------------------------------
    # 3. DATA TYPES
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("3. DATA TYPE ANALYSIS")
    print("-" * 80)

    dtype_counts = df.dtypes.value_counts()

    print(dtype_counts)

    print("\nColumns by data type:\n")

    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        print(f"\n{dtype}: {len(cols)} columns")
        print(cols[:15])

    # ------------------------------------------------------------------
    # 4. TARGET ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("4. TARGET ANALYSIS")
    print("-" * 80)

    target_counts = df["TARGET"].value_counts()
    target_pct = df["TARGET"].value_counts(normalize=True) * 100

    print("\nTarget counts:")
    print(target_counts)

    print("\nTarget percentages:")
    print(target_pct.round(2))

    # ------------------------------------------------------------------
    # 5. NUMERIC RANGE CHECKS
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("5. NUMERIC RANGE / IMPOSSIBLE VALUE CHECK")
    print("-" * 80)

    numeric_cols = df.select_dtypes(include=np.number).columns

    range_checks = []

    for col in numeric_cols:
        series = df[col]

        range_checks.append({
            "column": col,
            "min": series.min(),
            "max": series.max(),
            "negative_count": (series < 0).sum(),
            "zero_count": (series == 0).sum()
        })

    range_report = pd.DataFrame(range_checks)

    print("\nColumns containing negative values:\n")
    print(
        range_report[
            range_report["negative_count"] > 0
        ]
        .sort_values("negative_count", ascending=False)
        .head(30)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 6. IMPORTANT BUSINESS VARIABLES
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("6. IMPORTANT BUSINESS VARIABLE CHECKS")
    print("-" * 80)

    important_columns = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]

    for col in important_columns:
        if col in df.columns:
            print(f"\n{col}")
            print(f"  Missing : {df[col].isna().sum():,}")
            print(f"  Min     : {df[col].min()}")
            print(f"  Max     : {df[col].max()}")
            print(f"  Median  : {df[col].median()}")

    # ------------------------------------------------------------------
    # 7. CATEGORICAL CARDINALITY
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("7. CATEGORICAL VARIABLE ANALYSIS")
    print("-" * 80)

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        print(
            f"{col:35s} "
            f"unique={df[col].nunique(dropna=True):4d} "
            f"missing={df[col].isna().sum():8,}"
        )

    # ------------------------------------------------------------------
    # 8. SPECIAL VALUE CHECK
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("8. SPECIAL / SUSPICIOUS VALUES")
    print("-" * 80)

    if "DAYS_EMPLOYED" in df.columns:
        suspicious_employment = (
            df["DAYS_EMPLOYED"] > 0
        ).sum()

        print(
            f"DAYS_EMPLOYED > 0: "
            f"{suspicious_employment:,}"
        )

    if "DAYS_BIRTH" in df.columns:
        positive_birth = (
            df["DAYS_BIRTH"] >= 0
        ).sum()

        print(
            f"DAYS_BIRTH >= 0: "
            f"{positive_birth:,}"
        )

    # ------------------------------------------------------------------
    # 9. FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATA QUALITY SUMMARY")
    print("=" * 80)

    print(f"""
Rows                       : {len(df):,}
Columns                    : {df.shape[1]:,}
Duplicate rows             : {duplicate_rows:,}
Duplicate applicant IDs    : {duplicate_ids:,}
Columns with missing data  : {len(missing_report):,}
Numeric columns            : {len(numeric_cols):,}
Categorical columns        : {len(categorical_cols):,}
Default rate (TARGET=1)    : {target_pct.get(1, 0):.2f}%
Non-default rate (TARGET=0): {target_pct.get(0, 0):.2f}%
""")

    return df


if __name__ == "__main__":
    inspect_application_data()