import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/application_train.csv")
OUTPUT_DIR = Path("documents/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("HOME CREDIT — EXPLORATORY DATA ANALYSIS")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Applicants: {len(df):,}")


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("1. TARGET DISTRIBUTION")
print("=" * 80)

target_counts = df["TARGET"].value_counts()
target_rates = df["TARGET"].value_counts(normalize=True) * 100

print("\nTarget counts:")
print(target_counts)

print("\nTarget percentages:")
print(target_rates.round(2))

plt.figure(figsize=(7, 5))

target_counts.plot(kind="bar")

plt.title("Loan Default Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Applicants")
plt.xticks(
    [0, 1],
    ["Non-default (0)", "Default (1)"],
    rotation=0
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "01_target_distribution.png",
    dpi=200
)
plt.close()


# ============================================================
# 2. AGE VS DEFAULT
# ============================================================

print("\n" + "=" * 80)
print("2. AGE VS DEFAULT")
print("=" * 80)

df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

age_summary = (
    df.groupby("TARGET")["AGE_YEARS"]
    .agg(["mean", "median", "min", "max"])
)

print("\nAge summary by target:")
print(age_summary.round(2))

# Age bands
df["AGE_GROUP"] = pd.cut(
    df["AGE_YEARS"],
    bins=[18, 25, 30, 35, 40, 50, 60, 100],
    labels=[
        "18-25",
        "26-30",
        "31-35",
        "36-40",
        "41-50",
        "51-60",
        "60+"
    ]
)

age_default = (
    df.groupby("AGE_GROUP", observed=True)["TARGET"]
    .agg(["count", "mean"])
)

age_default["default_rate_pct"] = age_default["mean"] * 100

print("\nDefault rate by age group:")
print(age_default[["count", "default_rate_pct"]].round(2))

plt.figure(figsize=(9, 5))

age_default["default_rate_pct"].plot(kind="bar")

plt.title("Default Rate by Applicant Age")
plt.xlabel("Age Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_default_by_age.png",
    dpi=200
)
plt.close()


# ============================================================
# 3. INCOME VS DEFAULT
# ============================================================

print("\n" + "=" * 80)
print("3. INCOME VS DEFAULT")
print("=" * 80)

# Use quantile-based income groups to reduce distortion
df["INCOME_GROUP"] = pd.qcut(
    df["AMT_INCOME_TOTAL"],
    q=5,
    duplicates="drop"
)

income_default = (
    df.groupby("INCOME_GROUP", observed=True)["TARGET"]
    .agg(["count", "mean"])
)

income_default["default_rate_pct"] = income_default["mean"] * 100

print("\nDefault rate by income group:")
print(income_default[["count", "default_rate_pct"]].round(2))

plt.figure(figsize=(10, 5))

income_default["default_rate_pct"].plot(kind="bar")

plt.title("Default Rate by Income Group")
plt.xlabel("Income Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "03_default_by_income.png",
    dpi=200
)
plt.close()


# ============================================================
# 4. CREDIT AMOUNT VS DEFAULT
# ============================================================

print("\n" + "=" * 80)
print("4. CREDIT AMOUNT VS DEFAULT")
print("=" * 80)

df["CREDIT_GROUP"] = pd.qcut(
    df["AMT_CREDIT"],
    q=5,
    duplicates="drop"
)

credit_default = (
    df.groupby("CREDIT_GROUP", observed=True)["TARGET"]
    .agg(["count", "mean"])
)

credit_default["default_rate_pct"] = credit_default["mean"] * 100

print("\nDefault rate by credit amount group:")
print(credit_default[["count", "default_rate_pct"]].round(2))

plt.figure(figsize=(10, 5))

credit_default["default_rate_pct"].plot(kind="bar")

plt.title("Default Rate by Loan Credit Amount")
plt.xlabel("Credit Amount Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_default_by_credit_amount.png",
    dpi=200
)
plt.close()


# ============================================================
# 5. EXTERNAL CREDIT SCORES VS DEFAULT
# ============================================================

print("\n" + "=" * 80)
print("5. EXTERNAL CREDIT SCORES VS DEFAULT")
print("=" * 80)

external_sources = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

for col in external_sources:

    print(f"\n{col}")

    score_summary = (
        df.groupby("TARGET")[col]
        .agg(["mean", "median", "count"])
    )

    print(score_summary.round(4))

    # Quantile groups
    temp = df[[col, "TARGET"]].dropna().copy()

    temp["SCORE_GROUP"] = pd.qcut(
        temp[col],
        q=5,
        duplicates="drop"
    )

    score_default = (
        temp.groupby("SCORE_GROUP", observed=True)["TARGET"]
        .mean() * 100
    )

    print("\nDefault rate by score group:")
    print(score_default.round(2))

    plt.figure(figsize=(9, 5))

    score_default.plot(kind="bar")

    plt.title(f"Default Rate by {col}")
    plt.xlabel("Credit Score Group")
    plt.ylabel("Default Rate (%)")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"05_{col.lower()}_default.png",
        dpi=200
    )
    plt.close()


# ============================================================
# 6. EMPLOYMENT VS DEFAULT
# ============================================================

print("\n" + "=" * 80)
print("6. EMPLOYMENT VS DEFAULT")
print("=" * 80)

# Treat known anomalous placeholder as missing
df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(
    365243,
    np.nan
)

df["EMPLOYMENT_YEARS"] = (
    -df["DAYS_EMPLOYED_CLEAN"] / 365.25
)

# Negative values after cleaning are treated as unavailable
df.loc[
    df["EMPLOYMENT_YEARS"] < 0,
    "EMPLOYMENT_YEARS"
] = np.nan

employment_summary = (
    df.groupby("TARGET")["EMPLOYMENT_YEARS"]
    .agg(["mean", "median", "count"])
)

print("\nEmployment summary by target:")
print(employment_summary.round(2))

# Employment groups
df["EMPLOYMENT_GROUP"] = pd.cut(
    df["EMPLOYMENT_YEARS"],
    bins=[0, 1, 3, 5, 10, 20, 100],
    labels=[
        "<1 year",
        "1-3 years",
        "3-5 years",
        "5-10 years",
        "10-20 years",
        "20+ years"
    ]
)

employment_default = (
    df.groupby("EMPLOYMENT_GROUP", observed=True)["TARGET"]
    .agg(["count", "mean"])
)

employment_default["default_rate_pct"] = (
    employment_default["mean"] * 100
)

print("\nDefault rate by employment duration:")
print(
    employment_default[
        ["count", "default_rate_pct"]
    ].round(2)
)

plt.figure(figsize=(10, 5))

employment_default["default_rate_pct"].plot(kind="bar")

plt.title("Default Rate by Employment Duration")
plt.xlabel("Employment Duration")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "06_default_by_employment.png",
    dpi=200
)
plt.close()


# ============================================================
# 7. EDUCATION VS DEFAULT
# ============================================================

print("\n" + "=" * 80)
print("7. EDUCATION VS DEFAULT")
print("=" * 80)

education_default = (
    df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
    .agg(["count", "mean"])
)

education_default["default_rate_pct"] = (
    education_default["mean"] * 100
)

education_default = education_default.sort_values(
    "default_rate_pct",
    ascending=False
)

print("\nDefault rate by education:")
print(
    education_default[
        ["count", "default_rate_pct"]
    ].round(2)
)

plt.figure(figsize=(10, 6))

education_default["default_rate_pct"].plot(kind="bar")

plt.title("Default Rate by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "07_default_by_education.png",
    dpi=200
)
plt.close()


# ============================================================
# 8. FAMILY STATUS VS DEFAULT
# ============================================================

print("\n" + "=" * 80)
print("8. FAMILY STATUS VS DEFAULT")
print("=" * 80)

family_default = (
    df.groupby("NAME_FAMILY_STATUS")["TARGET"]
    .agg(["count", "mean"])
)

family_default["default_rate_pct"] = (
    family_default["mean"] * 100
)

family_default = family_default.sort_values(
    "default_rate_pct",
    ascending=False
)

print("\nDefault rate by family status:")
print(
    family_default[
        ["count", "default_rate_pct"]
    ].round(2)
)

plt.figure(figsize=(10, 6))

family_default["default_rate_pct"].plot(kind="bar")

plt.title("Default Rate by Family Status")
plt.xlabel("Family Status")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "08_default_by_family_status.png",
    dpi=200
)
plt.close()


# ============================================================
# 9. GENDER VS DEFAULT
# ============================================================

print("\n" + "=" * 80)
print("9. GENDER VS DEFAULT")
print("=" * 80)

gender_default = (
    df.groupby("CODE_GENDER")["TARGET"]
    .agg(["count", "mean"])
)

gender_default["default_rate_pct"] = (
    gender_default["mean"] * 100
)

print("\nDefault rate by gender:")
print(
    gender_default[
        ["count", "default_rate_pct"]
    ].round(2)
)

plt.figure(figsize=(7, 5))

gender_default["default_rate_pct"].plot(kind="bar")

plt.title("Default Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "09_default_by_gender.png",
    dpi=200
)
plt.close()


# ============================================================
# 10. INCOME-TO-CREDIT BURDEN
# ============================================================

print("\n" + "=" * 80)
print("10. INCOME-TO-CREDIT BURDEN")
print("=" * 80)

df["CREDIT_INCOME_RATIO"] = (
    df["AMT_CREDIT"] /
    df["AMT_INCOME_TOTAL"]
)

df["CREDIT_INCOME_GROUP"] = pd.qcut(
    df["CREDIT_INCOME_RATIO"],
    q=5,
    duplicates="drop"
)

burden_default = (
    df.groupby(
        "CREDIT_INCOME_GROUP",
        observed=True
    )["TARGET"]
    .agg(["count", "mean"])
)

burden_default["default_rate_pct"] = (
    burden_default["mean"] * 100
)

print("\nDefault rate by credit-to-income ratio:")
print(
    burden_default[
        ["count", "default_rate_pct"]
    ].round(2)
)

plt.figure(figsize=(10, 5))

burden_default["default_rate_pct"].plot(kind="bar")

plt.title("Default Rate by Credit-to-Income Ratio")
plt.xlabel("Credit-to-Income Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "10_default_by_credit_income_ratio.png",
    dpi=200
)
plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)

print(f"\nCharts saved to:")
print(OUTPUT_DIR.resolve())

print("\nGenerated charts:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  ✓ {file.name}")

print("\nNext step:")
print("Review the numerical results and identify the strongest business insights.")