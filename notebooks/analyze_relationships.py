from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def unique_count(file_name, column):
    """Count unique values in a column."""
    path = DATA_DIR / file_name

    df = pd.read_csv(
        path,
        usecols=[column]
    )

    return df[column].nunique()


def analyze_application_keys():
    print("\n" + "=" * 80)
    print("APPLICATION KEY ANALYSIS")
    print("=" * 80)

    train = pd.read_csv(
        DATA_DIR / "application_train.csv",
        usecols=["SK_ID_CURR", "TARGET"]
    )

    test = pd.read_csv(
        DATA_DIR / "application_test.csv",
        usecols=["SK_ID_CURR"]
    )

    print(f"Training applicants : {len(train):,}")
    print(f"Unique train IDs    : {train['SK_ID_CURR'].nunique():,}")

    print(f"Test applicants     : {len(test):,}")
    print(f"Unique test IDs     : {test['SK_ID_CURR'].nunique():,}")

    print(
        f"Duplicate train IDs : "
        f"{train['SK_ID_CURR'].duplicated().sum():,}"
    )

    print(
        f"Duplicate test IDs  : "
        f"{test['SK_ID_CURR'].duplicated().sum():,}"
    )


def analyze_bureau():
    print("\n" + "=" * 80)
    print("BUREAU RELATIONSHIP")
    print("=" * 80)

    bureau = pd.read_csv(
        DATA_DIR / "bureau.csv",
        usecols=["SK_ID_CURR", "SK_ID_BUREAU"]
    )

    print(f"Bureau rows: {len(bureau):,}")
    print(
        f"Unique applicants in bureau: "
        f"{bureau['SK_ID_CURR'].nunique():,}"
    )
    print(
        f"Unique bureau accounts: "
        f"{bureau['SK_ID_BUREAU'].nunique():,}"
    )

    records_per_customer = bureau.groupby(
        "SK_ID_CURR"
    ).size()

    print("\nBureau records per applicant:")
    print(records_per_customer.describe())


def analyze_bureau_balance():
    print("\n" + "=" * 80)
    print("BUREAU BALANCE RELATIONSHIP")
    print("=" * 80)

    balance = pd.read_csv(
        DATA_DIR / "bureau_balance.csv",
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE"]
    )

    print(f"Bureau balance rows: {len(balance):,}")
    print(
        f"Unique bureau accounts in balance: "
        f"{balance['SK_ID_BUREAU'].nunique():,}"
    )

    records_per_bureau = balance.groupby(
        "SK_ID_BUREAU"
    ).size()

    print("\nMonthly records per bureau account:")
    print(records_per_bureau.describe())


def analyze_previous_applications():
    print("\n" + "=" * 80)
    print("PREVIOUS APPLICATION RELATIONSHIP")
    print("=" * 80)

    previous = pd.read_csv(
        DATA_DIR / "previous_application.csv",
        usecols=["SK_ID_CURR", "SK_ID_PREV"]
    )

    print(f"Previous application rows: {len(previous):,}")
    print(
        f"Unique applicants: "
        f"{previous['SK_ID_CURR'].nunique():,}"
    )
    print(
        f"Unique previous applications: "
        f"{previous['SK_ID_PREV'].nunique():,}"
    )

    apps_per_customer = previous.groupby(
        "SK_ID_CURR"
    ).size()

    print("\nPrevious applications per applicant:")
    print(apps_per_customer.describe())


def analyze_previous_application_children():
    print("\n" + "=" * 80)
    print("PREVIOUS APPLICATION CHILD TABLES")
    print("=" * 80)

    previous_ids = pd.read_csv(
        DATA_DIR / "previous_application.csv",
        usecols=["SK_ID_PREV"]
    )

    previous_ids = set(previous_ids["SK_ID_PREV"])

    for file_name in [
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv"
    ]:

        path = DATA_DIR / file_name

        df = pd.read_csv(
            path,
            usecols=["SK_ID_PREV"]
        )

        unique_prev = df["SK_ID_PREV"].nunique()

        matched = df["SK_ID_PREV"].isin(
            previous_ids
        ).sum()

        print(f"\n{file_name}")
        print(f"Rows: {len(df):,}")
        print(f"Unique SK_ID_PREV: {unique_prev:,}")
        print(
            f"Rows with matching previous application: "
            f"{matched:,}"
        )


def main():
    print("=" * 80)
    print("HOME CREDIT RELATIONSHIP ANALYSIS")
    print("=" * 80)

    analyze_application_keys()
    analyze_bureau()
    analyze_bureau_balance()
    analyze_previous_applications()
    analyze_previous_application_children()


if __name__ == "__main__":
    main()