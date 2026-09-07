from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# Core Home Credit data files
CORE_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "previous_application.csv",
]


def inspect_file(file_path: Path) -> None:
    """Inspect one Home Credit dataset."""

    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    # Use chunking for very large datasets.
    # We first inspect the full file structure using the header.
    header = pd.read_csv(file_path, nrows=0)

    print(f"Columns: {len(header.columns):,}")
    print("\nColumn names:")
    print(header.columns.tolist())

    # Read a sample for datatype inspection.
    sample = pd.read_csv(file_path, nrows=10_000)

    print("\nData types:")
    print(sample.dtypes.value_counts())

    print("\nSample duplicate rows:")
    print(f"{sample.duplicated().sum():,}")

    print("\nMissing values in first 10,000 rows:")
    missing = sample.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        missing_percentage = (
            missing / len(sample) * 100
        ).round(2)

        missing_summary = pd.DataFrame({
            "missing_count": missing,
            "missing_percentage": missing_percentage
        })

        print(missing_summary.head(20))
    else:
        print("No missing values in sample.")

    if "TARGET" in sample.columns:
        print("\nTARGET distribution in sample:")
        print(sample["TARGET"].value_counts(dropna=False))

        print("\nTARGET percentage in sample:")
        print(
            (sample["TARGET"].value_counts(normalize=True) * 100)
            .round(2)
        )


def count_rows(file_path: Path) -> int:
    """Count rows without loading the complete dataset into memory."""

    rows = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=[0],
        chunksize=100_000
    ):
        rows += len(chunk)

    return rows


def main():

    print("=" * 80)
    print("HOME CREDIT DATASET INSPECTION")
    print("=" * 80)

    print(f"\nData directory:")
    print(DATA_DIR)

    print("\nCore files:")

    for filename in CORE_FILES:

        file_path = DATA_DIR / filename

        if file_path.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} -- NOT FOUND")

    for filename in CORE_FILES:

        file_path = DATA_DIR / filename

        if not file_path.exists():
            continue

        inspect_file(file_path)

        print("\nCounting rows...")

        row_count = count_rows(file_path)

        print(f"Total rows: {row_count:,}")


if __name__ == "__main__":
    main()