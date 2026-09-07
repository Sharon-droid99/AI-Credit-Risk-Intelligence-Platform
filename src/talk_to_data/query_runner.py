import duckdb
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATABASE_PATH = Path("data/credit_risk.duckdb")
SCHEMA_PATH = Path("sql/schema.sql")


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    print("=" * 80)
    print("INITIALIZING DUCKDB")
    print("=" * 80)

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nCreating database...")

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    print("Executing schema...")

    schema = SCHEMA_PATH.read_text(
        encoding="utf-8"
    )

    connection.execute(schema)

    print("\nDatabase initialized successfully.")

    return connection


# ============================================================
# DATABASE INFORMATION
# ============================================================

def get_schema():

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        columns = connection.execute(
            """
            DESCRIBE applicants
            """
        ).fetchdf()

        return columns

    finally:

        connection.close()

# ============================================================
# EXECUTE QUERY
# ============================================================

def execute_query(sql):

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True
    )

    try:

        result = connection.execute(sql).fetchdf()

        return result

    finally:

        connection.close()


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    connection = initialize_database()

    print("\nChecking applicants table...")

    result = connection.execute(
        """
        SELECT
            COUNT(*) AS applicants,
            SUM(TARGET) AS defaults,
            AVG(TARGET) AS default_rate
        FROM applicants
        """
    ).fetchdf()

    print("\nBasic dataset statistics:")
    print(result.to_string(index=False))

    # Close the initialization connection before
    # opening another database connection.
    connection.close()

    print("\nChecking schema...")

    schema = get_schema()
    print(
        f"Applicants table columns: {len(schema)}"
    )

    print(
        schema.head(20).to_string(index=False)
    )

    connection.close()

    print("\n" + "=" * 80)
    print("DUCKDB TEST COMPLETE")
    print("=" * 80)
